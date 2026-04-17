"""
Tests for checkpoint_service — specifically that it never calls torch.load
during listing or compatibility checks when a sidecar JSON is present.

Regression coverage for the bug where /checkpoints and /checkpoints/.../compatibility
hung indefinitely because torch.load was called on every best_scrum_model.pth
(which contains a ~100k-entry replay buffer).
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_METADATA = {
    "format_version": 2,
    "rule_signature": "abc123",
    "training_signature": "def456",
    "state_dim": 30,
    "num_actions": 8,
    "episode": 100_000,
    "average_reward": -3000.0,
    "legacy_checkpoint": False,
}


def _make_checkpoint(checkpoints_dir: Path, name: str, metadata: dict | None = None) -> Path:
    """Create a stub .pth and its sidecar .json in checkpoints_dir."""
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    pth = checkpoints_dir / name
    pth.write_bytes(b"stub-not-a-real-checkpoint")
    sidecar = pth.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata or FAKE_METADATA))
    return pth


def _make_run_with_best(runs_dir: Path, run_id: str, metadata: dict | None = None) -> Path:
    run_dir = runs_dir / run_id
    _make_checkpoint(run_dir / "checkpoints", "best_scrum_model.pth", metadata)
    return run_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_runs_dir(tmp_path, monkeypatch):
    """Point the checkpoint_service at a temp directory."""
    import services.checkpoint_service as svc
    # REPO_ROOT must be a parent of tmp_path so _checkpoint_id's relative_to works
    monkeypatch.setattr(svc, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(svc, "RUNS_DIR", tmp_path)
    # Clear the module-level cache before each test
    monkeypatch.setattr(svc, "_checkpoint_cache", None)
    monkeypatch.setattr(svc, "_checkpoint_cache_time", 0.0)
    monkeypatch.setattr(svc, "_checkpoint_cache_run_count", 0)
    # Disable non-run catalog dirs so only RUNS_DIR is scanned
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(svc, "CURRENT_CHECKPOINT_DIR", empty)
    monkeypatch.setattr(svc, "REFERENCE_V1_DIR", empty)
    monkeypatch.setattr(svc, "PLAYABLE_MODEL_V1_DIR", empty)
    yield tmp_path


# ---------------------------------------------------------------------------
# Tests: list_checkpoints never calls torch.load
# ---------------------------------------------------------------------------

class TestListCheckpointsNoTorchLoad:
    def test_sidecar_present_torch_not_called(self, _patch_runs_dir):
        """With a sidecar JSON, list_checkpoints must not call torch.load."""
        runs_dir = _patch_runs_dir
        _make_run_with_best(runs_dir, "run_001")

        import services.checkpoint_service as svc
        with patch("torch.load", side_effect=AssertionError("torch.load must not be called")) as mock_load:
            items = svc.list_checkpoints()
            mock_load.assert_not_called()

        assert len(items) == 1
        assert items[0]["checkpoint_type"] == "best"
        assert items[0]["compatibility_status"] == "tracked"
        assert items[0]["episode"] == 100_000
        assert items[0]["rule_signature"] == "abc123"

    def test_no_sidecar_returns_deferred_without_torch(self, _patch_runs_dir):
        """Without a sidecar, list_checkpoints returns deferred and never hangs."""
        runs_dir = _patch_runs_dir
        run_dir = runs_dir / "run_001"
        pth = run_dir / "checkpoints" / "best_scrum_model.pth"
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_bytes(b"stub")  # no sidecar

        import services.checkpoint_service as svc
        with patch("torch.load", side_effect=AssertionError("torch.load must not be called")):
            items = svc.list_checkpoints()

        assert len(items) == 1
        assert items[0]["compatibility_status"] == "deferred"

    def test_multiple_runs_all_use_sidecars(self, _patch_runs_dir):
        runs_dir = _patch_runs_dir
        for i in range(5):
            meta = {**FAKE_METADATA, "episode": (i + 1) * 10_000}
            _make_run_with_best(runs_dir, f"run_{i:03d}", meta)

        import services.checkpoint_service as svc
        with patch("torch.load", side_effect=AssertionError("torch.load must not be called")):
            items = svc.list_checkpoints()

        assert len(items) == 5
        assert all(item["compatibility_status"] == "tracked" for item in items)

    def test_latest_checkpoint_uses_sidecar(self, _patch_runs_dir):
        """latest_scrum_model.pth is also resolved via sidecar."""
        runs_dir = _patch_runs_dir
        run_dir = runs_dir / "run_001"
        _make_checkpoint(run_dir / "checkpoints", "latest_scrum_model.pth",
                         {**FAKE_METADATA, "episode": 150_000})

        import services.checkpoint_service as svc
        with patch("torch.load", side_effect=AssertionError("torch.load must not be called")):
            items = svc.list_checkpoints()

        latest = next(i for i in items if i["checkpoint_type"] == "latest")
        assert latest["episode"] == 150_000
        assert latest["compatibility_status"] == "tracked"

    def test_intermediate_checkpoint_deferred_without_torch(self, _patch_runs_dir):
        """Numbered episode checkpoints remain deferred (not torch-loaded)."""
        runs_dir = _patch_runs_dir
        run_dir = runs_dir / "run_001"
        pth = run_dir / "checkpoints" / "checkpoint_episode_100000.pth"
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_bytes(b"stub")

        import services.checkpoint_service as svc
        with patch("torch.load", side_effect=AssertionError("torch.load must not be called")):
            items = svc.list_checkpoints()

        assert items[0]["compatibility_status"] == "deferred"


# ---------------------------------------------------------------------------
# Tests: get_checkpoint_compatibility never calls torch.load
# ---------------------------------------------------------------------------

class TestCompatibilityNoTorchLoad:
    def test_sidecar_present_torch_not_called(self, _patch_runs_dir, monkeypatch):
        """Compatibility check with sidecar must not call torch.load."""
        runs_dir = _patch_runs_dir
        _make_run_with_best(runs_dir, "run_001")

        import services.checkpoint_service as svc

        # Stub _engine_imports so we don't need the real game engine
        fake_target_agent = _FakeAgent(state_dim=30, num_actions=8)
        monkeypatch.setattr(svc, "_engine_imports", lambda: (
            lambda *a, **kw: (fake_target_agent, None),  # build_agent_for_config
            None,                                         # load_checkpoint_payload
            lambda gc: "abc123",                          # compute_rule_signature
            None,                                         # load_game_config
        ))
        monkeypatch.setattr(svc, "_resolve_game_config_reference", lambda _: object())

        items = svc.list_checkpoints()
        checkpoint_id = items[0]["id"]

        with patch("torch.load", side_effect=AssertionError("torch.load must not be called")):
            result = svc.get_checkpoint_compatibility(checkpoint_id, "default_game_config")

        assert result["strict_resume_status"] == "compatible"
        assert result["shape_compatible"] is True

    def test_no_sidecar_still_returns_without_hanging(self, _patch_runs_dir, monkeypatch):
        """Without a sidecar, compatibility returns a result (unknown) rather than hanging."""
        runs_dir = _patch_runs_dir
        run_dir = runs_dir / "run_001"
        pth = run_dir / "checkpoints" / "best_scrum_model.pth"
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_bytes(b"stub")

        import services.checkpoint_service as svc

        fake_target_agent = _FakeAgent(state_dim=30, num_actions=8)
        monkeypatch.setattr(svc, "_engine_imports", lambda: (
            lambda *a, **kw: (fake_target_agent, None),
            None,
            lambda gc: "abc123",
            None,
        ))
        monkeypatch.setattr(svc, "_resolve_game_config_reference", lambda _: object())

        items = svc.list_checkpoints()
        checkpoint_id = items[0]["id"]

        # No torch.load should be called — returns unknown/null metadata
        with patch("torch.load", side_effect=AssertionError("torch.load must not be called")):
            result = svc.get_checkpoint_compatibility(checkpoint_id, "default_game_config")

        assert result["strict_resume_status"] == "legacy-unknown"


# ---------------------------------------------------------------------------
# Tests: resolve_checkpoint_path only returns tracked checkpoint files
# ---------------------------------------------------------------------------

class TestResolveCheckpointPath:
    def test_resolves_tracked_checkpoint_id(self, _patch_runs_dir):
        runs_dir = _patch_runs_dir
        checkpoint_path = _make_checkpoint(runs_dir / "run_001" / "checkpoints", "best_scrum_model.pth")

        import services.checkpoint_service as svc

        items = svc.list_checkpoints()
        resolved = svc.resolve_checkpoint_path(items[0]["id"])

        assert resolved == checkpoint_path.resolve()

    def test_rejects_untracked_path_traversal_id(self, _patch_runs_dir):
        runs_dir = _patch_runs_dir
        _make_checkpoint(runs_dir / "run_001" / "checkpoints", "best_scrum_model.pth")

        import services.checkpoint_service as svc

        with pytest.raises(ValueError, match="was not found"):
            svc.resolve_checkpoint_path("../../Windows/system32/config/SAM")


# ---------------------------------------------------------------------------
# Tests: sidecar written by save_checkpoint
# ---------------------------------------------------------------------------

class TestSidecarWritten:
    def test_save_checkpoint_writes_sidecar(self, tmp_path, monkeypatch):
        """save_checkpoint must write a .json sidecar alongside the .pth."""
        import checkpoint_utils

        # Mock torch.save so we don't need a real agent
        saved_payloads = {}

        def fake_torch_save(obj, path):
            saved_payloads[str(path)] = obj

        monkeypatch.setattr("torch.save", fake_torch_save)

        fake_agent = _FakeAgent(state_dim=10, num_actions=4)
        fake_game_config = _FakeGameConfig()
        fake_training_config = None

        pth_path = tmp_path / "checkpoints" / "best_scrum_model.pth"
        checkpoint_utils.save_checkpoint(
            pth_path,
            fake_agent,
            fake_game_config,
            fake_training_config,
            extra_metadata={"episode": 50_000},
        )

        sidecar = pth_path.with_suffix(".json")
        assert sidecar.exists(), "Sidecar .json was not written by save_checkpoint"

        meta = json.loads(sidecar.read_text())
        assert meta["episode"] == 50_000
        assert "rule_signature" in meta

    def test_sidecar_contains_average_reward_for_latest(self, tmp_path, monkeypatch):
        """latest_scrum_model.pth sidecar must include average_reward so the next
        continuation inherits best_average_reward correctly."""
        import checkpoint_utils

        def fake_torch_save(obj, path):
            pass

        monkeypatch.setattr("torch.save", fake_torch_save)

        fake_agent = _FakeAgent(state_dim=10, num_actions=4)
        fake_game_config = _FakeGameConfig()

        pth_path = tmp_path / "checkpoints" / "latest_scrum_model.pth"
        checkpoint_utils.save_checkpoint(
            pth_path,
            fake_agent,
            fake_game_config,
            extra_metadata={"episode": 100_000, "average_reward": -3000.0},
        )

        sidecar = pth_path.with_suffix(".json")
        meta = json.loads(sidecar.read_text())
        assert meta.get("average_reward") == -3000.0, (
            "latest_scrum_model.pth sidecar must carry average_reward so the next "
            "continuation inherits best_average_reward instead of starting at -inf"
        )


# ---------------------------------------------------------------------------
# Minimal stubs (avoid importing torch / game engine in test setup)
# ---------------------------------------------------------------------------

class _FakeReplayBuffer:
    def state_dict(self):
        return {}


class _FakeAgent:
    def __init__(self, state_dim=30, num_actions=8):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = "cpu"
        self.replay_buffer = _FakeReplayBuffer()
        self.training_steps = 0

    def training_state_dict(self):
        return {
            "optimizer_state_dict": {},
            "replay_buffer": {},
            "training_steps": 0,
        }

    class _FakeNetwork:
        def state_dict(self):
            return {}

    policy_network = _FakeNetwork()
    target_network = _FakeNetwork()


class _FakeGameConfig:
    def to_dict(self):
        return {}


# Patch compute_rule_signature and compute_training_signature used in save_checkpoint
import unittest.mock as _mock

_patch_rule_sig = _mock.patch(
    "checkpoint_utils.compute_rule_signature", return_value="test_rule_sig"
)
_patch_training_sig = _mock.patch(
    "checkpoint_utils.compute_training_signature", return_value="test_training_sig"
)


@pytest.fixture(autouse=True)
def _patch_config_manager():
    with _patch_rule_sig, _patch_training_sig:
        yield
