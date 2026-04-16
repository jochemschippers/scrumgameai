"""
Tests for dqn_agent.py — ReplayBuffer and DQNAgent.

These tests need the REAL torch and the REAL dqn_agent module.
If torch is not installed in the active venv the whole file is skipped
gracefully via pytest.importorskip.

NOTE: conftest.py installs a torch stub for service tests. Those tests are
collected first; by the time this module is imported the stub is already in
sys.modules. We explicitly remove the stub so the real torch (if available)
is imported instead.  If the real torch cannot be found the importorskip
guard will skip the module.
"""
from __future__ import annotations

import sys
import types

# ---- Remove conftest stubs before importing the real modules ---------------
# conftest.py inserts a minimal torch stub; if real torch is installed we must
# replace it before importing dqn_agent.
# Save the stub first so we can restore it if real torch is unusable.
_torch_stub = sys.modules.get("torch")
for _mod in list(sys.modules):
    if _mod == "torch" or _mod.startswith("torch."):
        del sys.modules[_mod]
if "dqn_agent" in sys.modules:
    del sys.modules["dqn_agent"]

# Skip the entire module gracefully when torch is absent or broken.
torch = pytest = None  # satisfy type-checkers below
import pytest  # noqa: E402 — must come after path manipulation

# PyTorch doesn't support Python 3.14+ yet; running on 3.14 causes a hard C abort.
if sys.version_info >= (3, 14):
    if _torch_stub is not None:
        sys.modules["torch"] = _torch_stub
    pytest.skip("torch not compatible with Python 3.14+ — skipping dqn_agent tests",
                allow_module_level=True)

try:
    import torch as _torch_check  # noqa: F401
except (ImportError, RuntimeError) as _e:
    # Restore the stub so other test modules (checkpoint_service, etc.) still work.
    if _torch_stub is not None:
        sys.modules["torch"] = _torch_stub
    pytest.skip(f"torch unavailable or broken ({_e}) — skipping dqn_agent tests",
                allow_module_level=True)

torch = pytest.importorskip("torch", reason="torch not installed — skipping dqn_agent tests")

# Now import the real dqn_agent (it lives one directory above the tests/ folder).
import importlib.util
from pathlib import Path

_ENGINE_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("dqn_agent", _ENGINE_DIR / "dqn_agent.py")
dqn_agent = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dqn_agent)

ReplayBuffer = dqn_agent.ReplayBuffer
DQNAgent = dqn_agent.DQNAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(state_dim: int = 4, num_actions: int = 3) -> DQNAgent:
    return DQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        learning_rate=1e-3,
        gamma=0.9,
        replay_capacity=1000,
        batch_size=8,
        target_update_frequency=100,
        device="cpu",
    )


def _dummy_state(dim: int = 4) -> list[float]:
    return [0.1] * dim


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_initial_length_is_zero(self):
        buf = ReplayBuffer(capacity=10)
        assert len(buf) == 0

    def test_push_increments_length(self):
        buf = ReplayBuffer(capacity=10)
        buf.push([0.0], 0, 1.0, [1.0], False)
        assert len(buf) == 1

    def test_push_respects_capacity(self):
        """Buffer must not grow beyond capacity (deque maxlen)."""
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push([float(i)], i % 3, float(i), [float(i + 1)], False)
        assert len(buf) == 5

    def test_sample_returns_correct_batch_size(self):
        buf = ReplayBuffer(capacity=50)
        for i in range(20):
            buf.push([float(i)], i % 3, float(i), [float(i + 1)], False)
        states, actions, rewards, next_states, dones = buf.sample(8)
        assert len(states) == 8
        assert len(actions) == 8
        assert len(rewards) == 8
        assert len(next_states) == 8
        assert len(dones) == 8

    def test_sample_raises_when_buffer_too_small(self):
        buf = ReplayBuffer(capacity=50)
        buf.push([0.0], 0, 0.0, [1.0], False)  # only 1 item
        with pytest.raises(ValueError):
            buf.sample(5)

    def test_state_dict_roundtrip_preserves_content(self):
        buf = ReplayBuffer(capacity=20)
        transitions = [
            ([float(i)], i % 2, float(i) * 0.1, [float(i + 1)], i % 2 == 0)
            for i in range(5)
        ]
        for t in transitions:
            buf.push(*t)

        sd = buf.state_dict()
        buf2 = ReplayBuffer(capacity=1)
        buf2.load_state_dict(sd)

        assert len(buf2) == len(buf)
        assert buf2.buffer.maxlen == 20
        # Spot-check first and last entry
        assert buf2.buffer[0] == buf.buffer[0]
        assert buf2.buffer[-1] == buf.buffer[-1]

    def test_state_dict_preserves_capacity(self):
        buf = ReplayBuffer(capacity=777)
        buf.push([1.0], 0, 0.5, [2.0], True)
        sd = buf.state_dict()
        assert sd["capacity"] == 777

    def test_load_state_dict_empty_buffer(self):
        buf = ReplayBuffer(capacity=10)
        buf.load_state_dict({"capacity": 10, "buffer": []})
        assert len(buf) == 0

    def test_load_state_dict_missing_capacity_uses_fallback(self):
        """A state dict without 'capacity' should not crash."""
        buf = ReplayBuffer(capacity=50)
        buf.load_state_dict({"buffer": [[0.0], 0, 1.0, [1.0], False]})
        # No exception means success; length may be 1 (list treated as one entry)
        # or 0 if the raw list is treated differently — just verify it doesn't crash.
        assert isinstance(len(buf), int)


# ---------------------------------------------------------------------------
# DQNAgent.choose_action
# ---------------------------------------------------------------------------

class TestDQNAgentChooseAction:
    def test_epsilon_one_always_random(self):
        """With epsilon=1.0 every action should be chosen randomly."""
        import random
        agent = _make_agent(state_dim=4, num_actions=5)
        state = _dummy_state(4)
        actions = {agent.choose_action(state, epsilon=1.0) for _ in range(200)}
        # With epsilon=1 and 200 draws over 5 actions we should see >1 action
        assert len(actions) > 1, "epsilon=1.0 must produce random actions, not always the same one"

    def test_epsilon_one_action_in_valid_range(self):
        agent = _make_agent(state_dim=4, num_actions=3)
        state = _dummy_state(4)
        for _ in range(50):
            a = agent.choose_action(state, epsilon=1.0)
            assert 0 <= a < 3

    def test_epsilon_zero_uses_network(self):
        """With epsilon=0 the agent always picks the greedy Q-value action."""
        agent = _make_agent(state_dim=4, num_actions=3)
        state = _dummy_state(4)
        # The network is randomly initialised but deterministic — same state
        # must return the same action on every call.
        first = agent.choose_action(state, epsilon=0.0)
        for _ in range(10):
            assert agent.choose_action(state, epsilon=0.0) == first, (
                "epsilon=0.0 must be deterministic (greedy)"
            )

    def test_epsilon_zero_action_in_valid_range(self):
        agent = _make_agent(state_dim=6, num_actions=8)
        state = _dummy_state(6)
        a = agent.choose_action(state, epsilon=0.0)
        assert 0 <= a < 8

    def test_epsilon_half_produces_variety(self):
        """epsilon=0.5 should occasionally explore and occasionally exploit."""
        agent = _make_agent(state_dim=4, num_actions=8)
        state = _dummy_state(4)
        actions = [agent.choose_action(state, epsilon=0.5) for _ in range(500)]
        # Expect at least 2 distinct actions over 500 trials with epsilon=0.5
        assert len(set(actions)) >= 2


# ---------------------------------------------------------------------------
# DQNAgent.training_state_dict / load_training_state_dict
# ---------------------------------------------------------------------------

class TestDQNAgentTrainingStateDict:
    def test_state_dict_has_required_keys(self):
        agent = _make_agent()
        sd = agent.training_state_dict()
        assert "optimizer_state_dict" in sd
        assert "replay_buffer" in sd
        assert "training_steps" in sd

    def test_training_steps_roundtrip(self):
        agent = _make_agent()
        agent.training_steps = 12345

        sd = agent.training_state_dict()
        agent2 = _make_agent()
        agent2.load_training_state_dict(sd)
        assert agent2.training_steps == 12345

    def test_replay_buffer_roundtrip(self):
        """Replay buffer content survives state_dict → load_training_state_dict."""
        agent = _make_agent(state_dim=4, num_actions=3)
        for i in range(15):
            agent.replay_buffer.push(
                [float(i)] * 4,
                i % 3,
                float(i) * 0.1,
                [float(i + 1)] * 4,
                False,
            )

        sd = agent.training_state_dict()
        agent2 = _make_agent(state_dim=4, num_actions=3)
        agent2.load_training_state_dict(sd, include_replay=True)
        assert len(agent2.replay_buffer) == 15

    def test_load_training_state_dict_skip_replay(self):
        """include_replay=False must not overwrite an existing buffer."""
        agent = _make_agent(state_dim=4, num_actions=3)
        for i in range(10):
            agent.replay_buffer.push([float(i)] * 4, 0, 0.0, [0.0] * 4, False)

        agent2 = _make_agent(state_dim=4, num_actions=3)
        # Pre-fill agent2 with different data
        for i in range(5):
            agent2.replay_buffer.push([99.0] * 4, 1, 1.0, [99.0] * 4, True)

        sd = agent.training_state_dict()
        agent2.load_training_state_dict(sd, include_replay=False)
        # Buffer in agent2 should be untouched (still 5 entries)
        assert len(agent2.replay_buffer) == 5

    def test_load_ignores_missing_optimizer_state(self):
        """A state dict without optimizer_state_dict must not crash."""
        agent = _make_agent()
        agent.load_training_state_dict({"training_steps": 99})
        assert agent.training_steps == 99

    def test_load_ignores_none_optimizer_state(self):
        agent = _make_agent()
        agent.load_training_state_dict({"optimizer_state_dict": None, "training_steps": 7})
        assert agent.training_steps == 7

    def test_optimizer_state_is_serializable(self):
        """The optimizer state dict must be picklable (torch.save compatible)."""
        import pickle
        agent = _make_agent()
        sd = agent.training_state_dict()
        # Just verify it can be pickled without error
        pickled = pickle.dumps(sd)
        restored = pickle.loads(pickled)
        assert restored["training_steps"] == sd["training_steps"]


# ---------------------------------------------------------------------------
# DQNAgent general
# ---------------------------------------------------------------------------

class TestDQNAgentGeneral:
    def test_initial_training_steps_zero(self):
        agent = _make_agent()
        assert agent.training_steps == 0

    def test_store_transition_adds_to_buffer(self):
        agent = _make_agent(state_dim=4, num_actions=3)
        assert len(agent.replay_buffer) == 0
        agent.store_transition([0.0] * 4, 1, 0.5, [1.0] * 4, False)
        assert len(agent.replay_buffer) == 1

    def test_train_step_returns_none_when_buffer_too_small(self):
        agent = _make_agent(state_dim=4, num_actions=3)
        # batch_size=8, buffer empty → must return None
        result = agent.train_step()
        assert result is None

    def test_train_step_returns_float_loss_when_buffer_full(self):
        agent = _make_agent(state_dim=4, num_actions=3)
        for i in range(20):
            agent.store_transition(
                [float(i)] * 4,
                i % 3,
                float(i) * 0.01,
                [float(i + 1)] * 4,
                False,
            )
        loss = agent.train_step()
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_step_increments_training_steps(self):
        agent = _make_agent(state_dim=4, num_actions=3)
        for i in range(20):
            agent.store_transition([float(i)] * 4, 0, 0.0, [0.0] * 4, False)
        agent.train_step()
        assert agent.training_steps == 1

    def test_networks_have_matching_architecture(self):
        """Policy and target networks must share the same weight shapes."""
        agent = _make_agent(state_dim=8, num_actions=5)
        policy_keys = set(agent.policy_network.state_dict().keys())
        target_keys = set(agent.target_network.state_dict().keys())
        assert policy_keys == target_keys
