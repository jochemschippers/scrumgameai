"""Configure sys.path so tests can import the backend services and engine modules."""
import sys
import types
from pathlib import Path

# game/v2_deep_rl/control_center/backend  — for `from services.X import Y`
BACKEND_DIR = Path(__file__).resolve().parents[1] / "control_center" / "backend"
# game/v2_deep_rl  — for engine modules (config_manager, dqn_agent, …)
ENGINE_DIR = Path(__file__).resolve().parents[1]

for p in (str(BACKEND_DIR), str(ENGINE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Stub out torch so checkpoint_utils can be imported without a GPU/ML install.
# Tests that care about torch.save / torch.load mock them individually.
# ---------------------------------------------------------------------------
if "torch" not in sys.modules:
    _torch_stub = types.ModuleType("torch")

    def _stub_save(obj, path, **kwargs):
        pass  # no-op by default; individual tests override via monkeypatch

    def _stub_load(path, map_location=None, weights_only=False, **kwargs):
        return {}  # no-op by default

    _torch_stub.save = _stub_save
    _torch_stub.load = _stub_load

    # Minimal nn stub
    _nn = types.ModuleType("torch.nn")

    class _FakeModule:
        def __init__(self, *args, **kwargs): pass
        def __init_subclass__(cls, **kwargs): pass
        def parameters(self): return iter([])
        def state_dict(self): return {}
        def load_state_dict(self, sd): pass
        def eval(self): return self
        def train(self): return self

    _nn.Module = _FakeModule
    _nn.Linear = _FakeModule
    _nn.ReLU = _FakeModule
    _nn.Sequential = _FakeModule
    _nn.SmoothL1Loss = _FakeModule
    _torch_stub.nn = _nn
    _torch_stub.no_grad = lambda: __import__("contextlib").nullcontext()
    _torch_stub.device = lambda x: x

    sys.modules["torch"] = _torch_stub
    sys.modules["torch.nn"] = _nn

# Stub heavy engine modules that import torch at module level so they don't
# block tests that only care about file I/O and service logic.
def _make_stub(name, attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod

class _FakeDQNAgent:
    def __init__(self, *a, **kw):
        self.state_dim = kw.get("state_dim", 30)
        self.num_actions = kw.get("num_actions", 8)
        self.device = "cpu"
        self.training_steps = 0

        class _FakeBuf:
            def state_dict(self): return {}
            def load_state_dict(self, s): pass
            def __len__(self): return 0

        self.replay_buffer = _FakeBuf()

        class _FakeNet:
            def state_dict(self): return {}
            def load_state_dict(self, sd, strict=True): pass
            def eval(self): return self
            def train(self): return self

        self.policy_network = _FakeNet()
        self.target_network = _FakeNet()

    def training_state_dict(self):
        return {"optimizer_state_dict": {}, "replay_buffer": {}, "training_steps": 0}

    def load_training_state_dict(self, state, include_replay=True): pass

def _fake_encode_state(state, env): return []

if "dqn_agent" not in sys.modules:
    sys.modules["dqn_agent"] = _make_stub("dqn_agent", {
        "DQNAgent": _FakeDQNAgent,
        "encode_state": _fake_encode_state,
    })

if "scrum_game_env" not in sys.modules:
    class _FakeEnv:
        num_actions = 8
        turns_with_loan = 0
        def reset(self, seed=None): return {}
        def step(self, action): return {}, 0, True, {}

    sys.modules["scrum_game_env"] = _make_stub("scrum_game_env", {
        "ScrumGameEnv": _FakeEnv,
    })
