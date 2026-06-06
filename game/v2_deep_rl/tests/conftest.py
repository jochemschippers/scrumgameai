"""Configure sys.path so tests can import the backend services and engine modules."""
import sys
import types
from pathlib import Path

# game/v2_deep_rl/control_center/backend  — for `from services.X import Y`
BACKEND_DIR = Path(__file__).resolve().parents[1] / "control_center" / "backend"
# game/v2_deep_rl  — for engine packages (config, rl, game_runtime, ...)
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

    # Handle stub save.
    def _stub_save(obj, path, **kwargs):
        pass  # no-op by default; individual tests override via monkeypatch

    # Handle stub load.
    def _stub_load(path, map_location=None, weights_only=False, **kwargs):
        return {}  # no-op by default

    _torch_stub.save = _stub_save
    _torch_stub.load = _stub_load

    # Minimal nn stub
    _nn = types.ModuleType("torch.nn")

    # Group the state and behavior for fake module.
    class _FakeModule:
        # Initialize the instance from the supplied configuration.
        def __init__(self, *args, **kwargs): pass
        # Initialize subclass.
        def __init_subclass__(cls, **kwargs): pass
        # Handle parameters.
        def parameters(self): return iter([])
        # Handle state dict.
        def state_dict(self): return {}
        # Load state dict.
        def load_state_dict(self, sd): pass
        # Handle eval.
        def eval(self): return self
        # Handle train.
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

# Implement the fake dqnagent's decision behavior.
class _FakeDQNAgent:
    # Initialize the instance from the supplied configuration.
    def __init__(self, *a, **kw):
        self.state_dim = kw.get("state_dim", 30)
        self.num_actions = kw.get("num_actions", 8)
        self.device = "cpu"
        self.training_steps = 0

        # Group the state and behavior for fake buf.
        class _FakeBuf:
            # Handle state dict.
            def state_dict(self): return {}
            # Load state dict.
            def load_state_dict(self, s): pass
            # Return the number of stored items.
            def __len__(self): return 0

        self.replay_buffer = _FakeBuf()

        # Group the state and behavior for fake net.
        class _FakeNet:
            # Handle state dict.
            def state_dict(self): return {}
            # Load state dict.
            def load_state_dict(self, sd, strict=True): pass
            # Handle eval.
            def eval(self): return self
            # Handle train.
            def train(self): return self

        self.policy_network = _FakeNet()
        self.target_network = _FakeNet()

    # Handle training state dict.
    def training_state_dict(self):
        return {"optimizer_state_dict": {}, "replay_buffer": {}, "training_steps": 0}

    # Load training state dict.
    def load_training_state_dict(self, state, include_replay=True): pass

# Handle fake encode state.
def _fake_encode_state(state, env): return []

if "rl.dqn_agent" not in sys.modules:
    sys.modules["rl.dqn_agent"] = _make_stub("rl.dqn_agent", {
        "DQNAgent": _FakeDQNAgent,
        "encode_state": _fake_encode_state,
    })

if "game_runtime.scrum_game_env" not in sys.modules:
    # Group the state and behavior for fake env.
    class _FakeEnv:
        num_actions = 8
        turns_with_loan = 0
        # Reset reset.
        def reset(self, seed=None): return {}
        # Advance step.
        def step(self, action): return {}, 0, True, {}

    sys.modules["game_runtime.scrum_game_env"] = _make_stub("game_runtime.scrum_game_env", {
        "ScrumGameEnv": _FakeEnv,
    })
