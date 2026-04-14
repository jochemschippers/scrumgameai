# Training Automation Plan

## Goal

Automate part of the training workflow for the Scrum Game RL system.

The automation should decide whether to:

- continue training unchanged
- continue training with adjusted hyperparameters
- stop training because learning has plateaued
- start a new run with changed game rules

## Recommendation

Start with a deterministic logic-based controller first.

An AI-backed controller can be added later, but only as a bounded policy selector. It should not freely rewrite rules or mutate configs without validation.

## Why This Approach

The current project already exposes the main signals needed for automation:

- training progress is written to `logs.csv`
- periodic evaluation is written to `evaluation_history.csv`
- training hyperparameters are centralized in `TrainingConfig`
- rule compatibility matters because rule changes can make checkpoints incompatible

Because of that, a rules-first automation layer is the safest and most practical first implementation.

## Existing Hook Points

Relevant code locations:

- `game/v2_deep_rl/train_dqn.py`
- `game/v2_deep_rl/config_manager.py`
- `game/v2_deep_rl/control_center/backend/jobs/`

Useful existing outputs:

- `reports/logs.csv`
- `reports/evaluation_history.csv`
- run metadata and checkpoints per run

## Phase 1: Logic-Based Automation

Train in blocks instead of one giant uninterrupted run.

After each block:

1. read recent training and evaluation metrics
2. compare recent evaluation windows
3. choose a bounded next action

Allowed next actions:

- continue unchanged
- lower learning rate
- extend exploration by increasing `epsilon_decay_episodes`
- stop training
- start a new run with a predefined rule variant

## Suggested Decision Rules

### Continue Unchanged

Continue if evaluation reward is still improving by roughly 2% to 5% over the last two evaluation windows.

### Lower Learning Rate

Lower `learning_rate` if:

- evaluation reward is improving
- but results are unstable or noisy
- and loss or reward variance stays high

### Extend Exploration

Increase `epsilon_decay_episodes` if:

- reward is mostly flat
- invalid actions remain high
- or bankruptcy stays high

This is useful when the policy appears under-explored rather than converged.

### Stop Training

Stop if:

- improvement stays below a small threshold
- that plateau lasts for 3 to 5 evaluation windows
- and bankruptcy is already low enough that more training is unlikely to change behavior much

### Propose Rule Changes

Only propose rule changes if:

- several training blocks have stalled
- hyperparameter adjustments did not help
- and the failure pattern looks structural rather than optimization-related

Examples:

- bankruptcy remains persistently high
- the agent collapses into a narrow action pattern
- reward remains poor across multiple seeds

## Rule Change Safety

Rule changes must be constrained.

They should not be generated arbitrarily because changing the game rules can alter:

- action space
- state encoding
- checkpoint compatibility
- training comparability between runs

Any rule changes should use predefined templates or validated mutations only.

## Phase 2: Optional NVIDIA AI Layer

An NVIDIA-hosted model can be added later as an advisory layer.

Use it only to choose among predefined actions.

Recommended model behavior:

- input: recent rows from `logs.csv`, recent rows from `evaluation_history.csv`, current training config, and a compact game config summary
- output: one of a fixed set such as `continue`, `lower_lr`, `extend_epsilon_decay`, `stop`, or `propose_rule_variant_A`

The model should not emit unrestricted config rewrites.

## NVIDIA Notes

Current official NVIDIA documentation indicates:

- NVIDIA Developer members can access hosted NIM endpoints for prototyping
- the LLM API supports an OpenAI-compatible chat-completions interface
- some catalog models are marked as free endpoints under trial-style terms

This makes NVIDIA reasonable for experimentation, but not the best foundation for the first control loop.

## Proposed Future Implementation

Add an automation module, for example:

- `game/v2_deep_rl/control_center/backend/services/training_autopilot.py`

Responsibilities:

- inspect completed run metrics
- classify training status
- choose the next bounded action
- write an automation decision record
- enqueue the next training job when appropriate

## Good First Deliverable

Implement a first version that:

1. reads the latest completed run
2. checks plateau, bankruptcy rate, and invalid action rate
3. decides whether to continue, adjust one hyperparameter, or stop
4. automatically queues the next job through the backend job system

## Long-Term Direction

Phase 1 should be deterministic and traceable.

Phase 2 can add an AI advisor.

Phase 3 can support AI-suggested rule variants, but only behind validation and human review.



ai key:
from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="abacusai/dracarys-llama-3.1-70b-instruct",
  messages=[{"role":"user","content":""}],
  temperature=0.5,
  top_p=1,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")