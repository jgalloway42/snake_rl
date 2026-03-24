# snake-rl

[![CI](https://github.com/jgalloway42/snake_rl/actions/workflows/ci.yml/badge.svg)](https://github.com/jgalloway42/snake_rl/actions/workflows/ci.yml)

A reinforcement learning agent trained to play Snake — built from scratch with a custom PyTorch policy network, Stable Baselines3 DQN, and a full experiment tracking pipeline.

> **Status:** Switched to DQN after 4 PPO runs failed to converge. See [Training history](#training-history) for details.

---

## What this project demonstrates

- **Custom PyTorch `nn.Module`** integrated with SB3 via `BaseFeaturesExtractor` — the network architecture is hand-rolled, not auto-generated
- **Clean separation of concerns** — game logic, RL environment, rendering, and training are fully decoupled modules with zero cross-contamination
- **Gymnasium-compliant environment** built on a headless, bug-fixed game engine — testable without a display
- **Config-driven hyperparameters** via YAML — no magic numbers in training code
- **MLflow experiment tracking** — every run is logged with params, metrics, and saved artifacts
- **90%+ test coverage** on all core logic (game engine, environment, policy network)

---

## Tech stack

| Layer | Technology |
|---|---|
| RL algorithm | Stable Baselines3 — DQN (Double DQN) |
| Policy network | PyTorch (custom MLP) |
| RL environment | Gymnasium |
| Game engine | Pure Python (headless) |
| Rendering | pygame |
| Experiment tracking | MLflow |
| Demo app | Streamlit |
| Testing | pytest + pytest-cov |

---

## Architecture

```
core.py          ← Pure Python game logic. Zero pygame, zero torch.
env.py           ← Gymnasium wrapper. Imports core. No pygame. No torch.
policy.py        ← PyTorch nn.Module. No pygame. No gymnasium internals.
heuristic.py     ← Rule-based action selector used to seed the replay buffer.
rendering.py     ← ALL pygame code lives here and only here.
train.py         ← SB3 DQN training loop. Wires together env + policy + MLflow.
streamlit_app.py ← Demo app. Loads trained model, renders live gameplay.
main.py          ← CLI entry point (snake-rl --train / --run).
```

The central design principle: **game logic, rendering, and RL are fully decoupled**. The test suite never touches pygame. The environment can run headlessly on any machine.

---

## Key engineering decisions

**Relative actions (STRAIGHT / TURN_LEFT / TURN_RIGHT).** Rather than absolute directions, the agent outputs actions relative to its current heading. This keeps the action space at 3 (no illegal 180° reverse) and makes the policy's job simpler — it doesn't need to learn that "pressing left while heading left is invalid."

**3-channel spatial observation.** The environment returns a `(3, H, W)` tensor with separate channels for the snake's head, body, and food. This is clean, extensible (a CNN policy could swap in without changing the env contract), and avoids the ambiguity of encoding multiple entities in a single channel.

**Config-driven training.** All hyperparameters live in `config/default.yaml`. The training script reads them at runtime — no hardcoded values. Every training run logs its full config to MLflow, making experiments reproducible and comparable.

**DQN over PPO for sparse rewards.** After 4 PPO runs (20M total steps) failed to converge, the diagnosis was structural: PPO is on-policy and discards experience after each rollout. With food sparsely placed on a 16×16 grid, too few food-eating transitions appear per rollout for reliable credit assignment. DQN's replay buffer lets rare positive events be replayed many times; Double DQN reduces Q-value overestimation of the "safe wandering" state.

---

## Quickstart

```bash
# Install
python -m venv .venv && source .venv/bin/activate
make install

# Run tests
make test

# Train the agent from the CLI (5M timesteps by default)
# Every 24 episodes a preview renders the current agent playing a full game.
make train

# Launch the unified Streamlit app (Train + Play tabs)
make run
```

The Streamlit app has two tabs:

- **Train** — configure a run, start/stop training, watch real-time metric charts (episode reward, TD loss, exploration rate, episode length), see a live game preview every 24 episodes, and inspect the full config in a collapsible panel.
- **Play** — load a trained model and watch it play. Run single episodes or continuously, with a live leaderboard.

---

## Training history

Four PPO runs (5M steps each) failed to converge before switching to DQN.

| Run | ent_coef | toward | Outcome | Root cause |
|-----|----------|--------|---------|------------|
| 1 | 0.05 | 0.10 | Wandered, never ate | Shaping reward ≈ food reward; agent optimised shaping |
| 2 | 0.05 | 0.01 | Same wandering | collision=-8 caused risk-aversion before food discovery |
| 3 | 0.10 | 0.01 | Policy stayed random | ent_coef too high; entropy bonus overwhelmed policy gradient |
| 4 | 0.02 | 0.0 | Reward flat at 2–5 | **Entropy correct, still failed** — PPO structural limit |

Run 4 was the decisive experiment: entropy landed in the productive range (-0.5 to -0.7) yet reward never trended upward. This ruled out entropy tuning as the root cause and confirmed the issue is algorithmic. See `notebooks/run_analysis.ipynb` for the full analysis.

---

## Results

> _To be updated after DQN training completes._

| Metric | Value |
|---|---|
| Mean episode reward | — |
| Mean episode length | — |
| Best score | — |
| Total timesteps | 5,000,000 |

---

## Project structure

```
snake-rl/
├── src/snake_rl/       # Package source
│   ├── core.py
│   ├── env.py
│   ├── policy.py
│   ├── heuristic.py
│   ├── rendering.py
│   ├── train.py
│   ├── streamlit_app.py
│   └── main.py
├── tests/              # pytest suite (90%+ coverage)
│   ├── test_core.py
│   ├── test_env.py
│   ├── test_policy.py
│   └── test_heuristic.py
├── config/
│   └── default.yaml    # All hyperparameters
├── notebooks/
│   └── run_analysis.ipynb  # Training run analysis and cross-run comparison
├── models/             # Saved model artifacts (after training)
├── mlruns/             # MLflow experiment data (after training)
├── Makefile
└── pyproject.toml
```

---

## Improving the agent

**Continue from a checkpoint**

Training saves `models/snake_dqn.zip` after each run. In the Streamlit Train tab, paste the path into "Continue from model" to pick up where you left off.

> **Note:** `continue_from` preserves the model's saved parameters exactly. Config changes (reward weights, buffer size, etc.) only take effect on a **fresh** training run — they are ignored when continuing from a checkpoint.

**Heuristic replay buffer pre-fill**

Set `dqn.heuristic_prefill_steps` in `config/default.yaml` (default: 5000) to seed the replay buffer with rule-based transitions before RL training begins. The heuristic avoids immediate wall and self-collisions and moves toward food. This gives DQN a much better initial data distribution than purely random exploration.

```yaml
dqn:
  heuristic_prefill_steps: 5000  # 0 = disabled
```

Pre-fill is skipped when continuing from a checkpoint (`continue_from`), since the buffer would be reset anyway.

**Longer runs**

Increase `training.total_timesteps` in `config/default.yaml`. DQN with a replay buffer needs time to fill the buffer and propagate Q-values. The first ~10k steps (`learning_starts`) are pure random exploration.

**Reward shaping**

All reward weights are in `config/default.yaml` under the `reward` key:

```yaml
reward:
  food: 16.0        # reward for eating food
  collision: -2.0   # penalty for hitting a wall or the snake's own body
  toward: 0.0       # disabled — DQN handles credit assignment via Q-value propagation
  away: 0.0         # disabled
```

With DQN, distance shaping is generally not needed — the Q-function propagates the value of being near food backward through the replay buffer automatically via TD bootstrapping. If you do add shaping, keep it very small relative to `food` reward.

**Swap to a CNN policy**

The current MLP flattens the 3-channel spatial observation. A CNN can exploit the 2-D structure. To switch, replace `SnakeMLP` in `policy.py` with a `NatureCNN`-style extractor and update the `policy_kwargs` in `train.py`. No changes to the environment are needed — the `(3, H, W)` observation shape is already CNN-friendly.

**Prioritized Experience Replay (future)**

SB3's built-in DQN does not support PER, but it could be added via a custom `ReplayBuffer` subclass. PER would further amplify rare food-eating transitions during training — useful if the current DQN still struggles with sparse rewards.

---

## Convergence troubleshooting

Training metrics are logged to MLflow every 2048 steps. The Streamlit Train tab charts them live.

**DQN-specific metrics to watch:**

| Metric | What to look for |
|--------|-----------------|
| `ep_rew_mean` | Should trend upward past 16 (= 1 food/episode) within ~1M steps |
| `ep_len_mean` | Rising length with rising reward = snake is growing, not just surviving |
| `loss` (TD loss) | Should decrease as Q-function converges; persistent high loss = unstable training |
| `exploration_rate` | Should decay from 1.0 → 0.05 over `exploration_fraction` × total steps |

**Common failure patterns:**

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ep_rew_mean` flat, `exploration_rate` still high | Buffer hasn't seen enough food events yet | Wait — this is expected until epsilon decays |
| `ep_rew_mean` flat after epsilon has decayed | Q-function overfit to wandering | Try smaller `learning_rate` or larger `buffer_size` |
| `loss` very high and not decreasing | Reward scale too large (high MSE) | Halve `reward.food` |
| `ep_len_mean` very short throughout | Agent dies too fast; not exploring | Reduce `collision` penalty magnitude |
| `ep_rew_mean` spikes then collapses | Target network out of sync | Reduce `target_update_interval` |

**DQN parameter reference:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `learning_rate` | 0.0001 | Adam step size |
| `buffer_size` | 100000 | Replay buffer capacity; larger = more diverse batches |
| `learning_starts` | 10000 | Random steps before first gradient update |
| `batch_size` | 32 | Mini-batch sampled from replay buffer per update |
| `gamma` | 0.99 | Discount factor; high value for long-horizon tasks |
| `train_freq` | 4 | Gradient update every N env steps |
| `target_update_interval` | 1000 | Steps between hard target-network syncs |
| `exploration_fraction` | 0.3 | Fraction of total steps over which epsilon decays |
| `exploration_final_eps` | 0.05 | Minimum epsilon (floor exploration rate) |
| `heuristic_prefill_steps` | 5000 | Heuristic transitions pre-loaded into replay buffer before training; 0 = disabled |

---

## License

MIT
