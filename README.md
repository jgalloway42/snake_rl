# snake-rl

[![CI](https://github.com/jgalloway42/snake_rl/actions/workflows/ci.yml/badge.svg)](https://github.com/jgalloway42/snake_rl/actions/workflows/ci.yml)

A reinforcement learning agent trained to play Snake — built from scratch with a custom PyTorch policy network, Stable Baselines3 PPO, and a full experiment tracking pipeline.

> **Status:** Training in progress. Results and demo GIF coming soon.

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
| RL algorithm | Stable Baselines3 — PPO |
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
rendering.py     ← ALL pygame code lives here and only here.
train.py         ← SB3 PPO training loop. Wires together env + policy + MLflow.
streamlit_app.py ← Demo app. Loads trained model, renders live gameplay.
main.py          ← CLI entry point (snake-rl --train / --run).
```

The central design principle: **game logic, rendering, and RL are fully decoupled**. The test suite never touches pygame. The environment can run headlessly on any machine.

---

## Key engineering decisions

**Integer grid coordinates throughout.** The original snake.py used Python 3's float division (`480 / 20 = 24.0`) mixed with pixel arithmetic, creating subtle boundary bugs. The refactored engine uses integer grid units everywhere; pixel conversion happens only in the renderer.

**Relative actions (STRAIGHT / TURN_LEFT / TURN_RIGHT).** Rather than absolute directions, the agent outputs actions relative to its current heading. This keeps the action space at 3 (no illegal 180° reverse) and makes the policy's job simpler — it doesn't need to learn that "pressing left while heading left is invalid."

**3-channel spatial observation.** The environment returns a `(3, H, W)` tensor with separate channels for the snake's head, body, and food. This is clean, extensible (a CNN policy could swap in without changing the env contract), and avoids the ambiguity of encoding multiple entities in a single channel.

**Config-driven training.** All hyperparameters live in `config/default.yaml`. The training script reads them at runtime — no hardcoded values. Every training run logs its full config to MLflow, making experiments reproducible and comparable.

---

## Quickstart

```bash
# Install
python -m venv .venv && source .venv/bin/activate
make install

# Run tests
make test

# Train the agent from the CLI (5M timesteps by default)
# Every 50 episodes a pygame window shows the current agent playing a full game.
make train

# Launch the unified Streamlit app (Train + Play tabs)
make run
```

The Streamlit app has two tabs:

- **Train** — configure a run, start/stop training, watch real-time metric charts (episode reward, value loss, policy loss, entropy, KL divergence, episode length), see a live game preview every 50 episodes, and inspect the full config (reward weights, PPO params, environment settings) in a collapsible panel.
- **Play** — load a trained model and watch it play. Run single episodes or continuously, with a live leaderboard.

---

## Results

> _To be updated after training completes._

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
│   ├── rendering.py
│   ├── train.py
│   ├── streamlit_app.py
│   └── main.py
├── tests/              # pytest suite (90%+ coverage)
│   ├── test_core.py
│   ├── test_env.py
│   └── test_policy.py
├── config/
│   └── default.yaml    # All hyperparameters
├── models/             # Saved model artifacts (after training)
├── mlruns/             # MLflow experiment data (after training)
├── Makefile
└── pyproject.toml
```

---

## Improving the agent

The default config trains for 5M timesteps on a 24×24 grid — a reasonable starting point. Here are five options for getting a better agent:

**1. Continue from a checkpoint**

Training saves `models/snake_ppo.zip` after each run. In the Streamlit Train tab, paste the path into "Continue from model" to pick up where you left off. Equivalent CLI flag: `--config config/default.yaml` (training auto-saves; re-run `make train` after loading the checkpoint manually via `PPO.load()`).

**2. Longer runs**

Increase `training.total_timesteps` in `config/default.yaml`. Snake is a hard exploration problem — agents often don't start consistently finding food until ~500k steps, and don't chain multiple food items until 2M+.

```yaml
training:
  total_timesteps: 5_000_000
```

**3. Reward shaping**

All reward weights are in `config/default.yaml` under the `reward` key — no code changes needed:

```yaml
reward:
  food: 10.0        # reward for eating food
  collision: -10.0  # penalty for hitting a wall or the snake's own body
  toward: 0.1       # reward per step moving closer to food (Manhattan distance)
  away: -0.3        # penalty per step moving away from food
```

- Increase the food reward relative to collision penalty to encourage risk-taking.
- Reduce or remove the Manhattan distance shaping (`toward`/`away`) once the agent starts finding food reliably — it can discourage longer paths that eventually reach food.
- Add a small survival bonus to encourage longer episodes (requires a code change to `env.py`).

**4. Curriculum: start on a smaller grid**

A 10×10 grid is much easier to explore than 24×24. Train to convergence on the small grid, then transfer:

```yaml
env:
  grid_w: 10
  grid_h: 10
  max_steps: 200
```

Load the small-grid checkpoint and continue training on the full 24×24 grid. The policy generalises better than training from scratch on the large grid.

**5. Swap to a CNN policy**

The current MLP flattens the 3-channel spatial observation. A CNN can exploit the 2-D structure. To switch, replace `SnakeMLP` in `policy.py` with a `NatureCNN`-style extractor and update the `policy_kwargs` in `train.py`. No changes to the environment are needed — the `(3, H, W)` observation shape is already CNN-friendly.

---

## License

MIT
