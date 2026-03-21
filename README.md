# snake-rl

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

# Train the agent (~1M timesteps)
# Launches MLflow UI in the background + starts training.
# Every 50 episodes a pygame window shows the current agent playing a full game.
make train

# Watch the trained agent play interactively (Streamlit app)
make run
```

---

## Results

> _To be updated after training completes._

| Metric | Value |
|---|---|
| Mean episode reward | — |
| Mean episode length | — |
| Best score | — |
| Total timesteps | 1,000,000 |

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

## License

MIT
