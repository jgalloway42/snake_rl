# Project Specification: `snake-rl`

## 0. Purpose of This Document

This spec is the single source of truth for implementing the `snake-rl` project.
It is written to be handed directly to a coding agent (Claude Code). Do not begin
implementation until this document has been read in full. Resolve any ambiguity
by following the explicit decisions recorded here rather than making assumptions.

---

## 1. Project Overview

Train a reinforcement learning agent to play Snake using:
- A **headless, bug-fixed refactor** of an existing `snake.py` as the game engine
- A **Gymnasium wrapper** exposing the game as an RL environment
- A **custom PyTorch `nn.Module`** as the policy network (MLP, hand-rolled)
- **Stable Baselines3 PPO** for the training loop (SB3 owns PPO; we own the network)
- **MLflow** for local experiment tracking
- A **Streamlit demo app** for watching the trained agent play
- A **configurable periodic renderer** (pygame) so the agent can be watched during training

The project is local-first. Cloud deployment (e.g., containerized Streamlit on AWS)
is a future concern; the architecture should not preclude it but need not implement it.

---

## 2. Bugs in the Original `snake.py` (Fix All of These)

The original file has several correctness issues that must be resolved in the refactor.
Do not carry any of these forward.

### Bug 1 — Mixed coordinate systems (most critical)
`GRID_WIDTH` and `GRID_HEIGHT` are floats (`480 / 20 = 24.0`) because Python 3
`/` always returns float. The code then mixes these floats with pixel coordinates
throughout, causing inconsistent boundary comparisons.

Specifically, `is_collision` checks:
```python
point[0] > SCREEN_WIDTH - GRID_WIDTH   # 480 - 24.0 = 456.0  ← pixel math
point[0] < GRID_WIDTH / 2              # 24.0 / 2 = 12.0     ← also pixel math
```
But `point` is computed in `move()` using pixel-scaled positions (`x * GRIDSIZE`),
so the right boundary is `460` (pixel of last valid cell), not `456`. The snake can
occupy the last column/row without triggering collision, or trigger early depending
on float rounding. **Fix: use integer grid coordinates throughout the core logic,
convert to pixels only in the renderer.**

### Bug 2 — Wrapping in `move()` masks wall collisions
```python
new = ((cur[0] + x*GRIDSIZE) % SCREEN_WIDTH, (cur[1] + y*GRIDSIZE) % SCREEN_HEIGHT)
```
The modulo wraps the position before `is_collision` is called, so the snake never
actually hits a wall — it teleports to the other side. But `is_collision` then
checks for wall collision on the already-wrapped coordinate, making the wall check
dead code. The original game appears to intend wall collision = death. **Fix: remove
the modulo; compute the new position linearly and let `is_collision` catch OOB.**

### Bug 3 — Food can spawn on float grid positions
`randomize_position` uses `GRID_WIDTH - 2` which is `22.0` (float), so
`random.randint(1, 22.0)` raises a `TypeError` in Python 3.  This works only if
the float happens to be a whole number and the Python version is lenient — it is
not reliable. **Fix: use explicit integer grid dimensions.**

### Bug 4 — Food can spawn on the snake
`randomize_position` picks a random grid cell with no awareness of where the snake
is. On a small board with a long snake this will spawn food inside the snake body.
**Fix: pass the snake's occupied cells to food placement and exclude them.**

### Bug 5 — `reset()` uses float center position
```python
self.positions = [(SCREEN_WIDTH/2, SCREEN_HEIGHT/2)]  # (240.0, 240.0)
```
This is a float tuple in a pixel coordinate. After the coordinate system fix, all
positions should be integer grid coordinates. **Fix: use integer grid-unit center.**

### Bug 6 — Score/length update happens outside the Snake class inconsistently
In `play_game()`, `snake.length += 1` and `snake.score += 1` are mutated directly
from outside the class, bypassing encapsulation. `update_high_score()` is then
called separately. **Fix: give `Snake` a `grow()` method that handles length,
score, and high-score update atomically. The game loop calls `snake.grow()`.**

---

## 3. Architecture

### 3.1 Separation of Concerns

The central design principle is that **game logic, rendering, and RL are fully
decoupled**:

```
core.py        ← pure Python, zero pygame, zero torch. Snake + Food logic only.
env.py         ← Gymnasium wrapper. Imports core. No pygame. No torch.
policy.py      ← PyTorch nn.Module. No pygame. No gymnasium internals.
rendering.py   ← ALL pygame code lives here and only here. Excluded from coverage.
train.py       ← SB3 PPO training loop. Wires together env + policy + MLflow.
streamlit_app.py ← Streamlit demo. Loads trained model, renders via rendering.py.
main.py        ← CLI entry point.
```

No module outside `rendering.py` should import `pygame`. The test suite should
never need pygame at all.

### 3.2 Package Structure

```
snake-rl/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── snake_rl/
│       ├── __init__.py
│       ├── core.py
│       ├── env.py
│       ├── policy.py
│       ├── rendering.py
│       ├── train.py
│       ├── streamlit_app.py
│       └── main.py
├── tests/
│   ├── test_core.py
│   ├── test_env.py
│   └── test_policy.py
├── config/
│   └── default.yaml
├── Makefile
├── pyproject.toml
└── README.md
```

---

## 4. Module Specifications

### 4.1 `core.py` — Headless Game Logic

**All coordinates are integer grid units, never pixels.**

#### Constants
```python
GRID_W: int = 24   # columns
GRID_H: int = 24   # rows
```

These are the default grid dimensions. The environment constructor must accept
`grid_w` and `grid_h` as parameters so tests and experiments can use smaller grids.

#### Direction
Use a `NamedTuple` with fields `dx: int, dy: int`. Define module-level constants:
`UP`, `DOWN`, `LEFT`, `RIGHT`. Define an `OPPOSITES` dict mapping each direction
to its reverse.

#### Action
Use an `IntEnum` with three values:
- `STRAIGHT = 0`
- `TURN_LEFT = 1`  (relative left from current heading)
- `TURN_RIGHT = 2` (relative right from current heading)

Provide a pure function `apply_action(direction, action) -> direction` that
applies a relative turn. This is what the environment calls each step.

Turn table (for reference, not pseudocode):
- Heading UP + TURN_LEFT → LEFT
- Heading UP + TURN_RIGHT → RIGHT
- (etc., all four headings × two turns)

#### `Snake` class
- `__init__(self, grid_w, grid_h)`: positions default to grid center, random direction
- `head` property: returns `positions[0]`
- `turn(direction)`: refuse 180° reversal when `length > 1`
- `step() -> bool`: move one cell in current direction. Return `True` on collision
  (caller handles reset), `False` otherwise. Does NOT wrap. Does NOT auto-reset.
- `grow()`: increment `length` and `score`, update `high_score` if exceeded
- `reset(start=None)`: single cell at `start` (default: grid center), random direction,
  score and length reset to 0 and 1. `high_score` is preserved across resets.
- `body_cells()`: return `positions[1:]`
- `_is_collision(point) -> bool`: OOB check + self-intersection check (body only,
  not current head position since head hasn't moved yet)

#### `Food` class
- `__init__(self, grid_w, grid_h)`: calls `randomize` immediately
- `randomize(occupied: set[tuple[int,int]])`: place food on a random free cell.
  Build the full list of free cells and sample from it. If the board is completely
  full (edge case), place at `(0, 0)`.

---

### 4.2 `env.py` — Gymnasium Wrapper

#### Observation Space
`Box(low=0, high=1, shape=(3, GRID_H, GRID_W), dtype=float32)`

Three channels:
- **Channel 0**: Head — single `1.0` at head grid position
- **Channel 1**: Body — `1.0` at each body cell
- **Channel 2**: Food — single `1.0` at food position

Despite using a spatial observation, the policy network is an MLP (see `policy.py`).
SB3 will flatten the observation before passing it to the network. This is
intentional — the 3-channel format is a clean, extensible representation that
would support a CNN swap later without changing the env contract.

#### Action Space
`Discrete(3)` mapping to `Action.STRAIGHT`, `Action.TURN_LEFT`, `Action.TURN_RIGHT`.

#### Reward Shaping
| Event | Reward |
|---|---|
| Ate food | `+10.0` |
| Collision (episode ends) | `-10.0` |
| Step closer to food (Manhattan) | `+0.1` |
| Step further from food | `-0.1` |

Distance is measured in grid units (Manhattan). Compute distance after move,
compare to distance before move.

#### Episode Termination
- `terminated = True` on collision
- `truncated = True` after `max_steps` steps (default: `500`)
- `max_steps` must be a constructor parameter

#### `render_mode`
Accept `render_mode` in the constructor per Gymnasium spec. Valid values:
`None`, `"human"`, `"rgb_array"`. Import `rendering.py` lazily inside the render
methods — never at module level — so the env can be instantiated headlessly.

#### Constructor signature
```python
def __init__(self, grid_w=24, grid_h=24, max_steps=500, render_mode=None)
```

#### `_get_info()` return dict
```python
{"score": int, "length": int, "steps": int}
```

---

### 4.3 `policy.py` — Custom PyTorch MLP

This is a hand-rolled `nn.Module` registered with SB3 as a custom policy.
The purpose of building this by hand is to get meaningful PyTorch experience —
don't let SB3 auto-generate the network.

#### Network Architecture

```
Input: flattened observation  (3 × 24 × 24 = 1728 floats)
→ Linear(1728, 256) + ReLU
→ Linear(256, 128)  + ReLU
→ Linear(128, 64)   + ReLU
→ output features (64-dim)   ← this is the "latent" representation SB3 uses
```

SB3 then attaches its own policy head (action logits) and value head on top of
the 64-dim features. To hook into SB3, implement this as a subclass of
`stable_baselines3.common.torch_layers.BaseFeaturesExtractor`.

Required:
```python
class SnakeMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        n_input = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs.flatten(start_dim=1))
```

Make `hidden_dims` a constructor parameter so the layer sizes are configurable
from `config/default.yaml` without code changes. Default: `[256, 128]` (plus the
final `features_dim=64` layer which is always present).

---

### 4.4 `rendering.py` — All Pygame Code

**This module is excluded from test coverage (see `pyproject.toml`).**

Responsibilities:
- `PygameRenderer(grid_w, grid_h, cell_size=20)`: initialises pygame window
- `draw(snake, food, score, episode, step)`: renders one frame. Displays score,
  episode number, and step count in the HUD.
- `get_rgb_array(snake, food, score) -> np.ndarray`: for `rgb_array` render mode
- `close()`: calls `pygame.quit()`
- `handle_events() -> bool`: pump pygame event queue, return `False` if quit requested

Grid visual spec (preserve from original):
- Border cells: white
- Interior cells: alternating dark/light checkerboard (`BG_COLOR_DARK`, `BG_COLOR_LIGHT`)
- Snake: green (`SNAKE_COLOR`)
- Food: yellow/gold (`FOOD_COLOR`)

---

### 4.5 `train.py` — Training Loop

#### Config-driven
All hyperparameters come from `config/default.yaml`. Load with PyYAML.
Do not hardcode any hyperparameter values in `train.py`.

#### Config file (`config/default.yaml`)
```yaml
training:
  total_timesteps: 1_000_000
  render_every_n_episodes: 50   # 0 or -1 = fully headless; 1 = every episode
  n_envs: 4                     # number of parallel envs for SB3
  save_path: "models/"
  model_name: "snake_ppo"

ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  clip_range: 0.2
  ent_coef: 0.01

env:
  grid_w: 24
  grid_h: 24
  max_steps: 500

policy:
  features_dim: 64
  hidden_dims: [256, 128]

mlflow:
  experiment_name: "snake-rl"
  tracking_uri: "mlruns"
```

#### MLflow Integration
- Call `mlflow.set_tracking_uri` and `mlflow.set_experiment` at start of training
- Log all config values as MLflow params at run start
- Use a SB3 callback to log `rollout/ep_rew_mean` and `rollout/ep_len_mean` to
  MLflow every `n_steps`
- Log the saved model as an MLflow artifact at the end of training
- Print the MLflow run ID and UI URL at completion

#### Periodic Rendering
Use a SB3 `BaseCallback`. Track episode count. When
`episode_count % render_every_n_episodes == 0` and `render_every_n_episodes > 0`:
- Instantiate a temporary single `SnakeEnv(render_mode="human")`
- Run one full episode with the current model
- Close the env (which closes the pygame window)

This gives a clean "checkpoint render" without keeping pygame open during training.
The user can set `render_every_n_episodes: 1` in config to see every episode.

#### Parallel Envs
Use `stable_baselines3.common.env_util.make_vec_env` with `n_envs` from config.
All envs in the vec env are headless (`render_mode=None`). The rendering callback
creates its own separate single env.

#### Model Save
Save the trained model to `{save_path}/{model_name}.zip` (SB3 format).
Also save the config YAML alongside it as `{save_path}/{model_name}_config.yaml`
so a saved model is always paired with the config that produced it.

---

### 4.6 `streamlit_app.py` — Demo App

#### Layout
- Sidebar: model path selector (file picker or text input), a "Load Model" button,
  speed slider (`render_fps`: 1–30, default 10), episode counter display
- Main area: live game frame rendered as `st.image` (use `rgb_array` render mode),
  real-time score and step count, "Run Episode" button, "Run Continuously" toggle

#### Behavior
- "Run Episode": run one full episode step by step, updating the displayed frame
  each step. Use `time.sleep(1/render_fps)` between steps.
- "Run Continuously": loop episodes until the toggle is turned off
- After each episode, display episode summary (score, steps, food eaten)
- Maintain a session-state leaderboard of top 5 episode scores in the sidebar

#### Model Loading
Load with `stable_baselines3.PPO.load(path)`. Display an error message (not a
crash) if the file doesn't exist or fails to load.

---

### 4.7 `main.py` — CLI Entry Point

```
snake-rl --train [--config path/to/config.yaml]
snake-rl --stream                        # run Streamlit app
```

`--train`: load config (default: `config/default.yaml`), call `train.py`
`--stream`: call `streamlit run` on `streamlit_app.py` as a subprocess

---

## 5. Build System

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "snake-rl"
version = "0.1.0"
description = "RL agent for Snake — custom PyTorch MLP + SB3 PPO"
requires-python = ">=3.10"
dependencies = [
    "gymnasium>=0.29",
    "stable-baselines3>=2.3",
    "torch>=2.2",
    "numpy>=1.26",
    "pygame>=2.5",
    "mlflow>=2.13",
    "pyyaml>=6.0",
    "streamlit>=1.35",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "black>=24.0",
    "pylint>=3.2",
]

[project.scripts]
snake-rl = "snake_rl.main:cli"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=snake_rl --cov-report=term-missing --cov-fail-under=90"

[tool.coverage.run]
source = ["snake_rl"]
omit = [
    "src/snake_rl/rendering.py",
    "src/snake_rl/streamlit_app.py",
    "src/snake_rl/main.py",
]

[tool.black]
line-length = 88
target-version = ["py310"]

[tool.pylint.messages_control]
disable = ["R", "C"]
```

---

## 6. Testing

### Philosophy
- 90%+ branch coverage on `core.py`, `env.py`, `policy.py`
- `rendering.py`, `streamlit_app.py`, and `main.py` are excluded (display/CLI code)
- Never import pygame in the test suite. Use `unittest.mock.patch` for any
  code path that would trigger a pygame import.
- Tests should be fast: no actual training, no actual rendering

### `tests/test_core.py` — Required Test Cases

**Snake movement:**
- Step in each of the four directions updates head position correctly
- Step into a wall returns `True` (collision)
- Step into body returns `True` (collision)
- Valid step returns `False`
- Step does not wrap around (confirm wrapping is removed)

**Turn logic:**
- Turn to a new direction is accepted
- 180° reversal is rejected when `length > 1`
- 180° reversal is accepted when `length == 1`

**grow():**
- Increments `length` by 1
- Increments `score` by 1
- Updates `high_score` when score exceeds it
- Does not update `high_score` when score is below it

**reset():**
- Length returns to 1
- Score returns to 0
- `high_score` is preserved
- Positions list has exactly one element

**apply_action():**
- All 12 combinations (4 headings × 3 actions) produce correct output direction

**Food:**
- `randomize()` never places food on an occupied cell
- `randomize()` handles a nearly-full board (only one free cell)

### `tests/test_env.py` — Required Test Cases

**Observation shape:**
- `reset()` returns obs with shape `(3, grid_h, grid_w)` and dtype `float32`
- `step()` returns obs with same shape

**Observation correctness:**
- Channel 0 has exactly one `1.0` at the head position
- Channel 1 has `1.0` at each body cell and `0.0` at head
- Channel 2 has exactly one `1.0` at the food position

**Reward:**
- Eating food yields `+10.0`
- Collision yields `-10.0`
- Moving toward food yields `+0.1`
- Moving away from food yields `-0.1`

**Termination:**
- `terminated=True` on collision
- `truncated=True` after `max_steps` steps
- Neither flag is set on a normal step

**Info dict:**
- Contains `score`, `length`, `steps` keys after reset and after step

**Gymnasium compliance:**
- `env.observation_space.contains(obs)` is True for all returned observations
- `check_env` from `stable_baselines3.common.env_checker` passes without warnings

### `tests/test_policy.py` — Required Test Cases

**Forward pass:**
- Output shape is `(batch_size, features_dim)` for a batch of random observations
- No NaN or Inf values in output
- Output changes between different inputs (sanity check: not a dead network)

**Configurability:**
- Network instantiates correctly with non-default `hidden_dims` and `features_dim`

---

## 7. CI Pipeline (`.github/workflows/ci.yml`)

Trigger: push and pull_request on `main`.

Steps:
1. Checkout
2. Set up Python 3.11
3. Install with `pip install -e ".[dev]"`
4. `make format` (check only — use `black --check`)
5. `make lint`
6. `make test` (with `SDL_VIDEODRIVER: dummy` and `SDL_AUDIODRIVER: dummy`
   set as env vars so pygame can initialise headlessly if anything triggers it)

---

## 8. Makefile

```makefile
.PHONY: install format lint test train stream clean

install:
	pip install -e ".[dev]"

format:
	black src/ tests/

lint:
	pylint --disable=R,C src/snake_rl/

test:
	SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy pytest tests/ -v

train:
	SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
	python -m snake_rl.main --train

stream:
	python -m snake_rl.main --stream

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/ .coverage mlruns/ models/
```

---

## 9. Future-Proofing Notes (Do Not Implement Now)

These are design constraints to keep in mind so the local implementation doesn't
create dead ends for later deployment.

- **No hardcoded local paths** in `streamlit_app.py` or `train.py`. Use relative
  paths or config values. This makes it straightforward to swap in S3 paths later.
- **Model saving uses SB3's `.zip` format** which is self-contained and portable.
  ONNX export can be added later for Lambda inference without touching the training code.
- **Config is YAML, not CLI args** (aside from the top-level `--train`/`--stream`).
  This makes it easy to inject config from environment variables or a secrets
  manager in a cloud context.
- **`SnakeEnv` is a standard Gymnasium env** with no local-only dependencies.
  It can be instantiated in a Lambda or container without modification.

---

## 10. Implementation Order (Suggested for Claude Code)

Implement in this order to ensure each layer is testable before building on top of it:

1. `core.py` + `tests/test_core.py` — get the game logic right and tested first
2. `env.py` + `tests/test_env.py` — build the Gymnasium wrapper on the solid core
3. `policy.py` + `tests/test_policy.py` — standalone PyTorch module, no env needed
4. `rendering.py` — pure display code, no tests needed
5. `train.py` + `config/default.yaml` — wire everything together
6. `streamlit_app.py` — demo layer on top of trained model
7. `main.py` — thin CLI wrapper
8. `Makefile` + `pyproject.toml` + `ci.yml` — build and CI scaffolding

---

## 11. Definition of Done

The implementation is complete when:
- [ ] `make install` succeeds from a clean virtualenv
- [ ] `make test` passes with ≥ 90% coverage and zero test failures
- [ ] `make lint` exits with code 0
- [ ] `make train` runs without error and produces a `.zip` model file and MLflow run
- [ ] `mlflow ui` shows the training run with logged params and metrics
- [ ] `make stream` launches the Streamlit app and the loaded model plays Snake
- [ ] CI pipeline passes on GitHub
