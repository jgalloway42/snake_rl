# Getting Started

## Install

```bash
python -m venv .venv && source .venv/bin/activate
make install
```

## Train

```bash
make train
```

Trains for 5 million steps by default. All hyperparameters are in `config/default.yaml`. Every run is logged to MLflow — view the experiment at `mlruns/`.

## Run the demo app

```bash
make run
```

Opens a Streamlit app with two tabs:

- **Train** — start and stop training, watch live metric charts and a game preview
- **Play** — load a trained model and watch it play

A trained model is saved to `models/snake_dqn.zip` after each run.

## Run tests

```bash
make test
```

## Project layout

```
config/default.yaml        # All hyperparameters
src/snake_rl/
  core.py                  # Game logic (no pygame, no torch)
  env.py                   # Gymnasium environment
  policy.py                # PyTorch Q-network
  heuristic.py             # Rule-based agent (replay buffer pre-fill)
  train.py                 # DQN training loop
  streamlit_app.py         # Demo app
models/                    # Saved model artifacts
mlruns/                    # MLflow experiment data
notebooks/run_analysis.ipynb  # Training run analysis
```
