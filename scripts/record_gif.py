"""
record_gif.py — Record a GIF of the trained agent playing Snake.

Runs N episodes headlessly, keeps the one with the highest score,
and writes it to docs/agent.gif.

Usage:
    python scripts/record_gif.py [--model models/snake_dqn.zip]
                                 [--out scripts/figures/agent.gif]
                                 [--episodes 5] [--fps 10]
"""

import argparse
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from pathlib import Path

import numpy as np
from PIL import Image
from stable_baselines3 import DQN

from snake_rl.env import SnakeEnv


def run_episode(model, env) -> tuple[list[np.ndarray], int]:
    """Run one episode. Returns (frames, score)."""
    obs, _ = env.reset()
    frames = [env.render()]
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        frames.append(env.render())
        if terminated or truncated:
            return frames, info["score"]


def save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """Write a list of RGB frames to an animated GIF."""
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=int(1000 / fps),
        optimize=True,
    )


def record(model_path: str, out_path: Path, n_episodes: int, fps: int) -> Path:
    """Run episodes, keep the best, save to out_path. Returns out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = SnakeEnv(grid_w=16, grid_h=16, max_steps=300, render_mode="rgb_array")
    model = DQN.load(model_path, env=env)

    best_frames, best_score = [], -1
    for i in range(n_episodes):
        frames, score = run_episode(model, env)
        print(f"  Episode {i + 1}/{n_episodes}: score={score}, frames={len(frames)}")
        if score > best_score:
            best_frames, best_score = frames, score

    env.close()

    print(f"Best score: {best_score}  ({len(best_frames)} frames)")
    save_gif(best_frames, out_path, fps)
    print(f"Saved → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/snake_dqn.zip")
    parser.add_argument("--out", default="scripts/figures/agent.gif")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run; best score is kept")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    record(args.model, Path(args.out), args.episodes, args.fps)



if __name__ == "__main__":
    main()
