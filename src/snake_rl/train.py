"""
train.py — SB3 PPO training loop.
All hyperparameters come from config/default.yaml — nothing is hardcoded here.
"""

from __future__ import annotations

import io
import shutil
import threading
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import safe_mean

from snake_rl.env import SnakeEnv
from snake_rl.policy import SnakeMLP

# ---------------------------------------------------------------------------
# Training state (shared between training thread and Streamlit UI)
# ---------------------------------------------------------------------------


class TrainingState:
    """Thread-safe container for live training metrics and control flags."""

    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []
        self.status: str = "idle"
        self.current_timesteps: int = 0
        self.total_timesteps: int = 0
        self.latest_frame: np.ndarray | None = None
        self.stop_requested: bool = False
        self.error_msg: str = ""
        self._lock = threading.Lock()

    def append_metrics(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self.metrics.append(entry)

    def get_metrics_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.metrics)


# ---------------------------------------------------------------------------
# MLflow + live-metrics callback
# ---------------------------------------------------------------------------


class MLflowCallback(BaseCallback):
    """Log SB3 rollout/train metrics to MLflow and optionally to TrainingState."""

    # ep_rew_mean / ep_len_mean are logged by PPO *after* _on_rollout_end fires,
    # so they are not yet in logger.name_to_value at callback time — read them
    # directly from model.ep_info_buffer instead (see _on_rollout_end).
    _SB3_KEYS: dict[str, str] = {
        "value_loss": "train/value_loss",
        "policy_loss": "train/policy_gradient_loss",
        "entropy": "train/entropy_loss",
        "approx_kl": "train/approx_kl",
        "clip_fraction": "train/clip_fraction",
    }

    def __init__(
        self,
        training_state: TrainingState | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.training_state = training_state

    def _on_step(self) -> bool:
        if self.training_state is not None:
            self.training_state.current_timesteps = self.num_timesteps
        return True

    def _on_rollout_end(self) -> None:
        entry: dict[str, Any] = {"step": self.num_timesteps}
        for key, sb3_key in self._SB3_KEYS.items():
            if sb3_key in self.logger.name_to_value:
                val = float(self.logger.name_to_value[sb3_key])
                entry[key] = val
                mlflow.log_metric(key, val, step=self.num_timesteps)
        if len(self.model.ep_info_buffer) > 0:
            ep_rew = float(safe_mean([ep["r"] for ep in self.model.ep_info_buffer]))
            ep_len = float(safe_mean([ep["l"] for ep in self.model.ep_info_buffer]))
            entry["ep_rew_mean"] = ep_rew
            entry["ep_len_mean"] = ep_len
            mlflow.log_metric("ep_rew_mean", ep_rew, step=self.num_timesteps)
            mlflow.log_metric("ep_len_mean", ep_len, step=self.num_timesteps)
        if self.training_state is not None and len(entry) > 1:
            self.training_state.append_metrics(entry)


# ---------------------------------------------------------------------------
# Stop callback
# ---------------------------------------------------------------------------


class StopCallback(BaseCallback):
    """Signals SB3 to halt when training_state.stop_requested is set."""

    def __init__(self, training_state: TrainingState, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.training_state = training_state

    def _on_step(self) -> bool:
        return not self.training_state.stop_requested


# ---------------------------------------------------------------------------
# Periodic render callback
# ---------------------------------------------------------------------------


class RenderCallback(BaseCallback):
    """Run one episode every N completed episodes.

    When *training_state* is provided, renders headlessly and pushes the last
    frame to ``training_state.latest_frame`` (for Streamlit preview).
    Otherwise opens a persistent pygame window (CLI training).
    """

    def __init__(
        self,
        render_every: int,
        env_kwargs: dict,
        training_state: TrainingState | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.render_every = render_every
        self.env_kwargs = env_kwargs
        self.training_state = training_state
        self._episode_count = 0
        self._render_env = None  # persistent pygame window
        self._render_thread: threading.Thread | None = None

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self._episode_count += 1
                if (
                    self.render_every > 0
                    and self._episode_count % self.render_every == 0
                ):
                    if self.training_state is not None:
                        # Skip if a render thread is still running
                        if (
                            self._render_thread is None
                            or not self._render_thread.is_alive()
                        ):
                            buf = io.BytesIO()
                            self.model.save(buf)
                            buf.seek(0)
                            model_snapshot = PPO.load(buf)
                            self._render_thread = threading.Thread(
                                target=self._run_headless_episode,
                                args=(model_snapshot,),
                                daemon=True,
                            )
                            self._render_thread.start()
                    else:
                        self._run_render_episode()
        return True

    def _run_render_episode(self) -> None:
        if self._render_env is None:
            self._render_env = SnakeEnv(**self.env_kwargs, render_mode="human")
        obs, _ = self._render_env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self._render_env.step(int(action))
            done = terminated or truncated
            if not self._render_env.handle_events():
                self._render_env.close()
                self._render_env = None
                return
        # Leave window open — last frame stays visible until the next episode

    def _run_headless_episode(self, model) -> None:
        import time  # pylint: disable=import-outside-toplevel

        env = SnakeEnv(**self.env_kwargs, render_mode="rgb_array")
        obs, _ = env.reset()
        done = False
        while not done:
            frame = env.render()
            if frame is not None and self.training_state is not None:
                self.training_state.latest_frame = frame
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            time.sleep(0.05)
        # Show the final (collision) frame briefly before the thread exits
        frame = env.render()
        if frame is not None and self.training_state is not None:
            self.training_state.latest_frame = frame
        time.sleep(0.5)
        env.close()

    def _on_training_end(self) -> None:
        if self._render_env is not None:
            self._render_env.close()
            self._render_env = None


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(
    config_path: str = "config/default.yaml",
    training_state: TrainingState | None = None,
    continue_from: str | None = None,
) -> None:
    """Run a full PPO training session.

    Args:
        config_path:     Path to the YAML config file.
        training_state:  Optional shared state for Streamlit live updates.
        continue_from:   Path to a .zip model to continue training from.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    ppo_cfg = cfg["ppo"]
    env_cfg = cfg["env"]
    reward_cfg = cfg.get("reward", {})
    policy_cfg = cfg["policy"]
    mlflow_cfg = cfg["mlflow"]

    if training_state is not None:
        training_state.total_timesteps = train_cfg["total_timesteps"]
        training_state.status = "training"

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run() as run:
        for section, values in cfg.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    mlflow.log_param(f"{section}.{k}", v)

        env_kwargs = {
            "grid_w": env_cfg["grid_w"],
            "grid_h": env_cfg["grid_h"],
            "max_steps": env_cfg["max_steps"],
            "food_reward": reward_cfg.get("food", 10.0),
            "collision_penalty": reward_cfg.get("collision", -10.0),
            "toward_reward": reward_cfg.get("toward", 0.1),
            "away_penalty": reward_cfg.get("away", -0.3),
        }
        vec_env = make_vec_env(
            SnakeEnv,
            n_envs=train_cfg["n_envs"],
            env_kwargs=env_kwargs,
        )

        policy_kwargs = {
            "features_extractor_class": SnakeMLP,
            "features_extractor_kwargs": {
                "features_dim": policy_cfg["features_dim"],
                "hidden_dims": policy_cfg["hidden_dims"],
            },
        }

        if continue_from:
            model = PPO.load(continue_from, env=vec_env)
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=ppo_cfg["learning_rate"],
                n_steps=ppo_cfg["n_steps"],
                batch_size=ppo_cfg["batch_size"],
                n_epochs=ppo_cfg["n_epochs"],
                gamma=ppo_cfg["gamma"],
                clip_range=ppo_cfg["clip_range"],
                ent_coef=ppo_cfg["ent_coef"],
                policy_kwargs=policy_kwargs,
                verbose=1,
            )

        callbacks = [MLflowCallback(training_state=training_state)]
        render_every = train_cfg.get("render_every_n_episodes", 0)
        if render_every and render_every > 0:
            callbacks.append(
                RenderCallback(render_every, env_kwargs, training_state=training_state)
            )
        if training_state is not None:
            callbacks.append(StopCallback(training_state))

        model.learn(
            total_timesteps=train_cfg["total_timesteps"],
            callback=callbacks,
        )

        if training_state is not None:
            training_state.status = (
                "stopped" if training_state.stop_requested else "done"
            )

        save_dir = Path(train_cfg["save_path"])
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / train_cfg["model_name"]
        model.save(str(model_path))
        config_save_path = save_dir / f"{train_cfg['model_name']}_config.yaml"
        shutil.copy(config_path, config_save_path)

        mlflow.log_artifact(str(model_path) + ".zip")
        mlflow.log_artifact(str(config_save_path))

        print("\nTraining complete.")
        print(f"Model saved to: {model_path}.zip")
        print(f"MLflow run ID:  {run.info.run_id}")
        print(
            f"MLflow UI:      mlflow ui --backend-store-uri {mlflow_cfg['tracking_uri']}"
        )
