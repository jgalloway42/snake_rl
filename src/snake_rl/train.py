"""
train.py — SB3 PPO training loop.
All hyperparameters come from config/default.yaml — nothing is hardcoded here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import mlflow
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from snake_rl.env import SnakeEnv
from snake_rl.policy import SnakeMLP

# ---------------------------------------------------------------------------
# MLflow callback
# ---------------------------------------------------------------------------


class MLflowCallback(BaseCallback):
    """Log SB3 rollout metrics to MLflow every rollout."""

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if "rollout/ep_rew_mean" in self.logger.name_to_value:
            mlflow.log_metric(
                "ep_rew_mean",
                self.logger.name_to_value["rollout/ep_rew_mean"],
                step=self.num_timesteps,
            )
        if "rollout/ep_len_mean" in self.logger.name_to_value:
            mlflow.log_metric(
                "ep_len_mean",
                self.logger.name_to_value["rollout/ep_len_mean"],
                step=self.num_timesteps,
            )


# ---------------------------------------------------------------------------
# Periodic render callback
# ---------------------------------------------------------------------------


class RenderCallback(BaseCallback):
    """Run one visible episode every N episodes during training.

    The pygame window stays open between episodes so the board remains visible.
    """

    def __init__(self, render_every: int, env_kwargs: dict, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.render_every = render_every
        self.env_kwargs = env_kwargs
        self._episode_count = 0
        self._render_env = None  # persistent — window stays open between episodes

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self._episode_count += 1
                if (
                    self.render_every > 0
                    and self._episode_count % self.render_every == 0
                ):
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

    def _on_training_end(self) -> None:
        if self._render_env is not None:
            self._render_env.close()
            self._render_env = None


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(config_path: str = "config/default.yaml") -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    ppo_cfg = cfg["ppo"]
    env_cfg = cfg["env"]
    policy_cfg = cfg["policy"]
    mlflow_cfg = cfg["mlflow"]

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

        callbacks = [MLflowCallback()]
        render_every = train_cfg.get("render_every_n_episodes", 0)
        if render_every and render_every > 0:
            callbacks.append(RenderCallback(render_every, env_kwargs))

        model.learn(
            total_timesteps=train_cfg["total_timesteps"],
            callback=callbacks,
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
