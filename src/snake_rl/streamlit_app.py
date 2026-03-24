"""
streamlit_app.py — Unified snake-rl demo: Train + Play tabs.
Run via: streamlit run src/snake_rl/streamlit_app.py
"""

from __future__ import annotations

import threading
import time

import pandas as pd
import yaml
import streamlit as st
from stable_baselines3 import DQN

from snake_rl.env import SnakeEnv
from snake_rl.train import TrainingState
from snake_rl.train import train as run_training

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="snake-rl", layout="wide")

# ---------------------------------------------------------------------------
# Cross-refresh singleton — survives page refreshes (session_state resets,
# cache_resource does not). Used only for zombie thread detection/cleanup.
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_mgr():
    class _Mgr:
        ts: TrainingState | None = None
        thread: threading.Thread | None = None

    return _Mgr()


_mgr = _get_mgr()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "model" not in st.session_state:
    st.session_state.model = None
if "episode_count" not in st.session_state:
    st.session_state.episode_count = 0
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []
if "run_continuously" not in st.session_state:
    st.session_state.run_continuously = False
if "training_state" not in st.session_state:
    st.session_state.training_state = None
if "training_thread" not in st.session_state:
    st.session_state.training_thread = None
if "exit_requested" not in st.session_state:
    st.session_state.exit_requested = False

# Reconnect to any training thread that survived a page refresh.
if (
    st.session_state.training_thread is None
    and _mgr.thread is not None
    and _mgr.thread.is_alive()
):
    st.session_state.training_state = _mgr.ts
    st.session_state.training_thread = _mgr.thread


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_training() -> bool:
    thread = st.session_state.training_thread
    return thread is not None and thread.is_alive()


def _start_training(config_path: str, continue_from: str | None) -> None:
    # Stop any zombie thread that survived a page refresh.
    if _mgr.thread is not None and _mgr.thread.is_alive():
        if _mgr.ts is not None:
            _mgr.ts.stop_requested = True
        _mgr.thread.join(timeout=10)

    ts = TrainingState()
    _mgr.ts = ts
    st.session_state.training_state = ts

    def _run() -> None:
        try:
            run_training(
                config_path=config_path,
                training_state=ts,
                continue_from=continue_from,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            ts.status = "error"
            ts.error_msg = str(exc)

    thread = threading.Thread(target=_run, daemon=True)
    _mgr.thread = thread
    st.session_state.training_thread = thread
    thread.start()


def _on_play_exit() -> None:
    """on_click callback for the Exit button."""
    st.session_state.run_continuously = False


def _record_result(result_score: int, result_steps: int, summary_ph) -> None:
    st.session_state.episode_count += 1
    st.session_state.leaderboard.append((result_score, result_steps))
    st.session_state.leaderboard.sort(key=lambda x: x[0], reverse=True)
    summary_ph.success(
        f"Episode {st.session_state.episode_count} — "
        f"Score: **{result_score}** | Steps: **{result_steps}**"
    )


def _make_chart_df(data: list[dict], metric_key: str) -> pd.DataFrame | None:
    """Extract a metric series and return a smoothed + raw DataFrame."""
    rows = [(m["step"], m[metric_key]) for m in data if metric_key in m]
    if not rows:
        return None
    steps, vals = zip(*rows)
    series = pd.Series(list(vals), index=list(steps), name="raw")
    smoothed = series.ewm(alpha=0.3).mean().rename("smoothed")
    return pd.DataFrame({"smoothed": smoothed, "raw": series})


def _env_from_model(model) -> SnakeEnv:
    """Construct a SnakeEnv whose grid matches the model's observation space.

    The obs space is flat: (3 * total_h * total_w + 4,).  Since the grid is
    always square we can recover total_side = sqrt((obs_size - 4) / 3), then
    playable_size = total_side - 2 (border cells on each side).
    """
    obs_size = model.observation_space.shape[0]
    total_side = int(((obs_size - 4) / 3) ** 0.5)
    playable = total_side - 2
    return SnakeEnv(grid_w=playable, grid_h=playable, render_mode="rgb_array")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_train, tab_play = st.tabs(["Train", "Play"])


# ===========================================================================
# TRAIN TAB
# ===========================================================================

with tab_train:
    # Title row — EXIT top-right
    t_title_col, t_exit_col = st.columns([5, 1])
    t_title_col.title("Train Agent")
    train_exit_btn = t_exit_col.button("Exit / Stop", key="train_exit", width="stretch")
    if train_exit_btn:
        if _mgr.ts is not None:
            _mgr.ts.stop_requested = True
        st.session_state.exit_requested = True
        st.rerun()

    # Status + progress
    state: TrainingState | None = st.session_state.training_state
    if state is not None:
        status_colors = {
            "idle": "gray",
            "training": "green",
            "done": "blue",
            "stopped": "orange",
            "error": "red",
        }
        color = status_colors.get(state.status, "gray")
        st.markdown(f"**Status:** :{color}[{state.status.upper()}]")
        if state.status == "error":
            st.error(state.error_msg)
        if state.total_timesteps > 0:
            progress = min(state.current_timesteps / state.total_timesteps, 1.0)
            st.progress(
                progress,
                text=f"{state.current_timesteps:,} / {state.total_timesteps:,} steps",
            )

    # STABLE 2-COLUMN LAYOUT — always rendered so structure never shifts on rerun
    t_left, t_right = st.columns([1, 2])

    with t_left:
        st.subheader("Live Preview")
        if state is not None and state.latest_frame is not None:
            st.image(state.latest_frame, channels="RGB", width="stretch")
        else:
            st.caption("Waiting for first render episode...")

    with t_right:
        st.subheader("Training Metrics")
        chart_configs = [
            ("Episode Reward", "ep_rew_mean"),
            ("Episode Length", "ep_len_mean"),
            ("TD Loss", "loss"),
            ("Exploration Rate", "exploration_rate"),
        ]
        metrics = state.get_metrics_snapshot() if state is not None else []
        row1 = st.columns(4)
        for chart_col, (title, mkey) in zip(row1, chart_configs):
            chart_col.caption(title)
            chart_df = _make_chart_df(metrics, mkey) if metrics else None
            if chart_df is not None:
                chart_col.line_chart(chart_df, height=150)
            else:
                chart_col.markdown("&nbsp;")  # stable placeholder — keeps column height

    st.markdown("---")

    # Config inputs — stacked full-width
    t_config_path = st.text_input(
        "Config YAML", value="config/default.yaml", key="t_config"
    )
    t_continue_from = st.text_input(
        "Continue from model (optional)",
        value="",
        key="t_continue",
    )

    # Single toggle button: Start Training ↔ Stop Training ↔ Stopping...
    stopping = _is_training() and _mgr.ts is not None and _mgr.ts.stop_requested
    if stopping:
        if st.session_state.exit_requested:
            st.button(
                "Stopping — will exit when done...", disabled=True, width="stretch"
            )
        else:
            st.button("Stopping...", disabled=True, width="stretch")
    elif _is_training():
        if st.button("Stop Training", width="stretch"):
            if _mgr.ts is not None:
                _mgr.ts.stop_requested = True
            st.rerun()
    else:
        if st.button("Start Training", width="stretch"):
            _start_training(t_config_path, t_continue_from or None)
            st.rerun()

    # Config summary — loaded from the YAML path the user entered
    try:
        with open(t_config_path, encoding="utf-8") as _f:
            _cfg = yaml.safe_load(_f)
        with st.expander("Config", expanded=False):
            _r = _cfg.get("reward", {})
            _d = _cfg.get("dqn", {})
            _e = _cfg.get("env", {})
            _t = _cfg.get("training", {})
            st.markdown("**Reward shaping**")
            st.table(
                pd.DataFrame(
                    [
                        ("food", _r.get("food", 10.0), "Reward for eating food"),
                        (
                            "collision",
                            _r.get("collision", -10.0),
                            "Penalty for hitting wall or body",
                        ),
                        (
                            "toward",
                            _r.get("toward", 0.1),
                            "Reward per step moving closer to food",
                        ),
                        (
                            "away",
                            _r.get("away", -0.3),
                            "Penalty per step moving away from food",
                        ),
                    ],
                    columns=["key", "value", "description"],
                ).astype({"value": str})
            )
            st.markdown("**DQN**")
            st.table(
                pd.DataFrame(
                    [
                        (
                            "learning_rate",
                            _d.get("learning_rate"),
                            "Adam learning rate",
                        ),
                        (
                            "gamma",
                            _d.get("gamma"),
                            "Discount factor for future rewards",
                        ),
                        (
                            "buffer_size",
                            _d.get("buffer_size"),
                            "Replay buffer capacity (transitions)",
                        ),
                        (
                            "exploration_fraction",
                            _d.get("exploration_fraction"),
                            "Fraction of training for epsilon decay",
                        ),
                        (
                            "exploration_final_eps",
                            _d.get("exploration_final_eps"),
                            "Final epsilon (floor exploration rate)",
                        ),
                        (
                            "target_update_interval",
                            _d.get("target_update_interval"),
                            "Steps between target network syncs",
                        ),
                        ("batch_size", _d.get("batch_size"), "Mini-batch size"),
                    ],
                    columns=["key", "value", "description"],
                ).astype({"value": str})
            )
            st.markdown("**Environment**")
            st.table(
                pd.DataFrame(
                    [
                        (
                            "grid_w × grid_h",
                            f"{_e.get('grid_w')} × {_e.get('grid_h')}",
                            "Board size (border cells are walls)",
                        ),
                        (
                            "max_steps",
                            _e.get("max_steps"),
                            "Steps before episode is truncated",
                        ),
                        (
                            "total_timesteps",
                            _t.get("total_timesteps"),
                            "Training budget",
                        ),
                        ("n_envs", _t.get("n_envs"), "Parallel environments"),
                    ],
                    columns=["key", "value", "description"],
                ).astype({"value": str})
            )
    except (FileNotFoundError, TypeError):
        pass  # config path not yet valid — silently skip

    # Auto-refresh while training; exit cleanly once thread finishes
    if _is_training():
        time.sleep(0.1)
        st.rerun()
    elif st.session_state.exit_requested:
        import os  # pylint: disable=import-outside-toplevel

        os._exit(0)


# ===========================================================================
# PLAY TAB
# ===========================================================================

with tab_play:
    st.title("Snake RL — Live Demo")

    play_left, play_right = st.columns([1, 2])

    with play_left:
        st.subheader("Model")
        p_model_path = st.text_input(
            "Model path (.zip)", value="models/snake_dqn.zip", key="p_model_path"
        )
        if st.button("Load Model"):
            try:
                st.session_state.model = DQN.load(p_model_path)
                st.success(f"Loaded: {p_model_path}")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                st.error(f"Failed to load model: {exc}")

        render_fps = st.slider("Render FPS", min_value=1, max_value=30, value=10)

        st.markdown("---")
        st.markdown(f"**Episodes played:** {st.session_state.episode_count}")

        st.markdown("### Top 5 Scores")
        if st.session_state.leaderboard:
            for i, (rank_score, rank_steps) in enumerate(
                st.session_state.leaderboard[:5], 1
            ):
                st.markdown(f"{i}. Score **{rank_score}** — {rank_steps} steps")
        else:
            st.markdown("_No episodes yet._")

    with play_right:
        p_frame_ph = st.empty()
        p_score_ph = st.empty()
        p_summary_ph = st.empty()

        p_col1, p_col2, p_col3 = st.columns(3)
        run_episode_btn = p_col1.button(
            "Run Episode",
            disabled=st.session_state.model is None,
        )
        p_col2.toggle("Run Continuously", key="run_continuously")
        p_col3.button("Exit / Stop", key="play_exit", on_click=_on_play_exit)

        reward_chart_ph = st.empty()

        # --- Run a full episode in one blocking loop ---
        # Updating st.empty() placeholders in-place avoids a full page rerun
        # per frame, eliminating the ~100-300ms rerun overhead that caused
        # frame dropping with the previous step-per-rerun design.
        should_run = run_episode_btn or (
            st.session_state.run_continuously and st.session_state.model is not None
        )
        if should_run and st.session_state.model is not None:
            env = _env_from_model(st.session_state.model)
            obs, _ = env.reset()
            frame_delay = 1.0 / render_fps
            rewards: list[float] = []

            while True:
                frame = env.render()
                if frame is not None:
                    p_frame_ph.image(frame, channels="RGB", width=480)

                action, _ = st.session_state.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                rewards.append(float(reward))

                p_score_ph.markdown(
                    f"**Score:** {info['score']}  |  **Steps:** {info['steps']}"
                )
                reward_chart_ph.line_chart(
                    pd.DataFrame({"cumulative reward": pd.Series(rewards).cumsum()}),
                    height=180,
                )

                time.sleep(frame_delay)
                if terminated or truncated:
                    break

            env.close()
            _record_result(info["score"], info["steps"], p_summary_ph)

            if st.session_state.run_continuously:
                st.rerun()
