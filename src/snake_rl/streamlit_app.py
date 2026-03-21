"""
streamlit_app.py — Unified snake-rl demo: Train + Play tabs.
Run via: streamlit run src/snake_rl/streamlit_app.py
"""

from __future__ import annotations

import threading
import time

import pandas as pd
import streamlit as st
from stable_baselines3 import PPO

from snake_rl.env import SnakeEnv
from snake_rl.train import TrainingState
from snake_rl.train import train as run_training

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="snake-rl", layout="wide")


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_training() -> bool:
    thread = st.session_state.training_thread
    return thread is not None and thread.is_alive()


def _start_training(config_path: str, continue_from: str | None) -> None:
    ts = TrainingState()
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
    st.session_state.training_thread = thread
    thread.start()


def _run_episode(
    model: PPO,
    fps: int,
    frame_ph,  # st.empty()
    score_ph,  # st.empty()
) -> tuple[int, int]:
    """Run one full episode; stream frames into placeholders."""
    env = SnakeEnv(render_mode="rgb_array")
    obs, _ = env.reset()
    info: dict = {}
    done = False
    while not done:
        frame = env.render()
        if frame is not None:
            frame_ph.image(frame, channels="RGB", width=480)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        score_ph.markdown(f"**Score:** {info['score']}  |  **Steps:** {info['steps']}")
        done = terminated or truncated
        time.sleep(1 / fps)
    env.close()
    return info["score"], info["steps"]


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


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_train, tab_play = st.tabs(["Train", "Play"])


# ===========================================================================
# TRAIN TAB
# ===========================================================================

with tab_train:
    st.title("Train Agent")

    # Config inputs
    col_c1, col_c2 = st.columns(2)
    t_config_path = col_c1.text_input(
        "Config YAML", value="config/default.yaml", key="t_config"
    )
    t_continue_from = col_c2.text_input(
        "Continue from model (.zip, optional)", value="", key="t_continue"
    )

    # Control buttons
    col_b1, col_b2 = st.columns(2)
    start_btn = col_b1.button("Start Training", disabled=_is_training())
    stop_btn = col_b2.button("Stop Training", disabled=not _is_training())

    if start_btn:
        _start_training(t_config_path, t_continue_from or None)
        st.rerun()

    if stop_btn and st.session_state.training_state is not None:
        st.session_state.training_state.stop_requested = True

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

    # Live preview
    if state is not None:
        preview_col, _ = st.columns([1, 3])
        with preview_col:
            st.subheader("Live Preview")
            live_frame_ph = st.empty()
            if state.latest_frame is not None:
                live_frame_ph.image(state.latest_frame, channels="RGB", width=300)
            else:
                live_frame_ph.markdown("_Waiting for first render episode..._")

        # Metric charts
        metrics = state.get_metrics_snapshot()
        if metrics:
            st.subheader("Training Metrics")

            chart_configs = [
                ("Episode Reward", "ep_rew_mean"),
                ("Value Loss", "value_loss"),
                ("Policy Loss", "policy_loss"),
                ("Episode Length", "ep_len_mean"),
                ("Entropy", "entropy"),
                ("KL Divergence", "approx_kl"),
            ]

            top_cols = st.columns(3)
            bot_cols = st.columns(3)
            for idx, (title, mkey) in enumerate(chart_configs):
                col = top_cols[idx] if idx < 3 else bot_cols[idx - 3]
                chart_df = _make_chart_df(metrics, mkey)
                if chart_df is not None:
                    col.subheader(title)
                    col.line_chart(chart_df, height=200)

    # Auto-refresh while training — fast enough to animate render episodes
    if _is_training():
        time.sleep(0.1)
        st.rerun()


# ===========================================================================
# PLAY TAB
# ===========================================================================

with tab_play:
    st.title("Snake RL — Live Demo")

    play_left, play_right = st.columns([1, 2])

    with play_left:
        st.subheader("Model")
        p_model_path = st.text_input(
            "Model path (.zip)", value="models/snake_ppo.zip", key="p_model_path"
        )
        if st.button("Load Model"):
            try:
                st.session_state.model = PPO.load(p_model_path)
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
            "Run Episode", disabled=st.session_state.model is None
        )
        run_continuously = p_col2.toggle(
            "Run Continuously", value=st.session_state.run_continuously
        )
        exit_btn = p_col3.button("Exit / Stop")

        if exit_btn:
            st.session_state.run_continuously = False
            st.rerun()

        st.session_state.run_continuously = run_continuously

        if run_episode_btn and st.session_state.model is not None:
            play_score, play_steps = _run_episode(
                st.session_state.model, render_fps, p_frame_ph, p_score_ph
            )
            _record_result(play_score, play_steps, p_summary_ph)
            st.rerun()

        if st.session_state.run_continuously and st.session_state.model is not None:
            play_score, play_steps = _run_episode(
                st.session_state.model, render_fps, p_frame_ph, p_score_ph
            )
            _record_result(play_score, play_steps, p_summary_ph)
            st.rerun()
