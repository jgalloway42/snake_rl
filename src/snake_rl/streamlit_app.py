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
from stable_baselines3 import PPO

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
# Play tab: persistent env across reruns so each rerun does one step
if "play_env" not in st.session_state:
    st.session_state.play_env = None
if "play_obs" not in st.session_state:
    st.session_state.play_obs = None
if "play_info" not in st.session_state:
    st.session_state.play_info = {}
if "play_rewards" not in st.session_state:
    st.session_state.play_rewards = []  # cumulative reward trace for current episode

# Reconnect to any training thread that survived a page refresh.
if st.session_state.training_thread is None and _mgr.thread is not None and _mgr.thread.is_alive():
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


def _start_play_episode() -> None:
    """Open a new env and store it in session state for step-by-step playback."""
    play_env = SnakeEnv(render_mode="rgb_array")
    play_obs, _ = play_env.reset()
    st.session_state.play_env = play_env
    st.session_state.play_obs = play_obs
    st.session_state.play_info = {}
    st.session_state.play_rewards = []


def _stop_play_episode() -> None:
    """Close the active env and clear play state."""
    if st.session_state.play_env is not None:
        st.session_state.play_env.close()
    st.session_state.play_env = None
    st.session_state.play_obs = None
    st.session_state.play_info = {}
    st.session_state.play_rewards = []


def _on_play_exit() -> None:
    """on_click callback for the Exit button — runs before widgets re-render.

    Streamlit forbids modifying a keyed widget's session state key after the
    widget is instantiated in the same script run.  Using on_click sidesteps
    this: the callback executes before the next render, so the toggle's bound
    key can be safely reset here.
    """
    _stop_play_episode()
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
    if train_exit_btn and _mgr.ts is not None:
        _mgr.ts.stop_requested = True
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
            ("Value Loss", "value_loss"),
            ("Policy Loss", "policy_loss"),
            ("Episode Length", "ep_len_mean"),
            ("Entropy", "entropy"),
            ("KL Divergence", "approx_kl"),
        ]
        metrics = state.get_metrics_snapshot() if state is not None else []
        row1 = st.columns(3)
        row2 = st.columns(3)
        for chart_col, (title, mkey) in zip(row1 + row2, chart_configs):
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
        value="models/snake_ppo.zip",
        key="t_continue",
    )

    # Single toggle button: Start Training ↔ Stop Training
    if _is_training():
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
            _p = _cfg.get("ppo", {})
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
            st.markdown("**PPO**")
            st.table(
                pd.DataFrame(
                    [
                        (
                            "learning_rate",
                            _p.get("learning_rate"),
                            "Adam learning rate",
                        ),
                        (
                            "gamma",
                            _p.get("gamma"),
                            "Discount factor for future rewards",
                        ),
                        (
                            "ent_coef",
                            _p.get("ent_coef"),
                            "Entropy bonus — higher = more exploration",
                        ),
                        (
                            "clip_range",
                            _p.get("clip_range"),
                            "PPO clip ε — limits policy update size",
                        ),
                        (
                            "n_steps",
                            _p.get("n_steps"),
                            "Steps collected per rollout per env",
                        ),
                        ("n_epochs", _p.get("n_epochs"), "Gradient steps per rollout"),
                        ("batch_size", _p.get("batch_size"), "Mini-batch size"),
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

    # Auto-refresh while training
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
        episode_active = st.session_state.play_env is not None
        run_episode_btn = p_col1.button(
            "Run Episode",
            disabled=st.session_state.model is None or episode_active,
        )
        # key= binds toggle to session_state.run_continuously.
        # Exit uses on_click=_on_play_exit so the callback resets the key
        # before the next render (direct assignment after widget creation
        # raises StreamlitAPIException).
        p_col2.toggle("Run Continuously", key="run_continuously")
        p_col3.button("Exit / Stop", key="play_exit", on_click=_on_play_exit)

        # --- Start a new episode ---
        if run_episode_btn and st.session_state.model is not None:
            _start_play_episode()
            st.rerun()

        if (
            st.session_state.run_continuously
            and st.session_state.model is not None
            and not episode_active
        ):
            _start_play_episode()
            st.rerun()

        # Reward chart placeholder (shown below the board)
        reward_chart_ph = st.empty()

        # --- Advance one step in the active episode ---
        if episode_active and st.session_state.model is not None:
            env = st.session_state.play_env
            obs = st.session_state.play_obs

            frame = env.render()
            if frame is not None:
                p_frame_ph.image(frame, channels="RGB", width=480)

            action, _ = st.session_state.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            st.session_state.play_obs = obs
            st.session_state.play_info = info
            st.session_state.play_rewards.append(float(reward))

            p_score_ph.markdown(
                f"**Score:** {info['score']}  |  **Steps:** {info['steps']}"
            )

            if terminated or truncated:
                _record_result(info["score"], info["steps"], p_summary_ph)
                _stop_play_episode()

        # Draw cumulative reward chart for current (or just-finished) episode
        if st.session_state.play_rewards:
            rewards = st.session_state.play_rewards
            cumulative = pd.Series(rewards).cumsum()
            reward_chart_ph.line_chart(
                pd.DataFrame({"cumulative reward": cumulative}),
                height=180,
            )

        if episode_active:
            time.sleep(1 / render_fps)
            st.rerun()
