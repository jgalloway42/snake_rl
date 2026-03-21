"""
streamlit_app.py — Interactive demo for watching a trained PPO agent play Snake.
Run via: streamlit run src/snake_rl/streamlit_app.py
"""

from __future__ import annotations

import time

import streamlit as st
from stable_baselines3 import PPO

from snake_rl.env import SnakeEnv

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="snake-rl demo", layout="wide")


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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("snake-rl")
    st.markdown("Watch a trained PPO agent play Snake.")

    model_path = st.text_input("Model path (.zip)", value="models/snake_ppo.zip")

    if st.button("Load Model"):
        try:
            st.session_state.model = PPO.load(model_path)
            st.success(f"Loaded: {model_path}")
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


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Snake RL — Live Demo")

frame_placeholder = st.empty()
score_placeholder = st.empty()
summary_placeholder = st.empty()

col1, col2 = st.columns(2)
run_episode_btn = col1.button("Run Episode", disabled=st.session_state.model is None)
run_continuously = col2.toggle(
    "Run Continuously", value=st.session_state.run_continuously
)
st.session_state.run_continuously = run_continuously


def run_episode(model: PPO) -> tuple[int, int]:
    """Run one full episode and return (score, steps)."""
    env = SnakeEnv(render_mode="rgb_array")
    obs, _ = env.reset()
    info: dict = {}
    done = False
    while not done:
        frame = env.render()
        if frame is not None:
            frame_placeholder.image(frame, channels="RGB", width=480)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        score_placeholder.markdown(
            f"**Score:** {info['score']}  |  **Steps:** {info['steps']}"
        )
        done = terminated or truncated
        time.sleep(1 / render_fps)
    env.close()
    return info["score"], info["steps"]


def record_result(result_score: int, result_steps: int) -> None:
    st.session_state.episode_count += 1
    st.session_state.leaderboard.append((result_score, result_steps))
    st.session_state.leaderboard.sort(key=lambda x: x[0], reverse=True)
    summary_placeholder.success(
        f"Episode {st.session_state.episode_count} — "
        f"Score: **{result_score}** | Steps: **{result_steps}**"
    )


if run_episode_btn and st.session_state.model is not None:
    ep_score, ep_steps = run_episode(st.session_state.model)
    record_result(ep_score, ep_steps)
    st.rerun()

if st.session_state.run_continuously and st.session_state.model is not None:
    ep_score, ep_steps = run_episode(st.session_state.model)
    record_result(ep_score, ep_steps)
    st.rerun()
