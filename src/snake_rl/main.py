"""
main.py — CLI entry point.

Usage:
    snake-rl --train [--config path/to/config.yaml]
    snake-rl --stream
"""

import argparse
import subprocess
import sys
from pathlib import Path


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="snake-rl",
        description="Train or demo a PPO agent for Snake.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Run PPO training")
    group.add_argument(
        "--stream", action="store_true", help="Launch Streamlit demo app"
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config (used with --train)",
    )
    args = parser.parse_args()

    if args.train:
        from snake_rl.train import train

        train(config_path=args.config)

    elif args.stream:
        app_path = Path(__file__).parent / "streamlit_app.py"
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=False,
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    cli()
