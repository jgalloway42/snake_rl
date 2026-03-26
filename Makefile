.PHONY: install format lint test check train run gif clean

install:
	pip install -e ".[dev]"

format:
	black src/ tests/

lint:
	pylint --disable=R,C src/snake_rl/

test:
	SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy pytest tests/ -v

check: format lint test

train:
	SDL_AUDIODRIVER=dummy python -m snake_rl.main --train

run:
	python -m snake_rl.main --run

gif:
	SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python scripts/record_gif.py --out scripts/figures/agent.gif

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/ .coverage models/
