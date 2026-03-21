.PHONY: install format lint test check train stream clean

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
	SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
	python -m snake_rl.main --train

stream:
	python -m snake_rl.main --stream

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/ .coverage mlruns/ models/
