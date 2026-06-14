.PHONY: install demo eval test all

install:        ## install the package + dev deps (pytest)
	pip install -e ".[dev]"

demo:           ## run the recommendation demo
	python scripts/recommend_demo.py

eval:           ## run the reproducible evaluation (metrics + baselines)
	python scripts/evaluate.py

test:           ## run the unit tests
	pytest

all: test eval demo
