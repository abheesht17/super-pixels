
.PHONY: env style requirements python-reqs

VENV = superpixels-env
export VIRTUAL_ENV := $(abspath ${VENV})
export PATH := ${VIRTUAL_ENV}/bin:${PATH}

${VENV}:
	python3 -m venv $@

env: requirements.txt | ${VENV}
	pip install --upgrade -r requirements.txt

# black --check --line-length 119 --target-version py38 src ./*.py
# isort --check-only src ./*.py #Remove these, black and isort contradict each other

quality:
	flake8 --ignore F401,F403,W503 --max-line-length 119 src ./*.py

style:
	black --line-length 119 --target-version py38 src *.py
	isort src *.py

final: style | ${VENV}
	pip freeze>requirements.txt
