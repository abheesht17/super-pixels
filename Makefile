
.PHONY: env style requirements python-reqs

VENV = superpixels-env
export VIRTUAL_ENV := $(abspath ${VENV})
export PATH := ${VIRTUAL_ENV}/bin:${PATH}

${VENV}:
	python3 -m venv $@

env: requirements.txt | ${VENV}
	pip install --upgrade -r requirements.txt

quality:
	black --check --line-length 119 --target-version py36 src ./*.py
	isort --check-only src ./*.py
	flake8  src ./*.py

style:
	black --line-length 119 --target-version py36 src *.py
	isort src *.py

final: style | ${VENV}
	pip freeze>requirements.txt
