.PHONY: prepare

prepare: .venv

.venv:
	python -m venv .venv
	pip install tensorflow keras
