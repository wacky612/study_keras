export PATH := .venv/bin:$(PATH)

.PHONY: prepare python

prepare: .venv

.venv:
	python -m venv .venv
	pip install tensorflow keras

python:
	python
