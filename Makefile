.PHONY: prepare clean

prepare: .venv

.venv:
	python -m venv .venv
	pip install matplotlib tensorflow keras

clean:
	rm *.png
