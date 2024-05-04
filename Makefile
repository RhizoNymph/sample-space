run:
	. venv/bin/activate
	python3 dash_app.py

install:
	python3 -m venv venv
	. venv/bin/activate
	pip install -r requirements.txt

all: install run