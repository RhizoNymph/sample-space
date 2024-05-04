run:
	. venv/bin/activate
	python3 dash_app.py

install:
	[ -d venv ] || python3 -m venv venv
	. venv/bin/activate
	pip install -r requirements.txt
	git pull --recurse-submodules
	cd Multicore-TSNE && pip install .

all: install run