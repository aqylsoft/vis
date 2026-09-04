IMAGE    ?= driver-assist-cv
DATA_DIR ?= $(CURDIR)
SOURCE   ?= video.mp4
OUTPUT   ?= output/result.mp4
CSV      ?= output/tracks.csv
ARGS     ?=
PYTHON   ?= python3

.PHONY: install install-dev test run clean \
        docker-build docker-test docker-run \
        face-install face-register face-watch face-list face-autolock face-greet face-web

## Local (venv) workflow

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) main.py --source $(SOURCE) --output $(OUTPUT) --csv $(CSV) $(ARGS)

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache output/*

## Docker workflow (headless — see README for webcam/preview caveats)

docker-build:
	docker build --target runtime -t $(IMAGE) .

docker-test:
	docker build --target test -t $(IMAGE):test .
	docker run --rm $(IMAGE):test

# Mounts DATA_DIR (default: repo root) at /data inside the container, so
# SOURCE/OUTPUT/CSV are paths relative to DATA_DIR on the host.
# Example: make docker-run DATA_DIR=~/videos SOURCE=dashcam.mp4
docker-run: docker-build
	mkdir -p $(DATA_DIR)/$(dir $(OUTPUT))
	docker run --rm \
		-v $(DATA_DIR):/data \
		$(IMAGE) --source /data/$(SOURCE) --output /data/$(OUTPUT) --csv /data/$(CSV) $(ARGS)

## Face recognition module

FACE_NAME ?= User
CAMERA    ?= 0
PHONE_URL ?= http://192.168.1.100:8080/video

face-install:
	@command -v cmake >/dev/null || (echo "Installing cmake..." && sudo apt-get install -y cmake)
	@dpkg -s libopenblas-dev >/dev/null 2>&1 || (echo "Installing dlib dependencies..." && sudo apt-get install -y build-essential libopenblas-dev liblapack-dev)
	pip install face_recognition

face-register: face-install
	$(PYTHON) -m face_module register --name "$(FACE_NAME)" --camera $(CAMERA)

face-list:
	$(PYTHON) -m face_module list

face-watch: face-install
	$(PYTHON) -m face_module watch --camera $(CAMERA)

face-phone: face-install
	$(PYTHON) -m face_module watch --camera "$(PHONE_URL)"

LOCK_TIMEOUT ?= 10
GREET_COOLDOWN ?= 30

face-autolock: face-install
	$(PYTHON) -m face_module autolock --camera $(CAMERA) --timeout $(LOCK_TIMEOUT) --preview

face-greet: face-install
	pip install anthropic python-dotenv
	$(PYTHON) -m face_module greet --camera $(CAMERA) --cooldown $(GREET_COOLDOWN)

WEB_PORT ?= 5000
WEB_MODE ?= watch

face-web: face-install
	pip install flask
	$(PYTHON) -m face_module web --camera $(CAMERA) --port $(WEB_PORT) --mode $(WEB_MODE)
