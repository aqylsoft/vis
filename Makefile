IMAGE    ?= driver-assist-cv
DATA_DIR ?= $(CURDIR)
SOURCE   ?= video.mp4
OUTPUT   ?= output/result.mp4
CSV      ?= output/tracks.csv
ARGS     ?=

.PHONY: install install-dev test run clean \
        docker-build docker-test docker-run

## Local (venv) workflow

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	python -m pytest

run:
	python main.py --source $(SOURCE) --output $(OUTPUT) --csv $(CSV) $(ARGS)

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
