# Driver Assist CV — prototype

Pipeline: object detection (YOLOv8) → tracking with a constant-velocity Kalman
filter and Hungarian association by IoU → TTC (time-to-collision) warning for
objects ahead → ego-path forecast from optical flow → auto-calibration of the
camera's road ROI and inference resolution. Classes of interest: person, car,
motorcycle, bus, truck.

## Install

```bash
pip install -r requirements.txt
# or: make install
```

`yolov8n.pt` weights download automatically on first run (nano model, the
fastest — good for CPU/prototyping; use `yolov8s.pt`/`yolov8m.pt` for better
accuracy).

## Run

```bash
# on a video file
python main.py --source path/to/video.mp4 --output output/result.mp4
# or: make run SOURCE=path/to/video.mp4

# from a webcam, no preview window (e.g. on a server)
python main.py --source 0 --no-display
```

Output:
- `output/result.mp4` — annotated video: boxes, track IDs, TTC labels/colors,
  the collision-warning banner, the real trajectory tail, the short-term
  motion forecast (yellow dashed line), and the predicted ego-path corridor
  (a perspective trapezoid projected onto the road ahead).
- `output/tracks.csv` — per-frame track coordinates, box width and TTC for
  further analysis (pandas/Grafana/whatever).

## How it works

- **tracker.py** — `MultiObjectTracker`: a separate constant-velocity Kalman
  filter per object, state `[cx, cy, w, h, vx, vy, vw, vh]`. Detections are
  associated to tracks via the Hungarian algorithm on IoU. `forecast_path()`
  rolls the motion model forward without updating the real filter state —
  that's the "motion forecast".
- **alerts.py** — pure detection-filtering and TTC-alert logic (no cv2/YOLO
  dependency): `filter_bogus_detections`, `is_ttc_eligible`, `ttc_color`. TTC
  is computed for cars/motorcycles/buses/trucks **and pedestrians** in the
  central lane band — a pedestrian walking toward the camera closes distance
  the same way a vehicle does (their box grows in width too), so excluding
  them from the alert would leave the most dangerous case unguarded.
- **ego_motion.py** — `EgoMotionEstimator`: estimates the car's own yaw
  (turning) from optical flow of the static background near the horizon (no
  CAN bus / steering data available), smooths it, and projects a perspective
  road corridor as the predicted ego-path.
- **calibration.py** — auto-calibrates, from a few seconds of video, where the
  dashboard/hood/hands-on-wheel region ends and the road begins
  (`road_roi_bottom`), and whether a larger YOLO inference resolution
  meaningfully improves detection (`imgsz`) — both vary by camera mount and
  are picked automatically instead of needing per-camera manual tuning.
- **main.py** — reads frames, runs YOLO, feeds detections into the tracker,
  draws the overlay (boxes, TTC, ego-path corridor, collision-warning
  banner), and writes the annotated video + CSV.

## Tests

Unit tests cover the pure logic (`tracker.py`, `alerts.py`) without depending
on cv2/ultralytics, so they run without installing the heavy CV/ML stack:

```bash
pip install -r requirements-dev.txt
python -m pytest
# or: make install-dev test
```

## Docker

For headless/batch use (processing a video file on a server, or in CI) a
`Dockerfile` is included so behavior doesn't depend on what's installed on
the host. It's a multi-stage build that keeps the build environment out of
the shipped image: a `builder` stage installs `requirements.txt` into a venv
(`test-builder` does the same plus `requirements-dev.txt`), while the final
`runtime`/`test` stages start fresh from a slim base, add only the OS libs
opencv/torch need even without a display, and copy in just the venv they
need — no pip, no apt package lists, no build-time files ever reach the
final image. `runtime` (the default target) runs `main.py --no-display`;
`test` runs pytest.

```bash
make docker-build                       # build the runtime image
make docker-test                        # run the test suite inside a container

# process a video: DATA_DIR is mounted at /data, SOURCE/OUTPUT/CSV are
# paths relative to it (defaults: DATA_DIR=. SOURCE=video.mp4
# OUTPUT=output/result.mp4 CSV=output/tracks.csv)
make docker-run DATA_DIR=~/videos SOURCE=dashcam.mp4
```

Equivalent plain `docker` commands, if you'd rather not use `make`:

```bash
docker build --target runtime -t driver-assist-cv .
docker build --target test -t driver-assist-cv:test . && docker run --rm driver-assist-cv:test
docker run --rm -v "$PWD":/data driver-assist-cv \
    --source /data/video.mp4 --output /data/output/result.mp4 --csv /data/output/tracks.csv
```

The container is headless by design (`--no-display` is baked into the
entrypoint) and has no camera device attached — for interactive use with a
live webcam preview (`--source 0`, `cv2.imshow`), run locally in a venv
instead of in Docker.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs the test suite on every pull
request targeting `master` and on every push to `master` (i.e. after a
merge). The push-to-`master` run reuses the PR's own test result instead of
re-running pytest: it caches a "verified" marker keyed by a hash of all
`*.py`/requirements files, so when the exact code just tested in the PR lands
on `master` unchanged (the normal merge case), the cache hits and the job
skips straight to success instead of re-testing.

## Where to take this next

- Replace the constant-velocity model with constant-acceleration or CTRV (for
  turns).
- Multiple lane bands / zone polygons instead of a single central band —
  track transitions between zones (e.g. a vehicle merging into the ego lane).
- Move inference into a separate process/service in Go (gRPC to a Python
  inference service, or ONNX Runtime straight from Go) — if this goes to
  production rather than staying a prototype.
- Export metrics to Prometheus instead of/alongside CSV — "where to stream
  this" can later point at Postgres or a queue just as easily.
- Pin dependency versions in `requirements.txt` (currently `>=`) for
  reproducible installs — a breaking `ultralytics`/`torch` release could
  otherwise break the build with no change to this repo at all.
