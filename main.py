"""
Dashcam-ассистент водителя: детекция + трекинг + предупреждение о сближении (TTC).

Использование:
    python main.py --source path/to/video.mp4 --output output/result.mp4
    python main.py --source 0 --no-display

Логика "прямо по курсу": предупреждение о сближении (TTC) считается только для
объектов, чей центр попадает в центральную горизонтальную полосу кадра — это
приближение "своей полосы" для лобовой dashcam-камеры. Объекты сбоку (встречка,
соседняя полоса) трекаются и рисуются, но не триггерят алерт.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from tracker import MultiObjectTracker
from calibration import calibrate
from ego_motion import EgoMotionEstimator

COCO_CLASSES_OF_INTEREST = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

COLORS = {
    "person": (0, 200, 255),
    "car": (0, 255, 0),
    "motorcycle": (255, 150, 0),
    "bus": (255, 0, 200),
    "truck": (0, 100, 255),
}

TTC_COLOR_OK = (0, 255, 0)
TTC_COLOR_CAUTION = (0, 220, 255)
TTC_COLOR_WARN = (0, 0, 255)


def filter_bogus_detections(detections, frame_w, frame_h, max_area_ratio, road_roi_bottom):
    """
    Отбрасывает детекции-мусор:
    - боксы, занимающие неправдоподобно большую долю кадра (типичный false
      positive nano-модели на нетипичном ракурсе — весь салон+лобовое стекло
      принимается за один гигантский "car");
    - боксы, чей центр лежит ниже road_roi_bottom (доля высоты кадра) — это зона
      капота/приборки/рук водителя на dashcam-видео из салона, она никогда не
      содержит реальные объекты на дороге, зато регулярно даёт ложный "person"
      на руках на руле;
    - вырожденные боксы нулевого размера.
    """
    frame_area = frame_w * frame_h
    road_limit_y = frame_h * road_roi_bottom
    clean = []
    for box, cls_id in detections:
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 1.0:
            continue
        if area / frame_area > max_area_ratio:
            continue
        cy = (y1 + y2) / 2
        if cy > road_limit_y:
            continue
        clean.append((box, cls_id))
    return clean


def in_lane_band(cx, frame_w, band):
    lo, hi = band[0] * frame_w, band[1] * frame_w
    return lo <= cx <= hi


def ttc_color(ttc, warn_s, caution_s):
    if ttc <= warn_s:
        return TTC_COLOR_WARN
    if ttc <= caution_s:
        return TTC_COLOR_CAUTION
    return TTC_COLOR_OK


def run(source, output_path, model_name, conf, display, forecast_steps,
        max_area_ratio, road_roi_bottom, lane_band, ttc_warn, ttc_caution, csv_path,
        imgsz, auto_calibrate):

    model = YOLO(model_name)

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть источник видео: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if auto_calibrate:
        picked = calibrate(model, cap)
        if road_roi_bottom is None:
            road_roi_bottom = picked["road_roi_bottom"]
        if imgsz is None:
            imgsz = picked["imgsz"]
        print(f"Автокалибровка: road_roi_bottom={road_roi_bottom:.2f}, imgsz={imgsz}")
    else:
        road_roi_bottom = road_roi_bottom if road_roi_bottom is not None else 0.55
        imgsz = imgsz if imgsz is not None else 640

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = MultiObjectTracker(fps=fps)
    ego_motion = EgoMotionEstimator()

    csv_rows = []
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        ego_motion.update(frame, road_roi_bottom)
        ego_center, ego_left, ego_right = ego_motion.forecast_ego_corridor(width, height, road_roi_bottom)

        results = model.predict(frame, conf=conf, imgsz=imgsz,
                                 classes=list(COCO_CLASSES_OF_INTEREST.keys()), verbose=False)[0]

        raw_detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            raw_detections.append((xyxy, cls_id))

        detections = filter_bogus_detections(raw_detections, width, height, max_area_ratio, road_roi_bottom)
        confirmed = tracker.update(detections)

        active_warning = False
        lo_x, hi_x = int(lane_band[0] * width), int(lane_band[1] * width)
        cv2.line(frame, (lo_x, 0), (lo_x, height), (80, 80, 80), 1)
        cv2.line(frame, (hi_x, 0), (hi_x, height), (80, 80, 80), 1)
        roi_y = int(road_roi_bottom * height)
        cv2.line(frame, (0, roi_y), (width, roi_y), (80, 80, 80), 1)

        # прогноз собственной траектории машины (ego-path) — перспективный
        # коридор, лежащий на дорожном полотне (широкий у капота, сужается к
        # горизонту), а не лента постоянной ширины
        overlay = frame.copy()
        corridor_polygon = np.array(ego_left + ego_right[::-1], dtype=np.int32)
        cv2.fillPoly(overlay, [corridor_polygon], (255, 0, 220))
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, dst=frame)
        cv2.polylines(frame, [np.array(ego_left, dtype=np.int32)], False, (255, 0, 220), 2, cv2.LINE_AA)
        cv2.polylines(frame, [np.array(ego_right, dtype=np.int32)], False, (255, 0, 220), 2, cv2.LINE_AA)

        for t in confirmed:
            cls_name = COCO_CLASSES_OF_INTEREST.get(t.cls_id, "obj")
            base_color = COLORS.get(cls_name, (200, 200, 200))
            x1, y1, x2, y2 = t.current_box().astype(int)
            cx, cy = float(t.kf.x[0, 0]), float(t.kf.x[1, 0])

            ttc = t.ttc_seconds()
            is_ahead = cls_name in VEHICLE_CLASSES and in_lane_band(cx, width, lane_band)

            label = f"#{t.id} {cls_name}"
            color = base_color
            if is_ahead and ttc is not None:
                color = ttc_color(ttc, ttc_warn, ttc_caution)
                label += f" TTC {ttc:.1f}s"
                if ttc <= ttc_warn:
                    active_warning = True
            elif t.width_growth_px_s() <= 0:
                label += " (receding)"

            box_thickness = 3 if (is_ahead and ttc is not None and ttc <= ttc_warn) else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for p1, p2 in zip(t.history, t.history[1:]):
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), base_color, 1)

            if forecast_steps > 0:
                forecast = t.forecast_path(forecast_steps)
                prev = (int(cx), int(cy))
                for fp in forecast:
                    cur = (int(fp[0]), int(fp[1]))
                    cv2.line(frame, prev, cur, (0, 255, 255), 1, cv2.LINE_AA)
                    prev = cur
                cv2.circle(frame, prev, 3, (0, 255, 255), -1)

            csv_rows.append({
                "frame": frame_idx, "track_id": t.id, "class": cls_name,
                "cx": round(cx, 1), "cy": round(cy, 1),
                "width_px": round(float(t.kf.x[2, 0]), 1),
                "ttc_s": round(ttc, 2) if ttc is not None else "",
                "in_lane_band": is_ahead,
            })

        if active_warning:
            cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 255), -1)
            cv2.putText(frame, "COLLISION WARNING", (width // 2 - 150, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        elapsed = time.time() - t0
        cv2.putText(frame, f"frame {frame_idx} | {frame_idx / max(elapsed, 1e-6):.1f} fps proc",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)
        if display:
            cv2.imshow("driver-assist-cv", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer_csv.writeheader()
            writer_csv.writerows(csv_rows)

    print(f"Готово. Кадров: {frame_idx}. Видео: {output_path}. CSV: {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Dashcam-ассистент: детекция, трекинг, TTC-предупреждение о сближении")
    p.add_argument("--source", required=True, help="Путь к видео или индекс камеры (0, 1, ...)")
    p.add_argument("--output", default="output/result.mp4", help="Куда сохранить размеченное видео")
    p.add_argument("--csv", default="output/tracks.csv", help="Куда сохранить траектории в CSV")
    p.add_argument("--model", default="yolov8n.pt", help="Веса YOLO (yolov8n/s/m/l/x.pt)")
    p.add_argument("--conf", type=float, default=0.3, help="Порог уверенности детекции")
    p.add_argument("--imgsz", type=int, default=None,
                    help="Разрешение инференса YOLO. По умолчанию подбирается автокалибровкой под конкретное видео")
    p.add_argument("--forecast-steps", type=int, default=8, help="На сколько кадров вперёд рисовать прогноз (0 = выкл)")
    p.add_argument("--max-area-ratio", type=float, default=0.35,
                    help="Отбрасывать детекции крупнее этой доли площади кадра (фильтр ложных боксов)")
    p.add_argument("--road-roi-bottom", type=float, default=None,
                    help="Игнорировать детекции ниже этой доли высоты кадра. По умолчанию подбирается "
                         "автокалибровкой под конкретную камеру/крепление (зона капота/приборки/рук)")
    p.add_argument("--no-auto-calibrate", action="store_true",
                    help="Отключить автокалибровку road-roi-bottom/imgsz и использовать дефолты (0.55 / 640) "
                         "или явно заданные значения")
    p.add_argument("--lane-band", type=float, nargs=2, default=(0.3, 0.7), metavar=("LO", "HI"),
                    help="Центральная полоса кадра (доли ширины 0..1), в которой считаем объект 'по курсу'")
    p.add_argument("--ttc-warn", type=float, default=2.5, help="TTC (сек) для красного алерта")
    p.add_argument("--ttc-caution", type=float, default=4.5, help="TTC (сек) для жёлтого предупреждения")
    p.add_argument("--no-display", action="store_true", help="Не открывать окно предпросмотра")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source=args.source,
        output_path=args.output,
        model_name=args.model,
        conf=args.conf,
        display=not args.no_display,
        forecast_steps=args.forecast_steps,
        max_area_ratio=args.max_area_ratio,
        road_roi_bottom=args.road_roi_bottom,
        lane_band=tuple(args.lane_band),
        ttc_warn=args.ttc_warn,
        ttc_caution=args.ttc_caution,
        csv_path=args.csv,
        imgsz=args.imgsz,
        auto_calibrate=not args.no_auto_calibrate,
    )