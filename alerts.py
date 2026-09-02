"""
Чистая логика фильтрации детекций и TTC-алертов.

Вынесена из main.py в отдельный модуль без зависимости от cv2/ultralytics,
чтобы её можно было юнит-тестировать без установки тяжёлых CV/ML-пакетов.
"""
from __future__ import annotations

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

# Классы, для которых считаем TTC и триггерим collision warning. Пешеход,
# идущий на камеру по центральной полосе, физически так же приближается
# (его бокс так же растёт по ширине), как и машина — исключать person из
# алерта означало бы не предупреждать о самом опасном сценарии на дороге.
TTC_ALERT_CLASSES = VEHICLE_CLASSES | {"person"}

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


def is_ttc_eligible(cls_name, cx, frame_w, lane_band):
    """Объект по курсу: класс, для которого имеет смысл TTC, и центр в полосе."""
    return cls_name in TTC_ALERT_CLASSES and in_lane_band(cx, frame_w, lane_band)


def ttc_color(ttc, warn_s, caution_s):
    if ttc <= warn_s:
        return TTC_COLOR_WARN
    if ttc <= caution_s:
        return TTC_COLOR_CAUTION
    return TTC_COLOR_OK
