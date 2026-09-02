"""
Многообъектный трекер с прогнозированием движения.

Модель: constant velocity Kalman filter на каждый трек.
Состояние: [cx, cy, w, h, vx, vy, vw, vh]
Ассоциация детекций с треками: Hungarian algorithm по IoU.
"""
from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU для боксов в формате [x1, y1, x2, y2]."""
    xa1, ya1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xa2, ya2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter_w, inter_h = max(0.0, xa2 - xa1), max(0.0, ya2 - ya1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


def cxcywh_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = state[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class Track:
    """Один сопровождаемый объект со своим Kalman-фильтром."""

    _next_id = 1

    def __init__(self, box_xyxy: np.ndarray, cls_id: int, fps: float):
        self.id = Track._next_id
        Track._next_id += 1
        self.cls_id = cls_id
        self.fps = fps
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history: list[np.ndarray] = []  # центры для рисования траектории

        cxcywh = xyxy_to_cxcywh(box_xyxy)

        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0
        self.kf.F = np.eye(8)
        for i in range(4):
            self.kf.F[i, i + 4] = dt
        self.kf.H = np.zeros((4, 8))
        self.kf.H[:4, :4] = np.eye(4)

        self.kf.R *= 5.0
        self.kf.P[4:, 4:] *= 100.0
        self.kf.P *= 10.0
        self.kf.Q[4:, 4:] *= 0.5
        self.kf.Q[:4, :4] *= 1.0

        self.kf.x[:4] = cxcywh.reshape(4, 1)
        self.history.append(cxcywh[:2].copy())

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return cxcywh_to_xyxy(self.kf.x[:4].flatten())

    def update(self, box_xyxy: np.ndarray, max_growth: float = 1.5):
        """
        Обновляет фильтр новой детекцией. Ширина/высота измерения клэмпятся
        относительно текущего предсказанного размера трека (не более чем в
        max_growth раз за один кадр) — это защита от "разъезжания" бокса:
        Hungarian-ассоциация по IoU иногда матчит трек с соседней машиной или
        неточной детекцией на один кадр, и без клэмпа Kalman-фильтр наследует
        этот скачок и дальше экстраполирует его дальше, раздувая бокс на весь
        кадр за несколько following-кадров.
        """
        measured = xyxy_to_cxcywh(box_xyxy)
        prev_w, prev_h = float(self.kf.x[2, 0]), float(self.kf.x[3, 0])
        if prev_w > 1 and prev_h > 1:
            measured[2] = np.clip(measured[2], prev_w / max_growth, prev_w * max_growth)
            measured[3] = np.clip(measured[3], prev_h / max_growth, prev_h * max_growth)

        self.kf.update(measured.reshape(4, 1))
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.kf.x[:2].flatten().copy())
        if len(self.history) > 60:
            self.history.pop(0)

    def current_box(self) -> np.ndarray:
        return cxcywh_to_xyxy(self.kf.x[:4].flatten())

    def velocity_px_s(self) -> tuple[float, float]:
        vx, vy = self.kf.x[4, 0], self.kf.x[5, 0]
        return vx * self.fps, vy * self.fps

    def speed_px_s(self) -> float:
        vx, vy = self.velocity_px_s()
        return float(np.hypot(vx, vy))

    def width_growth_px_s(self) -> float:
        """Скорость роста ширины бокса (px/s). Положительная = объект приближается."""
        return float(self.kf.x[6, 0]) * self.fps

    def ttc_seconds(self) -> float | None:
        """
        Time-to-collision по разрастанию бокса (looming), без калибровки камеры:
        TTC = w / (dw/dt). Работает, потому что при постоянной скорости сближения
        видимая ширина объекта растёт обратно пропорционально расстоянию до него.
        Возвращает None, если объект не приближается (dw/dt <= 0).
        """
        w = float(self.kf.x[2, 0])
        vw = self.width_growth_px_s()
        if vw <= 1e-3 or w <= 0:
            return None
        return w / vw

    def forecast_path(self, steps: int, dt: float = 1.0) -> list[np.ndarray]:
        """Прогноз будущих центров без обновления реального состояния фильтра."""
        x = self.kf.x.copy()
        f = self.kf.F.copy()
        f_step = np.eye(8)
        for i in range(4):
            f_step[i, i + 4] = dt
        path = []
        for _ in range(steps):
            x = f_step @ x
            path.append(x[:2].flatten().copy())
        return path


class MultiObjectTracker:
    def __init__(self, fps: float, iou_threshold: float = 0.3, max_age: int = 15, min_hits: int = 3):
        self.fps = fps
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []

    def update(self, detections: list[tuple[np.ndarray, int]]) -> list[Track]:
        """detections: список (box_xyxy, cls_id). Возвращает подтверждённые треки."""
        predicted_boxes = [t.predict() for t in self.tracks]

        if self.tracks and detections:
            cost = np.zeros((len(self.tracks), len(detections)))
            for i, pbox in enumerate(predicted_boxes):
                for j, (dbox, _) in enumerate(detections):
                    cost[i, j] = 1.0 - iou(pbox, dbox)
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx, col_idx = np.array([], dtype=int), np.array([], dtype=int)

        matched_tracks, matched_dets = set(), set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] <= (1.0 - self.iou_threshold):
                self.tracks[r].update(detections[c][0])
                matched_tracks.add(r)
                matched_dets.add(c)

        for j, (dbox, cls_id) in enumerate(detections):
            if j not in matched_dets:
                self.tracks.append(Track(dbox, cls_id, self.fps))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t for t in self.tracks if t.hits >= self.min_hits and t.time_since_update == 0]