"""
Прогноз траектории самой машины (ego-path) по монокулярному видео.

У нас нет данных о руле/скорости (CAN-шина недоступна) — только кадры. Поэтому
поворот оценивается по оптическому потоку статичной сцены: когда машина
поворачивает, весь видимый мир на кадре сдвигается в противоположную сторону
(при повороте направо дальние объекты уезжают влево). Это классический сигнал
yaw rate из чистого видео, без датчиков.

Это ОЦЕНКА, не точная физика: она реагирует на любое рысканье камеры (включая
неровности дороги, покачивание крепления), поэтому сглаживается экспоненциальным
скользящим средним, и результат стоит показывать как приблизительный прогноз
курса, а не точную траекторию.
"""
from __future__ import annotations

import cv2
import numpy as np


class EgoMotionEstimator:
    """Оценивает боковое рысканье (поворот) по оптическому потоку между кадрами
    в полосе у горизонта и сглаживает его в устойчивую оценку курса."""

    def __init__(self, smoothing: float = 0.85, flow_scale: tuple[int, int] = (240, 135)):
        self.smoothing = smoothing
        self.flow_scale = flow_scale
        self.prev_gray: np.ndarray | None = None
        self.turn_rate: float = 0.0  # сглаженный сигнал поворота, px/frame в уменьшенном масштабе

    def update(self, frame: np.ndarray, road_roi_bottom: float) -> float:
        """Кормим очередной кадр, возвращаем текущую сглаженную оценку поворота
        (положительная = поворот вправо, отрицательная = влево)."""
        small = cv2.resize(frame, self.flow_scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return self.turn_rate

        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 2, 15, 3, 5, 1.2, 0)
        h, w = gray.shape
        # полоса у горизонта: верхняя часть дорожной зоны, там боковой сдвиг сцены
        # при повороте виден отчётливее всего (дальние объекты); используем только
        # верхние 15-35% дорожного ROI, чтобы не ловить объекты вблизи камеры,
        # чьё собственное движение (TTC) не связано с поворотом машины
        band_top = int(h * 0.15)
        band_bottom = int(h * min(0.35, road_roi_bottom))
        if band_bottom <= band_top:
            band_bottom = band_top + 1
        band = flow[band_top:band_bottom, :, 0]

        raw_turn = -float(np.median(band))  # знак: сдвиг сцены влево = поворот вправо
        self.turn_rate = self.smoothing * self.turn_rate + (1 - self.smoothing) * raw_turn
        self.prev_gray = gray
        return self.turn_rate

    def forecast_ego_corridor(self, frame_w: int, frame_h: int, road_roi_bottom: float,
                               horizon_ratio: float = 0.35, points: int = 20,
                               near_half_width_ratio: float = 0.14,
                               far_half_width_ratio: float = 0.01) -> tuple[list, list, list]:
        """
        Строит коридор предсказанного курса как перспективную трапецию, лежащую
        на дорожном полотне: широкую у капота (near_half_width_ratio доли ширины
        кадра) и сужающуюся к горизонту (far_half_width_ratio) — так же, как
        реальная полосная разметка визуально сходится к точке схода. Раньше
        путь рисовался лентой постоянной ширины и из-за отсутствия перспективного
        сужения выглядел как повисший в воздухе флажок, а не как проекция на
        асфальт.

        Возвращает (центральная линия, левая кромка, правая кромка) — списки
        точек (x, y) от машины к горизонту.
        """
        start_y = frame_h * road_roi_bottom
        end_y = frame_h * horizon_ratio
        start_x = frame_w / 2.0

        curve_gain = frame_w * 0.06
        max_offset = curve_gain * np.clip(self.turn_rate / 2.0, -3.0, 3.0)

        near_half_width = frame_w * near_half_width_ratio
        far_half_width = frame_w * far_half_width_ratio

        center, left, right = [], [], []
        for i in range(points):
            t = i / (points - 1)
            y = start_y + (end_y - start_y) * t
            cx = start_x + max_offset * (t ** 1.5)
            # линейное сужение половины ширины коридора от near к far — перспективный клин
            half_w = near_half_width * (1 - t) + far_half_width * t
            center.append((int(cx), int(y)))
            left.append((int(cx - half_w), int(y)))
            right.append((int(cx + half_w), int(y)))
        return center, left, right

    def forecast_ego_path(self, frame_w: int, frame_h: int, road_roi_bottom: float,
                           horizon_ratio: float = 0.35, points: int = 20) -> list[tuple[int, int]]:
        """Только центральная линия курса, без ширины коридора (обратная совместимость)."""
        center, _, _ = self.forecast_ego_corridor(frame_w, frame_h, road_roi_bottom, horizon_ratio, points)
        return center