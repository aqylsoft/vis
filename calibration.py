"""
Автокалибровка под конкретный dashcam-стрим.

Идея: у каждого дэшкама своя геометрия салона (где руки, руль, приборка,
насколько широкий FOV) — нет универсального "правильного" числа для этого.
Но есть универсальный признак: пока машина едет, дорога/сцена в кадре
двигается, а салон (руль, приборка, капот) — почти неподвижен относительно
камеры. Это позволяет определить границу "интерьер / дорога" автоматически
по нескольким секундам видео, без ручной разметки под конкретную камеру.

Аналогично калибруется разрешение инференса: если на мелких/дальних объектах
детектор уверен слабо на стандартном 640, пробуем крупнее и смотрим, помогает
ли это — если да, остаёмся на большем разрешении для всего видео.
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_road_roi_bottom(cap: cv2.VideoCapture, sample_seconds: float = 2.0,
                              static_percentile: float = 30.0) -> float:
    """
    Определяет долю высоты кадра (0..1), ниже которой начинается статичная зона
    (капот/приборка/руки на руле), по относительному движению в каждой
    горизонтальной полосе кадра за первые sample_seconds видео.

    Из среднего движения по строке вычитается общий фон-шум кадра (вибрация
    камеры на скорости), иначе он забивает сигнал и делает все строки
    одинаково "подвижными" — иначе на высокой скорости трясётся весь кадр
    целиком, включая приборку, и абсолютный порог перестаёт что-либо различать.

    Возвращает 1.0 (весь кадр — "дорога"), если явной статичной зоны снизу не
    нашлось (например на очень высокой скорости, где вибрация делает движение
    рук и вибрацию неотличимыми от дороги) — тогда ROI-фильтр просто ничего
    не отрежет, и это безопасный fallback, а не ложное решение.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_samples = max(2, int(fps * sample_seconds))

    prev_gray = None
    motion_per_row = None
    height = width = None
    frames_used = 0
    global_motion_samples = []

    for _ in range(n_samples):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        if height is None:
            height, width = gray.shape
            motion_per_row = np.zeros(height, dtype=np.float64)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray).astype(np.float64)
            row_motion = diff.mean(axis=1)
            motion_per_row += row_motion
            global_motion_samples.append(row_motion.mean())
            frames_used += 1
        prev_gray = gray

    # отмотать источник обратно на начало для основного прохода
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if frames_used < 1 or motion_per_row is None:
        return 1.0

    motion_per_row /= frames_used
    # убираем общий фон-шум (вибрацию всего кадра целиком) — интересует только
    # то, что движется БОЛЬШЕ или МЕНЬШЕ среднего по кадру, а не абсолютная
    # величина движения
    baseline = float(np.mean(global_motion_samples))
    relative = motion_per_row - baseline

    spread = relative.max() - relative.min()
    if spread <= 1e-6:
        return 1.0

    threshold = np.percentile(relative, static_percentile)

    # Идём снизу вверх, ищем самую нижнюю непрерывную статичную зону.
    static_rows_from_bottom = 0
    for row_value in relative[::-1]:
        if row_value <= threshold:
            static_rows_from_bottom += 1
        else:
            break

    # Если статичная зона слишком маленькая (<8% высоты) — считаем, что чёткой
    # границы интерьер/дорога эвристика не нашла (например из-за сильной
    # вибрации на высокой скорости), и безопаснее ничего не отрезать.
    if static_rows_from_bottom < 0.08 * height:
        return 1.0

    cutoff_ratio = 1.0 - (static_rows_from_bottom / height)
    # небольшой запас вверх, чтобы не срезать нижний край реальных объектов на дороге
    return float(np.clip(cutoff_ratio + 0.03, 0.3, 1.0))


def estimate_inference_size(model, cap: cv2.VideoCapture, sample_frames: int = 4,
                             small_size: int = 640, large_size: int = 1280,
                             conf_check: float = 0.4, improvement_ratio: float = 1.5) -> int:
    """
    Прогоняет несколько кадров на small_size и large_size, сравнивает число
    уверенных детекций. Если крупное разрешение находит заметно больше объектов
    (типично для fisheye/4K видео, где объекты мелкие) — используем его для
    всего видео, иначе остаёмся на small_size ради скорости.
    """
    start_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

    frames = []
    for _ in range(sample_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

    if not frames:
        return small_size

    def count_confident(imgsz: int) -> int:
        total = 0
        for f in frames:
            res = model.predict(f, conf=conf_check, imgsz=imgsz, verbose=False)[0]
            total += len(res.boxes)
        return total

    small_count = count_confident(small_size)
    large_count = count_confident(large_size)

    if small_count == 0 or (large_count / max(small_count, 1)) >= improvement_ratio:
        return large_size
    return small_size


def calibrate(model, cap: cv2.VideoCapture) -> dict:
    """Полная автокалибровка под текущий видеопоток. Возвращает подобранные параметры."""
    road_roi_bottom = estimate_road_roi_bottom(cap)
    imgsz = estimate_inference_size(model, cap)
    return {"road_roi_bottom": road_roi_bottom, "imgsz": imgsz}