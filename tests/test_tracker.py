import numpy as np
import pytest

from tracker import (
    MultiObjectTracker,
    Track,
    cxcywh_to_xyxy,
    iou,
    xyxy_to_cxcywh,
)


class TestIou:
    def test_identical_boxes(self):
        box = np.array([0, 0, 10, 10])
        assert iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([100, 100, 110, 110])
        assert iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 5, 15, 15])
        # intersection 5x5=25, union 100+100-25=175
        assert iou(a, b) == pytest.approx(25 / 175)

    def test_degenerate_box_has_zero_iou(self):
        a = np.array([0, 0, 0, 0])
        b = np.array([0, 0, 10, 10])
        assert iou(a, b) == 0.0


class TestBoxConversion:
    def test_roundtrip(self):
        box = np.array([10.0, 20.0, 50.0, 80.0])
        cxcywh = xyxy_to_cxcywh(box)
        back = cxcywh_to_xyxy(cxcywh)
        assert back == pytest.approx(box)

    def test_center_and_size(self):
        box = np.array([0.0, 0.0, 10.0, 20.0])
        cx, cy, w, h = xyxy_to_cxcywh(box)
        assert (cx, cy, w, h) == pytest.approx((5.0, 10.0, 10.0, 20.0))


class TestTrackTtc:
    def _approach(self, start_w=20.0, growth_per_frame=2.0, steps=15, fps=25.0):
        """Симулирует объект, приближающийся к камере: ширина бокса растёт
        линейно на growth_per_frame пикселей за кадр."""
        x1, y1 = 100.0, 100.0
        box = np.array([x1, y1, x1 + start_w, y1 + start_w])
        track = Track(box, cls_id=2, fps=fps)
        w = start_w
        for _ in range(steps):
            track.predict()
            w += growth_per_frame
            box = np.array([x1, y1, x1 + w, y1 + w])
            track.update(box)
        return track

    def _stationary(self, w=20.0, steps=15, fps=25.0):
        x1, y1 = 100.0, 100.0
        box = np.array([x1, y1, x1 + w, y1 + w])
        track = Track(box, cls_id=2, fps=fps)
        for _ in range(steps):
            track.predict()
            track.update(box)
        return track

    def test_approaching_object_has_positive_ttc(self):
        track = self._approach()
        ttc = track.ttc_seconds()
        assert ttc is not None
        assert ttc > 0

    def test_stationary_object_has_no_ttc(self):
        track = self._stationary()
        assert track.ttc_seconds() is None

    def test_freshly_created_track_has_no_ttc(self):
        box = np.array([100.0, 100.0, 120.0, 120.0])
        track = Track(box, cls_id=2, fps=25.0)
        assert track.ttc_seconds() is None

    def test_forecast_path_length(self):
        track = self._approach()
        path = track.forecast_path(steps=5)
        assert len(path) == 5


class TestMultiObjectTracker:
    def test_new_detection_is_not_confirmed_immediately(self):
        tracker = MultiObjectTracker(fps=25.0, min_hits=3)
        box = np.array([100.0, 100.0, 140.0, 140.0])
        confirmed = tracker.update([(box, 2)])
        assert confirmed == []

    def test_detection_confirmed_after_min_hits(self):
        tracker = MultiObjectTracker(fps=25.0, min_hits=3)
        box = np.array([100.0, 100.0, 140.0, 140.0])
        confirmed = []
        for _ in range(3):
            confirmed = tracker.update([(box, 2)])
        assert len(confirmed) == 1
        assert confirmed[0].cls_id == 2

    def test_same_object_keeps_same_id_across_frames(self):
        tracker = MultiObjectTracker(fps=25.0, min_hits=1)
        box = np.array([100.0, 100.0, 140.0, 140.0])
        first = tracker.update([(box, 2)])
        track_id = first[0].id
        # небольшое смещение — всё ещё тот же объект по IoU
        moved = np.array([102.0, 100.0, 142.0, 140.0])
        second = tracker.update([(moved, 2)])
        assert second[0].id == track_id

    def test_track_dropped_after_max_age(self):
        tracker = MultiObjectTracker(fps=25.0, min_hits=1, max_age=2)
        box = np.array([100.0, 100.0, 140.0, 140.0])
        tracker.update([(box, 2)])
        assert len(tracker.tracks) == 1
        for _ in range(3):
            tracker.update([])
        assert tracker.tracks == []
