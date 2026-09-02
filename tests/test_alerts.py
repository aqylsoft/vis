import numpy as np
import pytest

from alerts import (
    TTC_COLOR_CAUTION,
    TTC_COLOR_OK,
    TTC_COLOR_WARN,
    filter_bogus_detections,
    in_lane_band,
    is_ttc_eligible,
    ttc_color,
)


class TestFilterBogusDetections:
    def test_keeps_normal_detection(self):
        detections = [(np.array([100, 100, 200, 200]), 2)]
        clean = filter_bogus_detections(detections, frame_w=640, frame_h=480,
                                         max_area_ratio=0.35, road_roi_bottom=0.9)
        assert len(clean) == 1

    def test_drops_oversized_box(self):
        # почти весь кадр — типичный false positive
        detections = [(np.array([0, 0, 630, 470]), 2)]
        clean = filter_bogus_detections(detections, frame_w=640, frame_h=480,
                                         max_area_ratio=0.35, road_roi_bottom=0.9)
        assert clean == []

    def test_drops_box_below_road_roi(self):
        # центр бокса в зоне капота/приборки
        detections = [(np.array([100, 440, 200, 470]), 0)]
        clean = filter_bogus_detections(detections, frame_w=640, frame_h=480,
                                         max_area_ratio=0.35, road_roi_bottom=0.55)
        assert clean == []

    def test_drops_degenerate_box(self):
        detections = [(np.array([100, 100, 100, 100]), 2)]
        clean = filter_bogus_detections(detections, frame_w=640, frame_h=480,
                                         max_area_ratio=0.35, road_roi_bottom=0.9)
        assert clean == []


class TestInLaneBand:
    def test_center_is_in_band(self):
        assert in_lane_band(320, 640, (0.3, 0.7)) is True

    def test_edge_of_frame_is_out(self):
        assert in_lane_band(10, 640, (0.3, 0.7)) is False

    @pytest.mark.parametrize("cx,expected", [(192, True), (191, False), (448, True), (449, False)])
    def test_boundaries_inclusive(self, cx, expected):
        assert in_lane_band(cx, 640, (0.3, 0.7)) is expected


class TestIsTtcEligible:
    """Регрессия: раньше TTC/collision warning считался только для машин,
    и пешеход по курсу прямо перед камерой не давал никакого алерта."""

    def test_pedestrian_in_lane_is_eligible(self):
        assert is_ttc_eligible("person", cx=320, frame_w=640, lane_band=(0.3, 0.7)) is True

    def test_pedestrian_outside_lane_is_not_eligible(self):
        assert is_ttc_eligible("person", cx=10, frame_w=640, lane_band=(0.3, 0.7)) is False

    @pytest.mark.parametrize("cls_name", ["car", "motorcycle", "bus", "truck"])
    def test_vehicle_in_lane_is_eligible(self, cls_name):
        assert is_ttc_eligible(cls_name, cx=320, frame_w=640, lane_band=(0.3, 0.7)) is True

    def test_unknown_class_is_not_eligible(self):
        assert is_ttc_eligible("bicycle", cx=320, frame_w=640, lane_band=(0.3, 0.7)) is False


class TestTtcColor:
    def test_warn_threshold(self):
        assert ttc_color(2.5, warn_s=2.5, caution_s=4.5) == TTC_COLOR_WARN

    def test_caution_threshold(self):
        assert ttc_color(4.5, warn_s=2.5, caution_s=4.5) == TTC_COLOR_CAUTION

    def test_ok_above_caution(self):
        assert ttc_color(10.0, warn_s=2.5, caution_s=4.5) == TTC_COLOR_OK

    def test_below_warn(self):
        assert ttc_color(0.1, warn_s=2.5, caution_s=4.5) == TTC_COLOR_WARN
