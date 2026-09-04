"""Modern UI components for camera interface."""

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# Color scheme (BGR format)
class Colors:
    # Main colors
    PRIMARY = (255, 150, 50)      # Orange accent
    SUCCESS = (100, 220, 100)     # Green
    WARNING = (50, 200, 255)      # Yellow/Orange
    DANGER = (80, 80, 255)        # Red

    # UI colors
    BG_DARK = (30, 30, 30)        # Dark background
    BG_PANEL = (45, 45, 45)       # Panel background
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (180, 180, 180)

    # Face box colors
    KNOWN_FACE = (255, 180, 50)   # Cyan-ish for known
    UNKNOWN_FACE = (100, 100, 255) # Red-ish for unknown


def draw_rounded_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    radius: int = 15,
    fill: bool = False,
    alpha: float = 1.0,
) -> np.ndarray:
    """Draw a rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Clamp radius
    radius = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

    if fill and alpha < 1.0:
        # Semi-transparent fill
        overlay = img.copy()

        # Draw filled rounded rect on overlay
        # Top left corner
        cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
        # Top right corner
        cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
        # Bottom right corner
        cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
        # Bottom left corner
        cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)

        # Fill rectangles
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)

        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    elif fill:
        # Solid fill
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    else:
        # Just border
        # Corners
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

        # Lines
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    return img


def draw_corner_brackets(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 3,
    length: int = 20,
) -> None:
    """Draw corner brackets around a region (modern face detection style)."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Top-left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)

    # Top-right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)

    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)

    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def draw_scanning_effect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    progress: float,  # 0.0 to 1.0
) -> None:
    """Draw scanning line effect inside a region."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Scanning line position
    scan_y = int(y1 + (y2 - y1) * progress)

    # Draw gradient line
    for i, alpha in enumerate([0.1, 0.3, 0.6, 1.0, 0.6, 0.3, 0.1]):
        y = scan_y - 3 + i
        if y1 <= y <= y2:
            overlay = img.copy()
            cv2.line(overlay, (x1 + 5, y), (x2 - 5, y), color, 1)
            cv2.addWeighted(overlay, alpha * 0.7, img, 1 - alpha * 0.7, 0, img)


def draw_label_badge(
    img: np.ndarray,
    text: str,
    position: tuple[int, int],
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int] = Colors.TEXT_PRIMARY,
    font_scale: float = 0.6,
    padding: int = 8,
) -> None:
    """Draw a text label with rounded background badge."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position

    # Badge background
    draw_rounded_rect(
        img,
        (x, y - text_h - padding),
        (x + text_w + padding * 2, y + padding // 2),
        bg_color,
        fill=True,
        alpha=0.85,
        radius=8,
    )

    # Text
    cv2.putText(
        img, text,
        (x + padding, y - padding // 2),
        font, font_scale, text_color, thickness, cv2.LINE_AA
    )


def draw_status_bar(
    img: np.ndarray,
    left_text: str,
    right_text: str = "",
    height: int = 40,
) -> None:
    """Draw a status bar at the bottom of the image."""
    h, w = img.shape[:2]

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - height), (w, h), Colors.BG_DARK, -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

    # Accent line at top
    cv2.line(img, (0, h - height), (w, h - height), Colors.PRIMARY, 2)

    # Left text
    cv2.putText(
        img, left_text,
        (15, h - height // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_PRIMARY, 1, cv2.LINE_AA
    )

    # Right text
    if right_text:
        (text_w, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            img, right_text,
            (w - text_w - 15, h - height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_SECONDARY, 1, cv2.LINE_AA
        )


def draw_top_bar(
    img: np.ndarray,
    title: str,
    subtitle: str = "",
    height: int = 50,
) -> None:
    """Draw a title bar at the top of the image."""
    h, w = img.shape[:2]

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, height), Colors.BG_DARK, -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

    # Accent line at bottom
    cv2.line(img, (0, height), (w, height), Colors.PRIMARY, 2)

    # Title
    cv2.putText(
        img, title,
        (15, height // 2 + 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, Colors.TEXT_PRIMARY, 2, cv2.LINE_AA
    )

    # Subtitle (right aligned)
    if subtitle:
        (text_w, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            img, subtitle,
            (w - text_w - 15, height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.TEXT_SECONDARY, 1, cv2.LINE_AA
        )


def draw_notification(
    img: np.ndarray,
    text: str,
    notification_type: str = "info",  # info, success, warning, error
    progress: float = 1.0,  # For fade out animation
) -> None:
    """Draw a notification toast at the top of the screen."""
    h, w = img.shape[:2]

    colors = {
        "info": Colors.PRIMARY,
        "success": Colors.SUCCESS,
        "warning": Colors.WARNING,
        "error": Colors.DANGER,
    }
    color = colors.get(notification_type, Colors.PRIMARY)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, 0.6, 1)

    # Center horizontally
    padding = 20
    box_w = text_w + padding * 2
    box_x = (w - box_w) // 2
    box_y = 60

    # Draw with fade
    alpha = 0.9 * progress
    overlay = img.copy()
    draw_rounded_rect(
        overlay,
        (box_x, box_y),
        (box_x + box_w, box_y + text_h + padding),
        color,
        fill=True,
        radius=10,
    )
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Text
    if progress > 0.3:
        cv2.putText(
            img, text,
            (box_x + padding, box_y + text_h + padding // 2 - 2),
            font, 0.6, Colors.TEXT_PRIMARY, 1, cv2.LINE_AA
        )


@dataclass
class FaceBox:
    """Animated face box state."""
    location: tuple[int, int, int, int]  # top, right, bottom, left
    name: Optional[str] = None
    distance: float = 0.0
    mood: str = ""
    scan_progress: float = 0.0
    is_scanning: bool = False
    last_seen: float = field(default_factory=time.time)

    def get_color(self) -> tuple[int, int, int]:
        if self.name:
            return Colors.KNOWN_FACE
        return Colors.UNKNOWN_FACE


class ModernUI:
    """Modern camera UI renderer."""

    def __init__(self, title: str = "Face Module"):
        self.title = title
        self.faces: dict[str, FaceBox] = {}
        self.notification: Optional[tuple[str, str, float]] = None  # text, type, start_time
        self.notification_duration = 3.0
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0

    def update_fps(self) -> None:
        """Update FPS counter."""
        self.fps_frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.fps_frame_count / (now - self.last_fps_time)
            self.fps_frame_count = 0
            self.last_fps_time = now

    def show_notification(self, text: str, notification_type: str = "info") -> None:
        """Show a notification toast."""
        self.notification = (text, notification_type, time.time())

    def update_faces(
        self,
        results: list[tuple[tuple[int, int, int, int], Optional[str], float]],
        moods: Optional[dict[str, str]] = None,
    ) -> None:
        """Update face tracking state."""
        moods = moods or {}
        now = time.time()

        # Update existing / add new faces
        seen_ids = set()
        for location, name, distance in results:
            face_id = name or f"unknown_{hash(location)}"
            seen_ids.add(face_id)

            if face_id in self.faces:
                # Update existing
                face = self.faces[face_id]
                face.location = location
                face.distance = distance
                face.last_seen = now
                if name and name in moods:
                    face.mood = moods[name]
            else:
                # New face - start scanning animation
                self.faces[face_id] = FaceBox(
                    location=location,
                    name=name,
                    distance=distance,
                    mood=moods.get(name, "") if name else "",
                    is_scanning=True,
                    scan_progress=0.0,
                )

        # Remove old faces
        to_remove = [fid for fid, face in self.faces.items()
                     if now - face.last_seen > 0.5]
        for fid in to_remove:
            del self.faces[fid]

    def render(self, frame: np.ndarray, status_left: str = "", status_right: str = "") -> np.ndarray:
        """Render the modern UI on a frame."""
        self.frame_count += 1
        self.update_fps()

        now = time.time()

        # Draw faces
        for face_id, face in self.faces.items():
            top, right, bottom, left = face.location
            color = face.get_color()

            # Update scanning animation
            if face.is_scanning:
                face.scan_progress += 0.05
                if face.scan_progress >= 1.0:
                    face.is_scanning = False
                    face.scan_progress = 0.0

            # Draw corner brackets
            draw_corner_brackets(frame, (left, top), (right, bottom), color, thickness=2, length=25)

            # Draw scanning effect if active
            if face.is_scanning:
                draw_scanning_effect(frame, (left, top), (right, bottom), color, face.scan_progress)

            # Draw label
            label = face.name or "Unknown"
            if face.mood:
                label = f"{label} • {face.mood}"

            bg_color = Colors.BG_PANEL if face.name else Colors.DANGER
            draw_label_badge(frame, label, (left, top - 10), bg_color)

            # Distance indicator (small)
            dist_text = f"{face.distance:.2f}"
            cv2.putText(
                frame, dist_text,
                (right - 40, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, Colors.TEXT_SECONDARY, 1, cv2.LINE_AA
            )

        # Draw top bar
        subtitle = f"FPS: {self.fps:.0f}"
        draw_top_bar(frame, self.title, subtitle)

        # Draw status bar
        draw_status_bar(frame, status_left or "Press Q to quit", status_right)

        # Draw notification if active
        if self.notification:
            text, ntype, start_time = self.notification
            elapsed = now - start_time
            if elapsed < self.notification_duration:
                # Fade out in last 0.5 seconds
                progress = 1.0
                if elapsed > self.notification_duration - 0.5:
                    progress = (self.notification_duration - elapsed) / 0.5
                draw_notification(frame, text, ntype, progress)
            else:
                self.notification = None

        return frame
