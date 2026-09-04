"""Face detection and recognition logic."""

from typing import Optional

import cv2
import face_recognition
import numpy as np

from .face_db import FaceDatabase


class FaceDetector:
    """Detects and recognizes faces using face_recognition library."""

    def __init__(
        self,
        db: FaceDatabase,
        tolerance: float = 0.6,
        model: str = "hog",
    ):
        """Initialize detector.

        Args:
            db: Face database with known encodings.
            tolerance: Distance threshold for face matching. Lower = stricter.
            model: Face detection model - "hog" (fast, CPU) or "cnn" (accurate, GPU).
        """
        self.db = db
        self.tolerance = tolerance
        self.model = model

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect face locations in a frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            List of face locations as (top, right, bottom, left) tuples.
        """
        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return face_recognition.face_locations(rgb_frame, model=self.model)

    def encode_face(self, frame: np.ndarray, location: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Get face encoding for a specific location.

        Args:
            frame: BGR image from OpenCV.
            location: Face location as (top, right, bottom, left).

        Returns:
            128-dimensional face encoding, or None if encoding failed.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame, [location])
        return encodings[0] if encodings else None

    def recognize_faces(
        self, frame: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int], Optional[str], float]]:
        """Detect and recognize faces in a frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            List of (location, name, distance) tuples.
            name is None if face is unknown.
            distance is the face distance (lower = more similar).
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        locations = face_recognition.face_locations(rgb_frame, model=self.model)
        if not locations:
            return []

        # Get encodings
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        results = []
        known_names, known_encodings = self.db.get_all_encodings()

        for location, encoding in zip(locations, encodings):
            name = None
            best_distance = float("inf")

            if known_encodings is not None and len(known_encodings) > 0:
                # Calculate distances to all known faces
                distances = face_recognition.face_distance(known_encodings, encoding)
                best_idx = np.argmin(distances)
                best_distance = distances[best_idx]

                if best_distance <= self.tolerance:
                    name = known_names[best_idx]

            results.append((location, name, best_distance))

        return results

    def capture_face_from_camera(
        self, camera_id: int | str = 0, window_name: str = "Capture Face"
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Capture a face from camera interactively.

        Shows camera feed, user presses SPACE to capture when face is detected.

        Args:
            camera_id: Camera device ID.
            window_name: Window title.

        Returns:
            Tuple of (frame, encoding) or None if cancelled.
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        # Check if phone camera (URL) - no mirror needed
        is_phone = isinstance(camera_id, str) and camera_id.startswith("http")

        print("Press SPACE to capture face, Q to cancel")

        result = None
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Mirror the frame (selfie mode) - skip for phone
                if not is_phone:
                    frame = cv2.flip(frame, 1)

                # Detect faces
                locations = self.detect_faces(frame)

                # Draw rectangles around faces
                display_frame = frame.copy()
                for top, right, bottom, left in locations:
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Show face count
                text = f"Faces: {len(locations)}"
                cv2.putText(
                    display_frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

                if not locations:
                    cv2.putText(
                        display_frame, "No face detected", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )
                elif len(locations) > 1:
                    cv2.putText(
                        display_frame, "Multiple faces - show only one", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                    )
                else:
                    cv2.putText(
                        display_frame, "Press SPACE to capture", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" ") and len(locations) == 1:
                    encoding = self.encode_face(frame, locations[0])
                    if encoding is not None:
                        result = (frame, encoding)
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return result