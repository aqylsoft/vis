"""Auto-lock screen when user leaves, unlock prompt when returns."""

import subprocess
import time
from typing import Optional

import cv2

from .detector import FaceDetector
from .face_db import FaceDatabase


def lock_screen() -> bool:
    """Lock the screen. Returns True if successful."""
    # Try different lock commands for various Linux DEs
    commands = [
        ["loginctl", "lock-session"],  # systemd (works on most modern distros)
        ["gnome-screensaver-command", "-l"],  # GNOME
        ["xdg-screensaver", "lock"],  # Generic
        ["dm-tool", "lock"],  # LightDM
    ]

    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    return False


def notify(title: str, message: str) -> None:
    """Show desktop notification."""
    try:
        subprocess.run(
            ["notify-send", "-u", "normal", "-t", "3000", title, message],
            capture_output=True,
            timeout=5
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def run_autolock(
    camera_id: int = 0,
    lock_timeout: float = 10.0,
    check_interval: float = 0.5,
    show_preview: bool = False,
    tolerance: float = 0.6,
) -> int:
    """Run auto-lock daemon.

    Args:
        camera_id: Camera device ID.
        lock_timeout: Seconds without recognized face before locking.
        check_interval: Seconds between face checks.
        show_preview: Show camera preview window.
        tolerance: Face matching tolerance.

    Returns:
        Exit code.
    """
    db = FaceDatabase()

    if db.is_empty():
        print("Error: No faces registered. Use 'make face-register' first.")
        return 1

    detector = FaceDetector(db, tolerance=tolerance)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return 1

    print(f"Auto-lock active (timeout: {lock_timeout}s)")
    print("Press Q to quit" if show_preview else "Press Ctrl+C to quit")

    last_seen: Optional[float] = None
    is_locked = False
    last_check = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror
            frame = cv2.flip(frame, 1)

            now = time.time()

            # Check faces at interval (saves CPU)
            if now - last_check >= check_interval:
                last_check = now
                results = detector.recognize_faces(frame)

                # Separate known and unknown faces
                known_faces = [(loc, name, dist) for loc, name, dist in results if name is not None]
                unknown_faces = [(loc, name, dist) for loc, name, dist in results if name is None]

                # INTRUDER ALERT: Unknown face detected - instant lock!
                if unknown_faces and not is_locked:
                    print(f"[{time.strftime('%H:%M:%S')}] INTRUDER DETECTED - locking immediately!")
                    if lock_screen():
                        is_locked = True
                        notify("INTRUDER!", "Unknown person detected - screen locked")
                    else:
                        print("Warning: Failed to lock screen")
                        is_locked = True
                    last_seen = None  # Reset
                    continue

                # Check if any known face is present
                known_present = len(known_faces) > 0

                if known_present:
                    if last_seen is None:
                        # Just returned
                        if is_locked:
                            notify("Welcome back!", "Screen was locked while you were away")
                            print(f"[{time.strftime('%H:%M:%S')}] User returned")
                            is_locked = False
                    last_seen = now
                else:
                    if last_seen is not None:
                        absence_time = now - last_seen

                        if absence_time >= lock_timeout and not is_locked:
                            print(f"[{time.strftime('%H:%M:%S')}] User absent for {lock_timeout}s - locking screen")
                            if lock_screen():
                                is_locked = True
                                notify("Screen locked", "You left the camera view")
                            else:
                                print("Warning: Failed to lock screen")
                                is_locked = True  # Don't spam lock attempts

            # Show preview if enabled
            if show_preview:
                # Draw face boxes
                for (top, right, bottom, left), name, dist in results:
                    if name:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, "INTRUDER", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Draw status
                if is_locked:
                    status = "LOCKED"
                    color = (0, 0, 255)
                elif last_seen is not None:
                    absence = now - last_seen
                    if absence < lock_timeout:
                        status = f"OK (lock in {lock_timeout - absence:.0f}s if leave)"
                        color = (0, 255, 0)
                    else:
                        status = "Locking..."
                        color = (0, 165, 255)
                else:
                    status = "Waiting for known face..."
                    color = (0, 165, 255)

                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("Auto-Lock", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Small delay when no preview
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

    return 0
