"""Face recognition module for detecting and identifying known faces via webcam.

Usage:
    python -m face_module register --name "Name"   # Register face from camera
    python -m face_module register -n "Name" -i photo.jpg  # Register from image
    python -m face_module list                     # List registered faces
    python -m face_module remove "Name"            # Remove a face
    python -m face_module watch                    # Watch camera for known faces
"""

from .face_db import FaceDatabase

__all__ = ["FaceDatabase"]


def get_detector():
    """Get FaceDetector (lazy import to avoid loading face_recognition until needed)."""
    from .detector import FaceDetector
    return FaceDetector
