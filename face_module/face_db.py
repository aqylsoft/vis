"""Face database management - stores face encodings with names."""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

# Default database location
DEFAULT_DB_DIR = Path.home() / ".face_module"
DEFAULT_DB_FILE = DEFAULT_DB_DIR / "faces.json"
DEFAULT_ENCODINGS_FILE = DEFAULT_DB_DIR / "encodings.npy"


class FaceDatabase:
    """Manages a database of known face encodings."""

    def __init__(self, db_dir: Optional[Path] = None):
        """Initialize database.

        Args:
            db_dir: Directory to store database files. Defaults to ~/.face_module
        """
        self.db_dir = Path(db_dir) if db_dir else DEFAULT_DB_DIR
        self.db_file = self.db_dir / "faces.json"
        self.encodings_file = self.db_dir / "encodings.npy"

        self.names: list[str] = []
        self.encodings: Optional[np.ndarray] = None

        self._ensure_db_dir()
        self._load()

    def _ensure_db_dir(self) -> None:
        """Create database directory if it doesn't exist."""
        self.db_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """Load database from disk."""
        if self.db_file.exists():
            with open(self.db_file, "r") as f:
                data = json.load(f)
                self.names = data.get("names", [])

        if self.encodings_file.exists():
            self.encodings = np.load(self.encodings_file)
        else:
            self.encodings = None

    def _save(self) -> None:
        """Save database to disk."""
        with open(self.db_file, "w") as f:
            json.dump({"names": self.names}, f, indent=2)

        if self.encodings is not None:
            np.save(self.encodings_file, self.encodings)

    def add_face(self, name: str, encoding: np.ndarray) -> int:
        """Add a face encoding to the database.

        Args:
            name: Name to associate with this face.
            encoding: 128-dimensional face encoding from face_recognition.

        Returns:
            Index of the added face.
        """
        self.names.append(name)

        if self.encodings is None:
            self.encodings = encoding.reshape(1, -1)
        else:
            self.encodings = np.vstack([self.encodings, encoding])

        self._save()
        return len(self.names) - 1

    def remove_face(self, name: str) -> bool:
        """Remove all faces with the given name.

        Args:
            name: Name to remove.

        Returns:
            True if any faces were removed.
        """
        indices_to_remove = [i for i, n in enumerate(self.names) if n == name]

        if not indices_to_remove:
            return False

        # Remove from names
        self.names = [n for i, n in enumerate(self.names) if i not in indices_to_remove]

        # Remove from encodings
        if self.encodings is not None:
            mask = np.ones(len(self.encodings), dtype=bool)
            mask[indices_to_remove] = False
            self.encodings = self.encodings[mask] if mask.any() else None

        self._save()
        return True

    def list_faces(self) -> list[str]:
        """Get list of unique names in the database."""
        return sorted(set(self.names))

    def get_all_encodings(self) -> tuple[list[str], Optional[np.ndarray]]:
        """Get all names and encodings.

        Returns:
            Tuple of (names list, encodings array or None if empty).
        """
        return self.names, self.encodings

    def is_empty(self) -> bool:
        """Check if database has no faces."""
        return len(self.names) == 0

    def count(self) -> int:
        """Get total number of face encodings."""
        return len(self.names)