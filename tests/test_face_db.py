"""Tests for face database."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from face_module.face_db import FaceDatabase


@pytest.fixture
def temp_db():
    """Create a temporary database directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield FaceDatabase(db_dir=Path(tmpdir))


def test_empty_database(temp_db):
    """Test empty database state."""
    assert temp_db.is_empty()
    assert temp_db.count() == 0
    assert temp_db.list_faces() == []


def test_add_face(temp_db):
    """Test adding a face."""
    encoding = np.random.randn(128)
    idx = temp_db.add_face("Alice", encoding)

    assert idx == 0
    assert not temp_db.is_empty()
    assert temp_db.count() == 1
    assert temp_db.list_faces() == ["Alice"]


def test_add_multiple_faces(temp_db):
    """Test adding multiple faces."""
    temp_db.add_face("Alice", np.random.randn(128))
    temp_db.add_face("Bob", np.random.randn(128))
    temp_db.add_face("Alice", np.random.randn(128))  # Second encoding for Alice

    assert temp_db.count() == 3
    assert temp_db.list_faces() == ["Alice", "Bob"]


def test_remove_face(temp_db):
    """Test removing a face."""
    temp_db.add_face("Alice", np.random.randn(128))
    temp_db.add_face("Bob", np.random.randn(128))

    assert temp_db.remove_face("Alice")
    assert temp_db.count() == 1
    assert temp_db.list_faces() == ["Bob"]


def test_remove_nonexistent_face(temp_db):
    """Test removing a face that doesn't exist."""
    temp_db.add_face("Alice", np.random.randn(128))
    assert not temp_db.remove_face("Charlie")
    assert temp_db.count() == 1


def test_get_all_encodings(temp_db):
    """Test getting all encodings."""
    enc1 = np.random.randn(128)
    enc2 = np.random.randn(128)

    temp_db.add_face("Alice", enc1)
    temp_db.add_face("Bob", enc2)

    names, encodings = temp_db.get_all_encodings()

    assert names == ["Alice", "Bob"]
    assert encodings.shape == (2, 128)
    np.testing.assert_array_almost_equal(encodings[0], enc1)
    np.testing.assert_array_almost_equal(encodings[1], enc2)


def test_persistence(temp_db):
    """Test that database persists to disk."""
    encoding = np.random.randn(128)
    temp_db.add_face("Alice", encoding)

    # Create new database instance pointing to same directory
    db2 = FaceDatabase(db_dir=temp_db.db_dir)

    assert db2.count() == 1
    assert db2.list_faces() == ["Alice"]

    names, encodings = db2.get_all_encodings()
    np.testing.assert_array_almost_equal(encodings[0], encoding)
