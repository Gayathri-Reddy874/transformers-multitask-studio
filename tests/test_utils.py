"""
Unit tests for core.utils — pure functions only, so these run in milliseconds
with no model downloads or GPU/CPU inference required.

Run with:
    pytest tests/
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

# Make the project root importable when running `pytest` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.utils import (  # noqa: E402
    ValidationError,
    format_score,
    split_batch_lines,
    validate_audio_file,
    validate_image_file,
    validate_text,
)


def test_validate_text_rejects_empty():
    with pytest.raises(ValidationError):
        validate_text("   ", max_chars=100)


def test_validate_text_rejects_too_long():
    with pytest.raises(ValidationError):
        validate_text("a" * 10, max_chars=5)


def test_validate_text_strips_and_returns():
    assert validate_text("  hello  ", max_chars=100) == "hello"


def test_split_batch_lines_basic():
    assert split_batch_lines("a\nb\n\nc", max_lines=10) == ["a", "b", "c"]


def test_split_batch_lines_rejects_too_many():
    with pytest.raises(ValidationError):
        split_batch_lines("\n".join(str(i) for i in range(5)), max_lines=3)


def test_split_batch_lines_rejects_all_blank():
    with pytest.raises(ValidationError):
        split_batch_lines("   \n   ", max_lines=10)


def _make_png_bytes(size=(10, 10)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color="red").save(buf, format="PNG")
    return buf.getvalue()


def test_validate_image_file_accepts_valid_png():
    image = validate_image_file(_make_png_bytes(), max_mb=10)
    assert image.size == (10, 10)
    assert image.mode == "RGB"


def test_validate_image_file_rejects_garbage_bytes():
    with pytest.raises(ValidationError):
        validate_image_file(b"this is not an image", max_mb=10)


def test_validate_audio_file_rejects_empty_bytes():
    with pytest.raises(ValidationError):
        validate_audio_file(b"", max_mb=10)


def test_validate_audio_file_accepts_nonempty_bytes():
    assert validate_audio_file(b"\x00\x01\x02", max_mb=10) == b"\x00\x01\x02"


def test_format_score():
    assert format_score(0.5) == "50.00%"
    assert format_score(1.0) == "100.00%"
    assert format_score(0.0) == "0.00%"