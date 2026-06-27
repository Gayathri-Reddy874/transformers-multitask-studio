"""
Pure-Python utility helpers: input validation and small formatting functions.

Deliberately free of any Streamlit or model imports so they can be unit
tested in isolation (see tests/test_utils.py) without loading any ML weights.
"""

from __future__ import annotations

import io
from typing import List

from PIL import Image, UnidentifiedImageError


class ValidationError(ValueError):
    """Raised when user-supplied input fails validation. Safe to show to users."""


def validate_text(text: str, max_chars: int) -> str:
    """Reject empty or oversized text input; return the trimmed text."""
    if not text or not text.strip():
        raise ValidationError("Input text is empty. Please enter some text.")
    if len(text) > max_chars:
        raise ValidationError(
            f"Input text is too long ({len(text)} chars). Limit is {max_chars} characters."
        )
    return text.strip()


def split_batch_lines(text: str, max_lines: int) -> List[str]:
    """Split newline-separated input into a validated list of non-empty lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValidationError("Please provide at least one non-empty line.")
    if len(lines) > max_lines:
        raise ValidationError(f"Too many lines ({len(lines)}). Limit is {max_lines}.")
    return lines


def validate_image_file(file_bytes: bytes, max_mb: int) -> Image.Image:
    """Validate size/format of an uploaded image and return it as an RGB PIL Image."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValidationError(
            f"Image is {size_mb:.1f} MB, which exceeds the {max_mb} MB limit."
        )
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValidationError("The uploaded file is not a valid image.") from exc
    return image.convert("RGB")


def validate_audio_file(file_bytes: bytes, max_mb: int) -> bytes:
    """Validate size/non-emptiness of an uploaded audio file and return its bytes."""
    if not file_bytes:
        raise ValidationError("The uploaded audio file is empty.")
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValidationError(
            f"Audio file is {size_mb:.1f} MB, which exceeds the {max_mb} MB limit."
        )
    return file_bytes


def format_score(score: float) -> str:
    """Format a 0-1 confidence score as a percentage string, e.g. 0.5 -> '50.00%'."""
    return f"{score * 100:.2f}%"