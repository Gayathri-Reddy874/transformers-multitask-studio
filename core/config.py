"""
Configuration management for the Transformer Playground application.

All tunables (which model backs each task, request size limits, log level,
compute device) are centralized here and driven by environment variables so
the app can be re-pointed at different models or limits without code
changes -- e.g. for a low-memory deployment vs. a GPU server.

Set values via a `.env` file (see `.env.example`) or real environment
variables; sensible defaults are used otherwise.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import torch


def _get_device() -> int:
    """Return the HF `pipeline` device index: -1 for CPU, 0 for the first CUDA GPU."""
    return 0 if torch.cuda.is_available() else -1


def _bool_env(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelConfig:
    """Hugging Face model identifiers, one per supported task."""

    sentiment_model: str = os.getenv(
        "SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
    )
    generation_model: str = os.getenv("GENERATION_MODEL", "gpt2")
    summarization_model: str = os.getenv(
        "SUMMARIZATION_MODEL", "facebook/bart-large-cnn"
    )
    ner_model: str = os.getenv("NER_MODEL", "dslim/bert-base-NER")
    image_classification_model: str = os.getenv(
        "IMAGE_CLASSIFICATION_MODEL", "google/vit-base-patch16-224"
    )
    clip_model: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    asr_model: str = os.getenv("ASR_MODEL", "openai/whisper-small")


@dataclass(frozen=True)
class AppLimits:
    """Guardrails against abusive or accidentally huge inputs."""

    max_text_chars: int = int(os.getenv("MAX_TEXT_CHARS", "5000"))
    max_image_mb: int = int(os.getenv("MAX_IMAGE_MB", "10"))
    max_audio_mb: int = int(os.getenv("MAX_AUDIO_MB", "25"))
    max_batch_lines: int = int(os.getenv("MAX_BATCH_LINES", "50"))


@dataclass(frozen=True)
class Settings:
    models: ModelConfig = field(default_factory=ModelConfig)
    limits: AppLimits = field(default_factory=AppLimits)
    device: int = field(default_factory=_get_device)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_telemetry: bool = _bool_env("ENABLE_TELEMETRY", False)


def configure_logging(settings: "Settings") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    return logging.getLogger("transformer_playground")


SETTINGS = Settings()
LOGGER = configure_logging(SETTINGS)