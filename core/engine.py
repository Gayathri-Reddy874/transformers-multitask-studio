"""
Model loading and inference engine for the Transformer Playground.

Design notes
------------
- **Lazy loading**: the original app loaded sentiment, generation, image-
  classification, ASR, and CLIP models all at startup, regardless of which
  task the user actually wanted. Here, each model is behind its own
  `@st.cache_resource`-decorated getter, so a model is only ever downloaded
  and loaded into memory the first time its task is used -- and then reused
  for the lifetime of the server process.
- **Separation of concerns**: this module has no `st.text_area`, `st.button`,
  etc. -- only model construction and inference. `app.py` owns presentation.
- **Uniform error handling**: every inference function catches model/runtime
  errors and re-raises them as `InferenceError` with a user-safe message,
  after logging the full traceback for operators.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List

import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, pipeline

from core.config import LOGGER, SETTINGS


@dataclass
class InferenceResult:
    """Standard envelope returned by every `run_*` function below."""

    output: Any
    latency_ms: float
    model_name: str
    warning: str | None = None


class InferenceError(RuntimeError):
    """Raised when a model call fails after input validation has already passed."""


# ----------------------------------------------------------------------------
# Lazy, cached model loaders.
# Each is its own `st.cache_resource` entry, keyed by function identity, so
# the app never pays the cost (time, memory, disk, bandwidth) of loading a
# model the user hasn't asked for, and never reloads one it already has.
# ----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    LOGGER.info("Loading sentiment model: %s", SETTINGS.models.sentiment_model)
    return pipeline(
        "sentiment-analysis", model=SETTINGS.models.sentiment_model, device=SETTINGS.device
    )


@st.cache_resource(show_spinner=False)
def get_generation_pipeline():
    LOGGER.info("Loading text-generation model: %s", SETTINGS.models.generation_model)
    return pipeline(
        "text-generation", model=SETTINGS.models.generation_model, device=SETTINGS.device
    )


@st.cache_resource(show_spinner=False)
def get_summarization_pipeline():
    LOGGER.info("Loading summarization model: %s", SETTINGS.models.summarization_model)
    return pipeline(
        "summarization", model=SETTINGS.models.summarization_model, device=SETTINGS.device
    )


@st.cache_resource(show_spinner=False)
def get_ner_pipeline():
    LOGGER.info("Loading NER model: %s", SETTINGS.models.ner_model)
    return pipeline(
        "ner",
        model=SETTINGS.models.ner_model,
        aggregation_strategy="simple",
        device=SETTINGS.device,
    )


@st.cache_resource(show_spinner=False)
def get_image_classification_pipeline():
    LOGGER.info(
        "Loading image-classification model: %s", SETTINGS.models.image_classification_model
    )
    return pipeline(
        "image-classification",
        model=SETTINGS.models.image_classification_model,
        device=SETTINGS.device,
    )


@st.cache_resource(show_spinner=False)
def get_clip():
    LOGGER.info("Loading CLIP model: %s", SETTINGS.models.clip_model)
    model = CLIPModel.from_pretrained(SETTINGS.models.clip_model)
    processor = CLIPProcessor.from_pretrained(SETTINGS.models.clip_model)
    device = "cuda" if SETTINGS.device >= 0 else "cpu"
    model.to(device)
    model.eval()
    return model, processor, device


@st.cache_resource(show_spinner=False)
def get_asr_pipeline():
    LOGGER.info("Loading ASR model: %s", SETTINGS.models.asr_model)
    return pipeline(
        "automatic-speech-recognition", model=SETTINGS.models.asr_model, device=SETTINGS.device
    )


# ----------------------------------------------------------------------------
# Inference functions: validated input in, InferenceResult out, never a raw
# stack trace leaked to the UI.
# ----------------------------------------------------------------------------


def run_sentiment(texts: List[str]) -> InferenceResult:
    nlp = get_sentiment_pipeline()
    try:
        start = time.perf_counter()
        output = nlp(texts)
        elapsed = (time.perf_counter() - start) * 1000
        return InferenceResult(output, elapsed, SETTINGS.models.sentiment_model)
    except Exception as exc:  # noqa: BLE001 - intentionally broad at the model boundary
        LOGGER.exception("Sentiment inference failed")
        raise InferenceError(f"Sentiment analysis failed: {exc}") from exc


def run_generation(
    prompt: str, max_new_tokens: int, temperature: float, top_p: float, num_sequences: int
) -> InferenceResult:
    gen = get_generation_pipeline()
    try:
        # Greedy decoding can't return more than one sequence, so force
        # sampling on whenever the user asked for multiple sequences or a
        # non-zero temperature. Without this guard, HF raises a ValueError.
        do_sample = temperature > 0 or num_sequences > 1
        start = time.perf_counter()
        output = gen(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01) if do_sample else temperature,
            top_p=top_p,
            num_return_sequences=num_sequences,
            do_sample=do_sample,
            pad_token_id=gen.tokenizer.eos_token_id,
        )
        elapsed = (time.perf_counter() - start) * 1000
        return InferenceResult(output, elapsed, SETTINGS.models.generation_model)
    except Exception as exc:
        LOGGER.exception("Text generation failed")
        raise InferenceError(f"Text generation failed: {exc}") from exc


def run_summarization(text: str, max_length: int, min_length: int) -> InferenceResult:
    summarizer = get_summarization_pipeline()
    try:
        # Summarization models have a hard limit on input length (1024 tokens
        # for facebook/bart-large-cnn). Without `truncation=True`, text longer
        # than that raises a low-level "index out of range in self" error
        # from the positional embedding lookup instead of a clean message.
        # We truncate safely and tell the user when that happens.
        tokenizer = summarizer.tokenizer
        model_max = getattr(tokenizer, "model_max_length", 1024)
        if not isinstance(model_max, int) or model_max > 100_000:
            model_max = 1024
        token_count = len(tokenizer.encode(text))
        warning = None
        if token_count > model_max:
            warning = (
                f"Input was ~{token_count} tokens, longer than this model's {model_max}-token "
                "limit, so only the first part was used. For long documents, split the text "
                "into smaller chunks and summarize each one separately."
            )

        start = time.perf_counter()
        output = summarizer(
            text, max_length=max_length, min_length=min_length, do_sample=False, truncation=True
        )
        elapsed = (time.perf_counter() - start) * 1000
        return InferenceResult(output, elapsed, SETTINGS.models.summarization_model, warning=warning)
    except Exception as exc:
        LOGGER.exception("Summarization failed")
        raise InferenceError(f"Summarization failed: {exc}") from exc


def run_ner(text: str) -> InferenceResult:
    ner = get_ner_pipeline()
    try:
        start = time.perf_counter()
        output = ner(text)
        elapsed = (time.perf_counter() - start) * 1000
        return InferenceResult(output, elapsed, SETTINGS.models.ner_model)
    except Exception as exc:
        LOGGER.exception("NER failed")
        raise InferenceError(f"Entity extraction failed: {exc}") from exc


def run_image_classification(image: Image.Image, top_k: int) -> InferenceResult:
    classifier = get_image_classification_pipeline()
    try:
        start = time.perf_counter()
        output = classifier(image, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        return InferenceResult(output, elapsed, SETTINGS.models.image_classification_model)
    except Exception as exc:
        LOGGER.exception("Image classification failed")
        raise InferenceError(f"Image classification failed: {exc}") from exc


def run_clip_zero_shot(image: Image.Image, candidate_labels: List[str]) -> InferenceResult:
    """Classify an image against an arbitrary, user-supplied set of text labels."""
    model, processor, device = get_clip()
    try:
        start = time.perf_counter()
        inputs = processor(
            text=candidate_labels, images=image, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()
        ranked = sorted(zip(candidate_labels, probs), key=lambda pair: pair[1], reverse=True)
        elapsed = (time.perf_counter() - start) * 1000
        return InferenceResult(ranked, elapsed, SETTINGS.models.clip_model)
    except Exception as exc:
        LOGGER.exception("CLIP zero-shot classification failed")
        raise InferenceError(f"Zero-shot classification failed: {exc}") from exc


def run_asr(audio_bytes: bytes) -> InferenceResult:
    asr = get_asr_pipeline()
    try:
        start = time.perf_counter()
        output = asr(audio_bytes)
        elapsed = (time.perf_counter() - start) * 1000
        return InferenceResult(output, elapsed, SETTINGS.models.asr_model)
    except Exception as exc:
        LOGGER.exception("ASR failed")
        raise InferenceError(f"Speech recognition failed: {exc}") from exc