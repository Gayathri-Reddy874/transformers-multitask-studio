# -*- coding: utf-8 -*-
"""
Transformer Playground — an industrial-grade Streamlit front end for
Hugging Face Transformers pipelines covering text, vision, and speech tasks.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Configuration is environment-driven — see core/config.py and .env.example.
This file is presentation-only: all model loading and inference logic lives
in core/engine.py, and all input validation lives in core/utils.py.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import streamlit as st
import torch

from core import engine
from core.config import SETTINGS
from core.utils import (
    ValidationError,
    format_score,
    split_batch_lines,
    validate_audio_file,
    validate_image_file,
    validate_text,
)

st.set_page_config(
    page_title="Transformer Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Session state: a rolling log of recent requests, shown in the sidebar.
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


def _log_history(task: str, summary: str, latency_ms: float) -> None:
    st.session_state.history.insert(
        0,
        {
            "task": task,
            "summary": summary,
            "latency_ms": round(latency_ms, 1),
            "timestamp": time.strftime("%H:%M:%S"),
        },
    )
    st.session_state.history = st.session_state.history[:25]


def _download_button(data, filename: str, label: str) -> None:
    if isinstance(data, (dict, list)):
        payload = json.dumps(data, indent=2, default=str).encode("utf-8")
        mime = "application/json"
    else:
        payload = str(data).encode("utf-8")
        mime = "text/plain"
    st.download_button(label, data=payload, file_name=filename, mime=mime)


# ----------------------------------------------------------------------------
# Sidebar: task picker, device/model status, recent activity
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 Transformer Playground")
    st.caption("Industrial-grade NLP / Vision / Speech demo built on 🤗 Transformers")

    device_label = "GPU (CUDA)" if SETTINGS.device >= 0 else "CPU"
    st.success(f"Compute device: **{device_label}**")
    if SETTINGS.device >= 0:
        st.caption(torch.cuda.get_device_name(0))

    task = st.selectbox(
        "Choose a task",
        [
            "Sentiment Analysis",
            "Text Generation",
            "Summarization",
            "Named Entity Recognition",
            "Image Classification",
            "Zero-Shot Image Classification (CLIP)",
            "Automatic Speech Recognition",
        ],
    )

    with st.expander("⚙️ Model registry"):
        st.json(
            {
                "sentiment": SETTINGS.models.sentiment_model,
                "generation": SETTINGS.models.generation_model,
                "summarization": SETTINGS.models.summarization_model,
                "ner": SETTINGS.models.ner_model,
                "image_classification": SETTINGS.models.image_classification_model,
                "clip": SETTINGS.models.clip_model,
                "asr": SETTINGS.models.asr_model,
            }
        )
        st.caption("Override any of these via environment variables — see .env.example.")

    with st.expander("🕒 Recent activity", expanded=bool(st.session_state.history)):
        if st.session_state.history:
            st.dataframe(
                pd.DataFrame(st.session_state.history),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No requests yet.")

st.title(task)

# ----------------------------------------------------------------------------
# Sentiment Analysis — now batch-capable (one item per line)
# ----------------------------------------------------------------------------
if task == "Sentiment Analysis":
    st.caption("Enter one or more lines of text. Each line is scored independently.")
    text = st.text_area(
        "Input text (one item per line)",
        "I love using transformers for AI projects\n"
        "This product completely broke after one use.",
        height=140,
    )

    if st.button("Analyze", type="primary"):
        try:
            lines = split_batch_lines(text, SETTINGS.limits.max_batch_lines)
            with st.spinner("Running sentiment analysis…"):
                result = engine.run_sentiment(lines)
            df = pd.DataFrame(
                {
                    "text": lines,
                    "label": [r["label"] for r in result.output],
                    "confidence": [format_score(r["score"]) for r in result.output],
                }
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Model: `{result.model_name}` · {result.latency_ms:.0f} ms")
            _download_button(df.to_dict(orient="records"), "sentiment_results.json", "⬇️ Download results")
            _log_history("Sentiment", f"{len(lines)} line(s) analyzed", result.latency_ms)
        except ValidationError as exc:
            st.warning(str(exc))
        except engine.InferenceError as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------------
# Text Generation — configurable sampling parameters
# ----------------------------------------------------------------------------
elif task == "Text Generation":
    prompt = st.text_area("Prompt", "Once upon a time, in a world powered by AI,")

    col1, col2, col3, col4 = st.columns(4)
    max_new_tokens = col1.slider("Max new tokens", 10, 200, 60)
    temperature = col2.slider("Temperature", 0.0, 1.5, 0.8, 0.1)
    top_p = col3.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    num_sequences = col4.slider("Sequences", 1, 3, 1)

    if st.button("Generate", type="primary"):
        try:
            clean_prompt = validate_text(prompt, SETTINGS.limits.max_text_chars)
            with st.spinner("Generating…"):
                result = engine.run_generation(
                    clean_prompt, max_new_tokens, temperature, top_p, num_sequences
                )
            for i, seq in enumerate(result.output, start=1):
                st.markdown(f"**Sequence {i}**")
                st.write(seq["generated_text"])
                st.divider()
            st.caption(f"Model: `{result.model_name}` · {result.latency_ms:.0f} ms")
            _download_button(result.output, "generated_text.json", "⬇️ Download results")
            _log_history("Generation", clean_prompt[:40], result.latency_ms)
        except ValidationError as exc:
            st.warning(str(exc))
        except engine.InferenceError as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------------
# Summarization — new task, not present in the original app
# ----------------------------------------------------------------------------
elif task == "Summarization":
    text = st.text_area(
        "Text to summarize", height=220, placeholder="Paste an article or long passage…"
    )
    col1, col2 = st.columns(2)
    max_length = col1.slider("Max summary length", 30, 300, 120)
    min_length = col2.slider("Min summary length", 10, 100, 30)

    if st.button("Summarize", type="primary"):
        try:
            clean_text = validate_text(text, SETTINGS.limits.max_text_chars)
            with st.spinner("Summarizing…"):
                result = engine.run_summarization(clean_text, max_length, min_length)
            summary = result.output[0]["summary_text"]
            st.markdown("**Summary**")
            st.success(summary)
            if result.warning:
                st.info(result.warning)
            st.caption(
                f"Model: `{result.model_name}` · {result.latency_ms:.0f} ms · "
                f"{len(clean_text.split())} → {len(summary.split())} words"
            )
            _download_button(summary, "summary.txt", "⬇️ Download summary")
            _log_history("Summarization", f"{len(clean_text.split())} words → summary", result.latency_ms)
        except ValidationError as exc:
            st.warning(str(exc))
        except engine.InferenceError as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------------
# Named Entity Recognition — new task
# ----------------------------------------------------------------------------
elif task == "Named Entity Recognition":
    text = st.text_area("Text", "Anthropic was founded in San Francisco and released Claude.")

    if st.button("Extract entities", type="primary"):
        try:
            clean_text = validate_text(text, SETTINGS.limits.max_text_chars)
            with st.spinner("Extracting entities…"):
                result = engine.run_ner(clean_text)
            if not result.output:
                st.info("No entities found.")
            else:
                df = pd.DataFrame(result.output)[["word", "entity_group", "score"]]
                df["score"] = df["score"].apply(format_score)
                df.columns = ["Entity text", "Type", "Confidence"]
                st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Model: `{result.model_name}` · {result.latency_ms:.0f} ms")
            _download_button(result.output, "entities.json", "⬇️ Download results")
            _log_history("NER", f"{len(result.output)} entities found", result.latency_ms)
        except ValidationError as exc:
            st.warning(str(exc))
        except engine.InferenceError as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------------
# Image Classification — now shows top-K with a confidence chart
# ----------------------------------------------------------------------------
elif task == "Image Classification":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    top_k = st.slider("Top-K predictions", 1, 10, 5)

    if uploaded_file is not None:
        try:
            image = validate_image_file(uploaded_file.getvalue(), SETTINGS.limits.max_image_mb)
            st.image(image, caption="Uploaded image", use_container_width=True)

            if st.button("Classify", type="primary"):
                with st.spinner("Classifying…"):
                    result = engine.run_image_classification(image, top_k)
                df = pd.DataFrame(result.output)
                st.bar_chart(df.set_index("label")["score"])
                df["score"] = df["score"].apply(format_score)
                df.columns = ["Label", "Confidence"]
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(f"Model: `{result.model_name}` · {result.latency_ms:.0f} ms")
                _download_button(result.output, "image_classification.json", "⬇️ Download results")
                _log_history("Image classification", df.iloc[0]["Label"], result.latency_ms)
        except ValidationError as exc:
            st.warning(str(exc))
        except engine.InferenceError as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------------
# Zero-Shot Image Classification — finally puts the loaded CLIP model to use
# ----------------------------------------------------------------------------
elif task == "Zero-Shot Image Classification (CLIP)":
    st.caption(
        "Classify an image against any labels you choose on the fly — "
        "no fixed, pre-trained label set required."
    )
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    labels_text = st.text_input(
        "Candidate labels (comma-separated)", "cat, dog, car, airplane, person, food"
    )

    if uploaded_file is not None:
        try:
            image = validate_image_file(uploaded_file.getvalue(), SETTINGS.limits.max_image_mb)
            st.image(image, caption="Uploaded image", use_container_width=True)
            candidate_labels = [label.strip() for label in labels_text.split(",") if label.strip()]

            if st.button("Classify with CLIP", type="primary"):
                if len(candidate_labels) < 2:
                    st.warning("Please provide at least 2 candidate labels.")
                else:
                    with st.spinner("Running CLIP zero-shot classification…"):
                        result = engine.run_clip_zero_shot(image, candidate_labels)
                    df = pd.DataFrame(result.output, columns=["label", "score"])
                    st.bar_chart(df.set_index("label")["score"])
                    df["score"] = df["score"].apply(format_score)
                    df.columns = ["Label", "Confidence"]
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.caption(f"Model: `{result.model_name}` · {result.latency_ms:.0f} ms")
                    _download_button(result.output, "clip_zero_shot.json", "⬇️ Download results")
                    _log_history("CLIP zero-shot", df.iloc[0]["Label"], result.latency_ms)
        except ValidationError as exc:
            st.warning(str(exc))
        except engine.InferenceError as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------------
# Automatic Speech Recognition
# ----------------------------------------------------------------------------
elif task == "Automatic Speech Recognition":
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "flac", "m4a"])

    if uploaded_file is not None:
        try:
            audio_bytes = validate_audio_file(uploaded_file.getvalue(), SETTINGS.limits.max_audio_mb)
            st.audio(audio_bytes)

            if st.button("Transcribe", type="primary"):
                with st.spinner("Transcribing…"):
                    result = engine.run_asr(audio_bytes)
                st.markdown("**Transcript**")
                st.success(result.output["text"])
                st.caption(f"Model: `{result.model_name}` · {result.latency_ms:.0f} ms")
                _download_button(result.output["text"], "transcript.txt", "⬇️ Download transcript")
                _log_history("ASR", result.output["text"][:40], result.latency_ms)
        except ValidationError as exc:
            st.warning(str(exc))
        except engine.InferenceError as exc:
            st.error(str(exc))

st.divider()
st.caption(
    "Built with 🤗 Transformers + Streamlit · Models load lazily and are cached per session · "
    f"Limits: text ≤ {SETTINGS.limits.max_text_chars} chars, "
    f"image ≤ {SETTINGS.limits.max_image_mb} MB, audio ≤ {SETTINGS.limits.max_audio_mb} MB."
)