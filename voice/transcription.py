"""
Audio transcription helpers for the Patient Voice Agent.

Supports:
- OpenAI Whisper API (recommended)
- Optional conversion from non-WAV formats if pydub+ffmpeg are available

Usage:
    from voice.transcription import transcribe_audio_bytes
    text = transcribe_audio_bytes(audio_bytes, input_format="webm")
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Optional

# OpenAI client (required for API transcription)
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Optional: format conversion (webm/ogg/mp3 -> wav)
try:
    from pydub import AudioSegment

    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False


def _convert_to_wav(audio_bytes: bytes, input_format: str = "wav") -> bytes:
    """
    Convert audio bytes to WAV. For non-WAV formats, requires pydub + ffmpeg.
    """
    fmt = (input_format or "wav").lower()
    if fmt == "wav":
        return audio_bytes

    if not HAS_PYDUB:
        raise ValueError(
            "Non-WAV input requires pydub + ffmpeg. "
            "Install with `pip install pydub` and ensure ffmpeg is available."
        )

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()


def transcribe_audio_bytes(
    audio_bytes: bytes,
    input_format: str = "wav",
    model: str = "whisper-1",
    language: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Transcribe audio bytes using OpenAI Whisper API.

    Args:
        audio_bytes: raw audio bytes.
        input_format: format of input bytes (e.g., "wav", "webm", "ogg", "mp3").
        model: Whisper model name (e.g., "whisper-1").
        language: optional language code hint (e.g., "en").
        api_key: optional explicit API key; falls back to OPENAI_API_KEY env var.

    Returns:
        Transcribed text string.
    """
    if not HAS_OPENAI:
        raise RuntimeError("OpenAI package not installed. Install with `pip install openai`.")

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Please provide an API key.")

    client = OpenAI(api_key=key)

    wav_bytes = _convert_to_wav(audio_bytes, input_format=input_format)

    # Write to temp file because OpenAI client expects a file-like object
    # Use delete=False on Windows to avoid permission issues
    tmp_file = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_file.write(wav_bytes)
        tmp_file.flush()
        tmp_file.close()  # Close before opening again
        
        # Now open the file for reading
        with open(tmp_file.name, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=model,
                file=f,
                language=language,
            )
        
        # OpenAI returns an object with `.text`
        result = resp.text if hasattr(resp, "text") else str(resp)
        return result
    finally:
        # Clean up temp file
        if tmp_file and os.path.exists(tmp_file.name):
            try:
                os.unlink(tmp_file.name)
            except:
                pass  # Ignore cleanup errors


__all__ = ["transcribe_audio_bytes"]


