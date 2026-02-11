"""FastAPI server for Qwen3-TTS text-to-speech generation.

Env:
  USE_ZMQ=1              - Use ZMQ (async engine loop + async queue).
  QWEN3_TTS_MODEL_PATH   - Model directory or HuggingFace model ID (e.g., Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).
  HOST, PORT             - Server bind address.
"""

import asyncio
from collections import deque
import logging
import os
import time
from contextlib import asynccontextmanager, suppress
from typing import Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Output format: 16-bit PCM at 24 kHz
TARGET_SAMPLE_RATE = 24000

logger = logging.getLogger(__name__)

# Ensure log messages appear on console (works when run as uvicorn server:app or python server.py)
if not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_handler)
    logging.getLogger().setLevel(logging.DEBUG if os.environ.get("DEBUG_TTS") else logging.INFO)

# Lazy imports to avoid loading heavy models at module load
_interface = None
_tokenizer = None
_zmq_bridge = None


def _use_zmq():
    """True if server should use ZMQ (background engine loop + queue-based generate)."""
    return os.environ.get("USE_ZMQ", "1").lower() in ("1", "true", "yes")


def get_interface():
    """Get or initialize the Qwen3TTSInterface (with or without ZMQ based on USE_ZMQ env)."""
    global _interface, _zmq_bridge
    if _interface is None:
        from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
        model_path = os.environ.get("QWEN3_TTS_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        
        # Check if it's a local path or HuggingFace model ID
        import os as os_module
        if os_module.isdir(model_path) or os_module.isfile(model_path):
            # Local path - use regular init
            if _use_zmq():
                from nano_qwen3tts_vllm.zmq import ZMQOutputBridge
                import warnings
                # Auto-find port if default is in use
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _zmq_bridge = ZMQOutputBridge(auto_find_port=True)
                    if w:
                        for warning in w:
                            logger.warning(str(warning.message))
                _interface = Qwen3TTSInterface(
                    model_path=model_path,
                    zmq_bridge=_zmq_bridge,
                    enforce_eager=False,
                )
            else:
                _interface = Qwen3TTSInterface(model_path=model_path)
        else:
            # HuggingFace model ID - use from_pretrained
            if _use_zmq():
                from nano_qwen3tts_vllm.zmq import ZMQOutputBridge
                import warnings
                # Auto-find port if default is in use
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _zmq_bridge = ZMQOutputBridge(auto_find_port=True)
                    if w:
                        for warning in w:
                            logger.warning(str(warning.message))
                _interface = Qwen3TTSInterface.from_pretrained(
                    pretrained_model_name_or_path=model_path,
                    zmq_bridge=_zmq_bridge,
                    enforce_eager=False,
                )
            else:
                _interface = Qwen3TTSInterface.from_pretrained(
                    pretrained_model_name_or_path=model_path,
                    enforce_eager=False,
                )
    return _interface


def get_tokenizer():
    """Get or initialize the Qwen3TTSTokenizer for decoding audio codes."""
    global _tokenizer
    if _tokenizer is None:
        from nano_qwen3tts_vllm.utils.speech_tokenizer_cudagraph import SpeechTokenizerCUDAGraph

        _tokenizer = SpeechTokenizerCUDAGraph(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            device="cuda:0",
            streaming_window_size=80,
        )
        
    return _tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up model and start ZMQ tasks when USE_ZMQ. Shutdown: stop ZMQ tasks and close bridge."""
    interface = get_interface()
    tokenizer = get_tokenizer()
    dummy_codes = torch.zeros((80, 16), dtype=torch.long)
    tokenizer.decode_streaming(dummy_codes, pad_to_size=80)
    if _use_zmq() and interface.zmq_bridge is not None:
        await interface.start_zmq_tasks()

    yield
    if _use_zmq() and _interface is not None and _interface.zmq_bridge is not None:
        await _interface.stop_zmq_tasks()
        if _zmq_bridge is not None:
            _zmq_bridge.close()


app = FastAPI(
    title="Qwen3-TTS API",
    description="Text-to-speech generation using Qwen3-TTS with vLLM-style optimizations",
    version="0.1.0",
    lifespan=lifespan,
)


class SpeechRequest(BaseModel):
    """Request body for speech generation."""

    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field(default="English", description="Language of the text")
    speaker: str = Field(default="Vivian", description="Speaker name")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def _float_to_pcm16(wav: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16 PCM."""
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767).astype(np.int16)


def _resample_to_24k(wav: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample waveform to 24 kHz if needed."""
    if orig_sr == TARGET_SAMPLE_RATE:
        return wav
    n_orig = len(wav)
    n_new = int(round(n_orig * TARGET_SAMPLE_RATE / orig_sr))
    if n_new == 0:
        return wav
    indices = np.linspace(0, n_orig - 1, n_new, dtype=np.float64)
    return np.interp(indices, np.arange(n_orig), wav).astype(np.float32)


def _crossfade(prev_tail: np.ndarray, new_head: np.ndarray) -> np.ndarray:
    """Linear crossfade between end of previous chunk and start of new chunk."""
    n = min(len(prev_tail), len(new_head))
    if n <= 0:
        return new_head
    w = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return prev_tail[:n] * (1.0 - w) + new_head[:n] * w


def _add_ref_code_context(
    window_codes: torch.Tensor,
    ref_code_context: Optional[torch.Tensor],
    ref_code_frames: int,
    decode_window_frames: int,
) -> tuple[torch.Tensor, int]:
    """Prepend ref_code as decoder context prefix when window is smaller than decode_window_frames."""
    if ref_code_context is None or window_codes.shape[0] >= decode_window_frames:
        return window_codes, 0

    available_space = decode_window_frames - window_codes.shape[0]
    ref_prefix_frames = min(available_space, ref_code_frames)
    if ref_prefix_frames <= 0:
        return window_codes, 0

    if not torch.is_tensor(ref_code_context):
        ref_code_context = torch.as_tensor(ref_code_context, dtype=window_codes.dtype)
    ref_prefix = ref_code_context[-ref_prefix_frames:].to(window_codes.dtype)
    return torch.cat([ref_prefix, window_codes], dim=0), ref_prefix_frames


async def generate_speech_stream(
    request: SpeechRequest,
    emit_every_frames: int = 4,
    decode_window_frames: int = 80,
    overlap_samples: int = 0,
    ref_code_context=None,
    ref_code_frames: int = 0,
):
    """Streaming decode with sliding window and optional crossfade."""
    interface = get_interface()
    tokenizer = get_tokenizer()
    loop = asyncio.get_event_loop()
    samples_per_frame = tokenizer.get_decode_upsample_rate()
    request_start_ts = time.perf_counter()
    first_chunk_emitted = False

    codes_queue: asyncio.Queue[tuple | None] = asyncio.Queue(maxsize=2)

    async def producer() -> None:
        window_codes = deque(maxlen=decode_window_frames)
        total_frames = 0
        last_emitted_frame_index = 0
        frames_since_emit = 0
        last_chunk_time = None
        try:
            async for audio_code in interface.generate_custom_voice_async(
                text=request.text,
                language=request.language,
                speaker=request.speaker,
            ):
                current_time = time.perf_counter()
                if last_chunk_time is not None:
                    inner_latency = current_time - last_chunk_time
                    logger.debug("[producer] inner chunk latency: %.2fms", inner_latency * 1000.0)
                last_chunk_time = current_time

                frame = torch.as_tensor(audio_code, dtype=torch.long).cpu()
                window_codes.append(frame)
                total_frames += 1
                frames_since_emit += 1

                if frames_since_emit >= emit_every_frames:
                    n_new_frames = total_frames - last_emitted_frame_index
                    await codes_queue.put(("chunk", list(window_codes), n_new_frames, 0))
                    last_emitted_frame_index = total_frames
                    frames_since_emit = 0

            unemitted_frames = total_frames - last_emitted_frame_index
            if unemitted_frames > 0:
                window_snapshot = list(window_codes)
                skip_frames = len(window_snapshot) - unemitted_frames
                await codes_queue.put(("flush", window_snapshot, unemitted_frames, skip_frames))
        finally:
            try:
                await codes_queue.put(None)
            except asyncio.CancelledError:
                with suppress(asyncio.QueueFull):
                    codes_queue.put_nowait(None)
                raise

    producer_task = asyncio.create_task(producer())
    decoded_tail = None

    try:
        while True:
            item = await codes_queue.get()
            if item is None:
                break

            msg_type, window_codes_list, n_new_frames, skip_frames = item
            if not window_codes_list or n_new_frames <= 0:
                continue
            window_tensor = torch.stack(window_codes_list, dim=0)  # [T, 16]
            window_tensor, ref_prefix_used = _add_ref_code_context(
                window_tensor, ref_code_context, ref_code_frames, decode_window_frames
            )

            wav, sr = await loop.run_in_executor(
                None,
                lambda w=window_tensor: tokenizer.decode_streaming(
                    w, pad_to_size=decode_window_frames
                ),
            )

            if msg_type == "chunk":
                new_samples = int(n_new_frames * samples_per_frame)
                chunk = wav[-new_samples:] if new_samples > 0 else np.empty(0, dtype=np.float32)
            else:
                skip_samples = int((skip_frames + ref_prefix_used) * samples_per_frame)
                chunk = wav[skip_samples:] if skip_samples > 0 else wav

            if decoded_tail is not None and overlap_samples > 0 and len(chunk) > 0:
                ov = min(overlap_samples, len(decoded_tail), len(chunk))
                if ov > 0:
                    head = _crossfade(decoded_tail[-ov:], chunk[:ov])
                    chunk = np.concatenate([head, chunk[ov:]], axis=0)

            if overlap_samples > 0 and len(chunk) > 0:
                decoded_tail = chunk[-overlap_samples:].copy()
            else:
                decoded_tail = None

            chunk_24k = _resample_to_24k(chunk, sr)
            pcm16 = _float_to_pcm16(chunk_24k)
            if len(pcm16) > 0:
                if not first_chunk_emitted:
                    ttfb_ms = (time.perf_counter() - request_start_ts) * 1000.0
                    logger.debug("[consumer] first chunk latency (TTFB): %.2fms", ttfb_ms)
                    first_chunk_emitted = True
                yield pcm16.tobytes()
    except asyncio.CancelledError:
        producer_task.cancel()
        with suppress(asyncio.CancelledError):
            await producer_task
        raise
    finally:
        if not producer_task.done():
            producer_task.cancel()
        with suppress(asyncio.CancelledError):
            await producer_task


@app.post("/v1/audio/speech", response_class=StreamingResponse)
async def generate_speech(request: SpeechRequest):
    """
    Generate speech from text.
    Returns raw PCM 16-bit mono at 24 kHz (audio/L16).
    Uses generate_custom_voice_async (requires USE_ZMQ=1).
    """
    try:
        return StreamingResponse(
            generate_speech_stream(request),
            media_type="audio/L16",
            headers={"Sample-Rate": str(TARGET_SAMPLE_RATE)},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Qwen3-TTS API",
        "docs": "/docs",
        "health": "/health",
        "speech": "POST /v1/audio/speech (PCM16, 24 kHz mono)",
        "zmq": _use_zmq(),
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    if _use_zmq():
        logger.info("Starting Qwen3-TTS API with ZMQ (async engine loop).")
    uvicorn.run(app, host=host, port=port)
