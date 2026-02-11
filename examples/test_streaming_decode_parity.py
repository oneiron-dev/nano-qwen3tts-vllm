"""
Streaming decode parity check against full decode.

Usage:
  python examples/test_streaming_decode_parity.py \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
"""

import argparse
import os
import sys
from collections import deque

import numpy as np
import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def reconstruct_streamed_waveform(
    tokenizer,
    audio_codes,
    emit_every_frames: int = 4,
    decode_window_frames: int = 80,
):
    """Reconstruct waveform using streaming window decode logic with overlap disabled."""
    window = deque(maxlen=decode_window_frames)
    samples_per_frame = tokenizer.get_decode_upsample_rate()
    total_frames = 0
    last_emitted_frame_index = 0
    chunks = []
    stream_sr = None

    for code in audio_codes:
        frame = torch.as_tensor(code, dtype=torch.long).cpu()
        window.append(frame)
        total_frames += 1

        if total_frames - last_emitted_frame_index >= emit_every_frames:
            n_new_frames = total_frames - last_emitted_frame_index
            window_tensor = torch.stack(list(window), dim=0)
            wav, sr = tokenizer.decode_streaming(window_tensor, pad_to_size=decode_window_frames)
            stream_sr = sr if stream_sr is None else stream_sr
            new_samples = int(n_new_frames * samples_per_frame)
            chunk = wav[-new_samples:] if new_samples > 0 else np.empty(0, dtype=np.float32)
            chunks.append(chunk)
            last_emitted_frame_index = total_frames

    unemitted_frames = total_frames - last_emitted_frame_index
    if unemitted_frames > 0:
        window_snapshot = list(window)
        skip_frames = len(window_snapshot) - unemitted_frames
        window_tensor = torch.stack(window_snapshot, dim=0)
        wav, sr = tokenizer.decode_streaming(window_tensor, pad_to_size=decode_window_frames)
        stream_sr = sr if stream_sr is None else stream_sr
        skip_samples = int(skip_frames * samples_per_frame)
        chunk = wav[skip_samples:] if skip_samples > 0 else wav
        chunks.append(chunk)

    if not chunks:
        return np.empty(0, dtype=np.float32), stream_sr
    return np.concatenate(chunks).astype(np.float32), stream_sr


def main():
    parser = argparse.ArgumentParser(description="Streaming decode parity test")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("QWEN3_TTS_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        help="Model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Tokenizer model path or HuggingFace ID",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Streaming parity test sentence. This should produce enough frames to validate boundaries.",
        help="Text to synthesize",
    )
    parser.add_argument("--language", type=str, default="English", help="Language")
    parser.add_argument("--speaker", type=str, default="Vivian", help="Speaker name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Tokenizer device")
    parser.add_argument("--emit-every-frames", type=int, default=4, help="Emit cadence in codec frames")
    parser.add_argument("--decode-window-frames", type=int, default=80, help="Streaming decode window size")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for np.allclose")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
    from nano_qwen3tts_vllm.utils.speech_tokenizer_cudagraph import SpeechTokenizerCUDAGraph

    print("Loading interface...")
    if os.path.isdir(args.model_path) or os.path.isfile(args.model_path):
        interface = Qwen3TTSInterface(model_path=args.model_path, enforce_eager=False)
    else:
        interface = Qwen3TTSInterface.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            enforce_eager=False,
        )

    try:
        print("Generating codec frames...")
        audio_codes = list(
            interface.generate_custom_voice(
                text=args.text,
                language=args.language,
                speaker=args.speaker,
            )
        )
        if not audio_codes:
            raise RuntimeError("No codec frames were generated.")

        print("Loading tokenizer...")
        tokenizer = SpeechTokenizerCUDAGraph(
            args.tokenizer_path,
            device=args.device,
            streaming_window_size=args.decode_window_frames,
        )

        print("Decoding full utterance...")
        full_wavs, full_sr = tokenizer.decode([{"audio_codes": audio_codes}])
        full_wav = np.asarray(full_wavs[0], dtype=np.float32)

        print("Decoding via streaming windows...")
        streamed_wav, streamed_sr = reconstruct_streamed_waveform(
            tokenizer,
            audio_codes,
            emit_every_frames=args.emit_every_frames,
            decode_window_frames=args.decode_window_frames,
        )

        if streamed_sr is None:
            raise RuntimeError("Streaming decode produced no sample rate.")
        if full_sr != streamed_sr:
            raise AssertionError(f"Sample rate mismatch: full={full_sr}, streamed={streamed_sr}")
        if len(full_wav) != len(streamed_wav):
            raise AssertionError(f"Length mismatch: full={len(full_wav)}, streamed={len(streamed_wav)}")

        if not np.allclose(full_wav, streamed_wav, atol=args.atol):
            abs_diff = np.abs(full_wav - streamed_wav)
            max_diff = float(abs_diff.max())
            first_mismatch = int(np.argmax(abs_diff > args.atol))
            print(f"Parity failed: max_abs_diff={max_diff:.6f}, first_mismatch_idx={first_mismatch}")
            raise AssertionError("streaming decode waveform differs from full decode beyond tolerance")

        print(f"Parity passed: sr={full_sr}, samples={len(full_wav)}, atol={args.atol}")
    finally:
        interface.shutdown()


if __name__ == "__main__":
    main()
