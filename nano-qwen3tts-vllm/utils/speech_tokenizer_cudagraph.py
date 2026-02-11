"""Speech tokenizer with CUDA graph capture for fast decode."""
import logging
import os
import sys
import threading
import numpy as np
import torch
from typing import Union, Tuple, List, Optional


logger = logging.getLogger(__name__)

try:
    qwen_tts_path = os.path.expanduser(os.environ.get("QWEN_TTS_PATH", "/home/sang/work/Qwen3-TTS"))
    if os.path.exists(qwen_tts_path) and qwen_tts_path not in sys.path:
        sys.path.insert(0, qwen_tts_path)
    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer as _Qwen3TTSTokenizer
    HAS_SPEECH_TOKENIZER = True
except ImportError as e:
    HAS_SPEECH_TOKENIZER = False
    _Qwen3TTSTokenizer = None


def _capture_decoder_cudagraphs(decoder, device: str, graph_lengths: List[int]):
    """Capture CUDA graphs for requested decoder lengths and patch decoder.forward."""
    graph_lengths = sorted({int(length) for length in graph_lengths if int(length) > 0})
    if not graph_lengths:
        return

    decoder.eval()
    # Warmup
    with torch.inference_mode():
        _ = decoder(torch.randint(0, 100, (1, 16, 100), device=device))
        torch.cuda.synchronize()

    decoder.graphs = {}
    decoder.graph_inputs = {}
    decoder.graph_outputs = {}
    graph_pool = None
    replay_lock = threading.Lock()

    with torch.inference_mode():
        for T in reversed(graph_lengths):
            graph = torch.cuda.CUDAGraph()
            input_buf = torch.randint(0, 100, (1, 16, T), device=device, dtype=torch.long)
            _ = decoder(input_buf)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, graph_pool):
                output_buf = decoder(input_buf)
            if graph_pool is None:
                graph_pool = graph.pool()
            decoder.graphs[T] = graph
            decoder.graph_inputs[T] = input_buf
            decoder.graph_outputs[T] = output_buf
            torch.cuda.synchronize()

    decoder.original_forward = decoder.forward

    def forward_with_graph_replay(codes):
        T = codes.shape[2]
        if T in decoder.graphs:
            # Guard shared graph buffers so all decode entrypoints can run safely.
            with replay_lock:
                decoder.graph_inputs[T].copy_(codes)
                decoder.graphs[T].replay()
                return decoder.graph_outputs[T].clone()
        return decoder.original_forward(codes)

    decoder.forward = forward_with_graph_replay


class SpeechTokenizerCUDAGraph:
    """Qwen3-TTS speech tokenizer with 50 captured CUDA graphs for fast decode.

    Loads the 12Hz tokenizer, captures one graph per decode length T=1..50,
    and patches the decoder to replay graphs when shape matches (like predictor_model_runner).
    """

    def __init__(
        self,
        model_path: str,
        device: str = None,
        dtype: torch.dtype = torch.bfloat16,
        num_graph_lengths: int = 50,
        streaming_window_size: int = 80,
    ):
        """Load tokenizer and capture CUDA graphs for decoder.

        Args:
            model_path: Path to model dir; speech tokenizer loaded from {model_path}/speech_tokenizer.
            device: Device for model (default: cuda:0 if available).
            dtype: Model dtype (default: bfloat16).
            num_graph_lengths: Number of graphs to capture for lengths 1..num_graph_lengths (default: 50).
            streaming_window_size: Additional streaming window length to capture (default: 80).
        """
        if not HAS_SPEECH_TOKENIZER:
            raise ImportError(
                "qwen_tts not found. Install Qwen3-TTS and set QWEN_TTS_PATH or add to path."
            )
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        speech_tokenizer_path = model_path

        print(f"Loading speech tokenizer (CUDAGraph) from {speech_tokenizer_path}...")
        self.tokenizer = _Qwen3TTSTokenizer.from_pretrained(
            speech_tokenizer_path,
            device_map=device,
        )
        self.tokenizer.model = self.tokenizer.model.to(dtype)

        if hasattr(self.tokenizer.config, "sample_rate"):
            self.sample_rate = self.tokenizer.config.sample_rate
        elif hasattr(getattr(self.tokenizer, "feature_extractor", None), "sampling_rate"):
            self.sample_rate = self.tokenizer.feature_extractor.sampling_rate
        else:
            self.sample_rate = 12500

        self.device = device
        self.dtype = dtype

        if device.startswith("cuda") and (num_graph_lengths > 0 or streaming_window_size > 0):
            graph_lengths = list(range(1, num_graph_lengths + 1))
            if streaming_window_size > num_graph_lengths:
                graph_lengths.append(streaming_window_size)
            graph_lengths = sorted(set(graph_lengths))
            print(f"Capturing {len(graph_lengths)} CUDA graphs for decoder (T={graph_lengths[0]}..{graph_lengths[-1]})...")
            _capture_decoder_cudagraphs(self.tokenizer.model.decoder, device, graph_lengths)
            print("CUDA graph capture done.")
        else:
            print("Skipping CUDA graph capture (CPU or no graph lengths requested).")

        print(f"Speech tokenizer (CUDAGraph) loaded: sample_rate={self.sample_rate}Hz, device={self.device}")

    @torch.inference_mode()
    def decode(self, inputs: List[dict]) -> Tuple[List, int]:
        """Decode audio_codes to waveform. Same API as Qwen3TTSTokenizer.decode.

        Args:
            inputs: List of dicts with key 'audio_codes' (tensor [time, 16] or list).

        Returns:
            (wavs, sample_rate).
        """
        return self.tokenizer.decode(inputs)

    @torch.inference_mode()
    def chunked_decode(
        self, inputs: List[dict], chunk_size: int = 300, left_context_size: int = 25,
    ) -> Tuple[List, int]:
        """Decode audio_codes using overlap-add chunking to avoid boundary artifacts.

        Args:
            inputs: List of dicts with key 'audio_codes' (tensor [time, 16] or list).
            chunk_size: Number of codec frames per chunk (default: 300).
            left_context_size: Overlap frames for crossfade between chunks (default: 25).

        Returns:
            (wavs, sample_rate).
        """
        audio_codes = inputs[0]["audio_codes"]

        # Convert to [1, 16, time] tensor
        if isinstance(audio_codes, list):
            codes = torch.tensor(audio_codes, dtype=torch.long, device=self.device)
        else:
            codes = audio_codes.to(self.device)
        if codes.dim() == 2:  # [time, 16] -> [1, 16, time]
            codes = codes.transpose(0, 1).unsqueeze(0)

        # Chunked decode via decoder (each chunk hits CUDAGraph-patched forward)
        decoder_model = self.tokenizer.model.decoder
        wav = decoder_model.chunked_decode(codes, chunk_size, left_context_size)

        wavs = [wav.squeeze().to(torch.float32).cpu().numpy()]
        sr = int(self.tokenizer.model.get_output_sample_rate())
        return wavs, sr

    @torch.inference_mode()
    def decode_codec_ids(self, codec_ids: torch.Tensor) -> Tuple[List, int]:
        """Decode codec IDs [batch, 16, time] to (audio_list, sample_rate). Drop-in for SpeechTokenizer.decode."""
        batch_size = codec_ids.shape[0]
        inputs = []
        for i in range(batch_size):
            codes = codec_ids[i]
            if codes.dim() == 2:
                codes = codes.transpose(0, 1)
            inputs.append({"audio_codes": codes})
        return self.decode(inputs)

    def get_decode_upsample_rate(self) -> int:
        """Waveform samples per codec frame."""
        model = self.tokenizer.model
        output_sample_rate = None
        frame_rate = None

        if hasattr(model, "get_output_sample_rate"):
            try:
                output_sample_rate = float(model.get_output_sample_rate())
            except Exception:
                output_sample_rate = None
        if hasattr(model, "config"):
            frame_rate = getattr(model.config, "frame_rate", None)

        if output_sample_rate and frame_rate:
            try:
                return int(round(output_sample_rate / float(frame_rate)))
            except Exception:
                pass

        logger.warning("Could not determine decode upsample rate from model config, falling back to 1000")
        return 1000

    @torch.inference_mode()
    def decode_streaming(
        self,
        audio_codes: torch.Tensor,
        pad_to_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Decode a single streaming window of codec frames.

        Args:
            audio_codes: [T, 16] codec frames.
            pad_to_size: Optional frame length for right padding before decode.

        Returns:
            (wav_float32_numpy, sample_rate)
        """
        if isinstance(audio_codes, list):
            codes = torch.tensor(audio_codes, dtype=torch.long)
        elif torch.is_tensor(audio_codes):
            codes = audio_codes.to(dtype=torch.long)
        else:
            codes = torch.as_tensor(audio_codes, dtype=torch.long)

        if codes.dim() != 2:
            raise ValueError(f"decode_streaming expects [T, 16], got shape={tuple(codes.shape)}")

        actual_t = int(codes.shape[0])
        codes = codes.to(self.device)
        codes = codes.transpose(0, 1).unsqueeze(0)  # [T, 16] -> [1, 16, T]

        if pad_to_size is not None and pad_to_size > codes.shape[2]:
            codes = torch.nn.functional.pad(codes, (0, pad_to_size - codes.shape[2]), value=0)

        wav = self.tokenizer.model.decoder(codes)

        if pad_to_size is not None and pad_to_size > actual_t:
            samples_per_frame = self.get_decode_upsample_rate()
            wav = wav[:, :, : actual_t * samples_per_frame]

        wav_np = wav.squeeze().to(torch.float32).cpu().numpy()
        sr = int(self.tokenizer.model.get_output_sample_rate())
        return wav_np, sr
