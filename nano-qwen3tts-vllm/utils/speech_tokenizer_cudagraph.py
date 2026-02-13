"""Speech tokenizer with CUDA graph capture for fast decode (e.g. 64 graphs B=1..16 x T=1..4)."""
import os
import sys
import torch
import numpy as np
from typing import Union, Tuple, List, Generator, Any
import logging
import time

from .streaming_decoder import (
    StreamingDecoderState,
    CUDAGraphStreamingState,
    init_streaming_state,
    init_cudagraph_streaming_state,
    decode_incremental as _decode_incremental,
    warmup_compiled_decoder,
    capture_streaming_cudagraphs,
    decode_cudagraph as _decode_cudagraph,
)

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


def _capture_decoder_cudagraphs(
    decoder,
    device: str,
    graph_lengths: List[int],
    batch_sizes: List[int] = None,
):
    """Capture CUDA graphs for decoder. Patches decoder.forward in-place.

    If batch_sizes is None: legacy mode, one graph per T with batch=1 only (key = T).
    If batch_sizes is set: key = (B, T) for B in batch_sizes, T in graph_lengths (e.g. 16*4=64 graphs).
    """
    decoder.eval()
    if batch_sizes is None:
        batch_sizes = [1]
    # Warmup
    with torch.inference_mode():
        _ = decoder(torch.randint(0, 100, (max(batch_sizes), 16, max(graph_lengths)), device=device))
        torch.cuda.synchronize()

    decoder.graphs = {}
    decoder.graph_inputs = {}
    decoder.graph_outputs = {}
    decoder._graph_key_is_tuple = len(batch_sizes) > 1 or max(batch_sizes) > 1
    graph_pool = None

    # Capture in reverse order (larger shapes first) for pool reuse
    with torch.inference_mode():
        for B in reversed(batch_sizes):
            for T in reversed(graph_lengths):
                graph = torch.cuda.CUDAGraph()
                input_buf = torch.randint(0, 100, (B, 16, T), device=device, dtype=torch.long)
                _ = decoder(input_buf)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph, graph_pool):
                    output_buf = decoder(input_buf)
                if graph_pool is None:
                    graph_pool = graph.pool()
                key = (B, T) if decoder._graph_key_is_tuple else T
                decoder.graphs[key] = graph
                decoder.graph_inputs[key] = input_buf
                decoder.graph_outputs[key] = output_buf
                torch.cuda.synchronize()

    decoder.original_forward = decoder.forward
    def forward_with_graph_replay(codes):
        import time
        B, T = codes.shape[0], codes.shape[2]
        if decoder._graph_key_is_tuple:
            key = (B, T) if (B, T) in decoder.graphs else None
        else:
            key = T if T in decoder.graphs and B == 1 else None
        if key is not None:
            start_time = time.time()
            decoder.graph_inputs[key].copy_(codes)
            decoder.graphs[key].replay()
            # Note: no torch.cuda.synchronize() here, so this is launch-only. The real wait
            # happens later at .cpu().numpy() in decode_window, which syncs + transfers GPU→CPU.
            # So "decoder latency" in the server includes executor + decode_window (+ .cpu()) + resample.
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > 1.0:  # only log when non-trivial
                logger.info(f"graph replay (launch) key={key} {elapsed_ms:.2f}ms")
            return decoder.graph_outputs[key]

        logger.info(f"using original forward for key: {key}")
        return decoder.original_forward(codes)

    decoder.forward = forward_with_graph_replay


class SpeechTokenizerCUDAGraph:
    """Qwen3-TTS speech tokenizer with CUDA graph capture for fast decode.

    Can capture either:
    - Legacy: one graph per seq length T=1..num_graph_lengths, batch=1 only.
    - Multi-batch (streaming window): graphs for B=1..graph_batch_sizes and T=1..graph_seq_lengths
      (e.g. 16 x 4 = 64 graphs) so decode_window with small context always hits a graph.
    """

    def __init__(
        self,
        model_path: str,
        device: str = None,
        dtype: torch.dtype = torch.bfloat16,
        num_graph_lengths: int = 0,
        graph_batch_sizes: int = 0,
        graph_seq_lengths: int = 0,
    ):
        """Load tokenizer and capture CUDA graphs for decoder.

        Args:
            model_path: Path to model dir or HuggingFace id.
            device: Device for model (default: cuda:0 if available).
            dtype: Model dtype (default: bfloat16).
            num_graph_lengths: Legacy: capture T=1..num_graph_lengths with batch=1 only (0 = skip).
            graph_batch_sizes: If >0, capture (B, T) for B=1..graph_batch_sizes, T=1..graph_seq_lengths (64 graphs for 16x4).
            graph_seq_lengths: Max seq length for multi-batch graphs (e.g. 4 for streaming window).
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

        # Phase 2: CUDA graph state for streaming decode (lazy init)
        self._streaming_graphs = None
        self._streaming_graph_state = None
        self._streaming_compiled_ready = False

        if device.startswith("cuda"):
            if graph_batch_sizes > 0 and graph_seq_lengths > 0:
                batch_sizes = list(range(1, graph_batch_sizes + 1))
                seq_lengths = list(range(1, graph_seq_lengths + 1))
                n = len(batch_sizes) * len(seq_lengths)
                print(f"Capturing {n} CUDA graphs for decoder (B=1..{graph_batch_sizes} x T=1..{graph_seq_lengths})...")
                _capture_decoder_cudagraphs(
                    self.tokenizer.model.decoder,
                    device,
                    seq_lengths,
                    batch_sizes=batch_sizes,
                )
                print("CUDA graph capture done.")
            elif num_graph_lengths > 0:
                graph_lengths = list(range(1, num_graph_lengths + 1))
                print(f"Capturing {len(graph_lengths)} CUDA graphs for decoder (T=1..{num_graph_lengths}, B=1)...")
                _capture_decoder_cudagraphs(self.tokenizer.model.decoder, device, graph_lengths)
                print("CUDA graph capture done.")
            else:
                print("Skipping CUDA graph capture (no graph_batch_sizes/graph_seq_lengths or num_graph_lengths).")
        else:
            print("Skipping CUDA graph capture (CPU).")

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
    def decode_window(
        self,
        inputs: List[dict],
        left_context_frames: int = 0,
    ) -> Tuple[List, int]:
        """Decode audio_codes and return only the audio after the first left_context_frames.

        Used for streaming: decode a window [context + new], trim to get only the "new" part.

        Args:
            inputs: List of dicts with key 'audio_codes' (tensor [time, 16] or list).
            left_context_frames: Number of leading code frames to drop from decoded audio (0 = return full).

        Returns:
            (wavs, sample_rate). wavs[0] is float32 numpy 1D.
        """
        audio_codes = inputs[0]["audio_codes"]
        if isinstance(audio_codes, list):
            codes = torch.tensor(audio_codes, dtype=torch.long, device=self.device)
        else:
            codes = audio_codes.to(self.device)
        if codes.dim() == 2:
            codes = codes.transpose(0, 1).unsqueeze(0)  # [1, 16, T]

        decoder_model = self.tokenizer.model.decoder
        samples_per_frame = int(decoder_model.total_upsample)
        sr = int(self.tokenizer.model.get_output_sample_rate())

        t0 = time.perf_counter()
        wav = decoder_model(codes)  # [1, 1, samples]
        torch.cuda.synchronize()
        gpu_sync_ms = (time.perf_counter() - t0) * 1000

        keep_start = left_context_frames * samples_per_frame
        wav_new = wav[..., keep_start:].squeeze(0).squeeze(0).to(torch.float32).cpu().numpy()
        cpu_transfer_ms = (time.perf_counter() - t0) * 1000 - gpu_sync_ms
        total_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"decode_window: gpu+sync={gpu_sync_ms:.2f}ms cpu_transfer={cpu_transfer_ms:.2f}ms total={total_ms:.2f}ms "
            f"(executor_ms - total ≈ thread-pool wait)"
        )
        return [wav_new], sr

    @torch.inference_mode()
    def streaming_decode(
        self,
        inputs: List[dict],
        chunk_size: int,
        context_size: int,
    ) -> Generator[Tuple[Any, int], None, None]:
        """Decode audio_codes in streaming chunks with left context for continuity.

        First chunk: decode codes[0:chunk_size] → output chunk_size frames of audio.
        Later chunks: decode codes[start-context_size:start+chunk_size], then trim to
        keep only the last chunk_size frames of audio (the "new" part).

        Args:
            inputs: List of dicts with key 'audio_codes' (tensor [time, 16] or list of length-T).
            chunk_size: Max number of code frames to decode per chunk (and to output per yield).
            context_size: Number of previous code frames to prepend when decoding (except first chunk).

        Yields:
            (wav_chunk, sample_rate) per chunk. wav_chunk is float32 numpy 1D or tensor.
        """
        audio_codes = inputs[0]["audio_codes"]
        if isinstance(audio_codes, list):
            codes = torch.tensor(audio_codes, dtype=torch.long, device=self.device)
        else:
            codes = audio_codes.to(self.device)
        if codes.dim() == 2:
            codes = codes.transpose(0, 1).unsqueeze(0)  # [1, 16, T]

        decoder_model = self.tokenizer.model.decoder
        samples_per_frame = int(decoder_model.total_upsample)
        sr = int(self.tokenizer.model.get_output_sample_rate())
        T = codes.shape[2]

        start = 0
        while start < T:
            end = min(start + chunk_size, T)
            if start == 0:
                context_frames = 0
                window = codes[:, :, 0:end]
            else:
                left = max(0, start - context_size)
                context_frames = start - left
                window = codes[:, :, left:end]

            wav_chunk = decoder_model(window)
            keep_start = context_frames * samples_per_frame
            wav_new = wav_chunk[..., keep_start:].squeeze(0).squeeze(0)
            wav_np = wav_new.to(torch.float32).cpu().numpy()
            yield wav_np, sr
            start = end

    def init_streaming(self, mode: int = 1) -> StreamingDecoderState | CUDAGraphStreamingState:
        """Initialize streaming state for a new synthesis request.

        Args:
            mode: 1 = cached streaming (with optional torch.compile),
                  2 = CUDA graph streaming (manual capture).

        Returns:
            Fresh state object. Pass to decode_incremental() or decode_incremental_cudagraph().
        """
        decoder = self.tokenizer.model.decoder

        if mode == 2:
            return init_cudagraph_streaming_state(
                decoder, device=self.device, dtype=self.dtype
            )

        return init_streaming_state(decoder, device=self.device, dtype=self.dtype)

    def warmup_streaming(self, compile: bool = True, cudagraph: bool = False):
        """Warmup streaming decode paths.

        Args:
            compile: Warmup torch.compile path (mode 1).
            cudagraph: Warmup and capture CUDA graphs for streaming (mode 2).
        """
        decoder = self.tokenizer.model.decoder

        if compile:
            warmup_state = init_streaming_state(decoder, device=self.device, dtype=self.dtype)
            warmup_compiled_decoder(decoder, warmup_state, device=self.device)
            self._streaming_compiled_ready = True
            logger.info("[streaming] torch.compile warmup complete")

        if cudagraph:
            graph_state = init_cudagraph_streaming_state(
                decoder, device=self.device, dtype=self.dtype
            )
            self._streaming_graphs = capture_streaming_cudagraphs(
                decoder, graph_state, device=self.device, chunk_sizes=[1, 2, 3, 4]
            )
            # Reset state after capture for clean inference
            graph_state.kv_cache.reset()
            graph_state.position_offset.zero_()
            for c in graph_state.conv_caches:
                if c.numel() > 0:
                    c.zero_()
            for c in graph_state.transconv_caches:
                if c.numel() > 0:
                    c.zero_()
            self._streaming_graph_state = graph_state
            logger.info("[streaming] CUDA graph warmup + capture complete")

    @torch.inference_mode()
    def decode_incremental(
        self,
        new_codes: torch.Tensor,
        state: StreamingDecoderState,
        compiled: bool = True,
    ) -> Tuple[np.ndarray, int, StreamingDecoderState]:
        """Decode only new frames using cached streaming state.

        ~20x less compute than decode_window() for typical streaming
        (4 new frames vs 80-frame window re-decode).

        Args:
            new_codes: [B, 16, N] new codec frames.
            state: StreamingDecoderState from init_streaming() or previous call.
            compiled: If True and torch.compile is warmed up, use compiled path.

        Returns:
            (wav_new, sample_rate, state).
            wav_new is a float32 numpy 1D array of new audio samples.
        """
        decoder = self.tokenizer.model.decoder
        sr = int(self.tokenizer.model.get_output_sample_rate())
        use_compiled = compiled and self._streaming_compiled_ready

        t0 = time.perf_counter()
        wav = _decode_incremental(decoder, new_codes, state, compiled=use_compiled)
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1000

        wav_np = wav.squeeze(0).squeeze(0).to(torch.float32).cpu().numpy()
        total_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"decode_incremental: gpu+sync={gpu_ms:.2f}ms total={total_ms:.2f}ms "
            f"frames={new_codes.shape[2]} samples={len(wav_np)} compiled={use_compiled}"
        )
        return wav_np, sr, state

    @torch.inference_mode()
    def decode_incremental_cudagraph(
        self,
        new_codes: torch.Tensor,
    ) -> Tuple[np.ndarray, int]:
        """Decode new frames using captured CUDA graphs (mode 2).

        The CUDA graph state is maintained internally. Call warmup_streaming(cudagraph=True)
        before first use. State resets are managed per-session externally.

        Args:
            new_codes: [B, 16, N] new codec frames.

        Returns:
            (wav_new, sample_rate).
        """
        if self._streaming_graphs is None:
            raise RuntimeError(
                "CUDA graph streaming not initialized. Call warmup_streaming(cudagraph=True) first."
            )

        sr = int(self.tokenizer.model.get_output_sample_rate())

        t0 = time.perf_counter()
        wav = _decode_cudagraph(new_codes, self._streaming_graphs, self._streaming_graph_state)
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1000

        wav_np = wav.squeeze(0).squeeze(0).to(torch.float32).cpu().numpy()
        total_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"decode_incremental_cudagraph: gpu+sync={gpu_ms:.2f}ms total={total_ms:.2f}ms "
            f"frames={new_codes.shape[2]} samples={len(wav_np)}"
        )
        return wav_np, sr

    def reset_cudagraph_state(self):
        """Reset CUDA graph streaming state for a new utterance."""
        if self._streaming_graph_state is not None:
            self._streaming_graph_state.kv_cache.reset()
            self._streaming_graph_state.position_offset.zero_()
            for c in self._streaming_graph_state.conv_caches:
                if c.numel() > 0:
                    c.zero_()
            for c in self._streaming_graph_state.transconv_caches:
                if c.numel() > 0:
                    c.zero_()

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
