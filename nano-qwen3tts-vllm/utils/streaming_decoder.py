"""Cached streaming decoder for Qwen3-TTS codec.

Eliminates the ~20x redundant compute in the windowed decode approach
by maintaining conv layer state and Transformer KV cache across calls.

Instead of re-decoding the entire 80-frame window each time 4 new frames
arrive, this processes only the new frames through the full decoder
pipeline with cached state.

Three acceleration modes:
  - Mode 0 (default): Windowed decode with CUDA graphs (existing path)
  - Mode 1: Cached streaming + torch.compile(mode="reduce-overhead")
  - Mode 2: Cached streaming + manual CUDA graph capture

Usage:
    from streaming_decoder import init_streaming_state, decode_incremental

    state = init_streaming_state(decoder, device="cuda")
    while True:
        new_codes = ...  # [1, 16, N] where N = emit_every_frames
        wav_new = decode_incremental(decoder, new_codes, state)
        # wav_new: [1, 1, N*1920] new audio samples
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

logger = logging.getLogger(__name__)


# ── State classes ─────────────────────────────────────────────────────


@dataclass
class StreamingDecoderState:
    """All streaming state for one decode session."""

    conv_caches: dict[str, torch.Tensor] = field(default_factory=dict)
    transconv_caches: dict[str, torch.Tensor] = field(default_factory=dict)
    kv_cache: DynamicCache | None = None
    position_offset: int = 0


class StaticStreamingKVCache:
    """Drop-in for DynamicCache with pre-allocated fixed-size buffers.

    Designed for CUDA graph compatibility:
    - update() uses index_copy_ with tensor indices (no .item())
    - Returns full max_seq_len buffer; attention mask handles valid positions
    - get_mask_sizes() returns constant (max_seq_len, 0) for fixed mask shapes

    The sliding_window=72 causal mask ensures attention only sees valid
    positions: future positions are masked by causality, and positions
    beyond the window are masked by sliding_window.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype,
    ):
        self.key_cache = [
            torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.value_cache = [
            torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self._max_len = max_seq_len
        self._num_layers = num_layers
        # Tensor-based position tracking for CUDA graph compat
        self._seq_len = torch.zeros(1, dtype=torch.long, device=device)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        B, H, N, D = key_states.shape
        cache_B, cache_H, cache_S, cache_D = self.key_cache[layer_idx].shape
        if H != cache_H or D != cache_D:
            raise ValueError(
                f"StaticStreamingKVCache shape mismatch at layer {layer_idx}: "
                f"cache=[{cache_B},{cache_H},{cache_S},{cache_D}] "
                f"key_states=[{B},{H},{N},{D}]"
            )
        # Tensor-only indexing for CUDA graph compatibility (no .item())
        indices = torch.arange(N, device=key_states.device) + self._seq_len.squeeze()
        self.key_cache[layer_idx].index_copy_(2, indices, key_states)
        self.value_cache[layer_idx].index_copy_(2, indices, value_states)
        if layer_idx == self._num_layers - 1:
            self._seq_len.add_(N)
        # Return full buffer — causal + sliding window mask handles valid positions
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self._seq_len.item()

    def get_mask_sizes(self, cache_position, layer_idx=0):
        """Return (kv_length, kv_offset) for attention mask creation.

        Returns constant max_seq_len so mask shape is fixed across calls —
        required for CUDA graph compatibility. The causal + sliding_window
        mask handles which positions are actually valid.
        """
        return self._max_len, 0

    def get_max_cache_shape(self):
        return [self._max_len] * self._num_layers

    def reset(self):
        """Reset cache for a new sequence without reallocating."""
        for i in range(self._num_layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()
        self._seq_len.zero_()


@dataclass
class CUDAGraphStreamingState:
    """Streaming state for manual CUDA graph path (mode 2).

    Uses integer-indexed lists instead of string-keyed dicts for
    zero-overhead cache lookup during graph replay.
    """
    conv_caches: list[torch.Tensor] = field(default_factory=list)
    transconv_caches: list[torch.Tensor] = field(default_factory=list)
    kv_cache: Optional[StaticStreamingKVCache] = None
    position_offset: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.long))
    # Map from string cache key to list index (built at init, used for debugging)
    conv_key_map: dict[str, int] = field(default_factory=dict)
    transconv_key_map: dict[str, int] = field(default_factory=dict)
    # Reference to decoder module for eager fallback when chunk size has no captured graph
    decoder: Optional[object] = None


# ── Conv1d streaming ─────────────────────────────────────────────────


def _conv_streaming(
    conv_module,
    x: torch.Tensor,
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process a CausalConvNet layer in streaming mode.

    Cache is always pre-allocated (zeros on first call produce identical
    output to zero-padding). No None checks needed.

    Args:
        conv_module: Qwen3TTSTokenizerV2CausalConvNet instance.
        x: [B, C_in, N] new input samples.
        cache: [B, C_in, padding] from previous call (pre-allocated zeros for first call).

    Returns:
        (output [B, C_out, N'], new_cache [B, C_in, padding]).
    """
    padding = conv_module.padding

    if padding == 0:
        # Pointwise conv (kernel=1) — no state needed
        return conv_module.conv(x).contiguous(), cache

    x_padded = torch.cat([cache, x], dim=-1)

    # Save last `padding` input samples for next call
    new_cache = x_padded[:, :, -padding:].clone()

    # Run the raw Conv1d (bypass CausalConvNet.forward which re-pads)
    output = conv_module.conv(x_padded).contiguous()

    return output, new_cache


# ── ConvTranspose1d streaming ────────────────────────────────────────


def _transconv_streaming(
    transconv_module,
    x: torch.Tensor,
    cache: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Process a CausalTransConvNet layer in streaming mode.

    For kernel_size == stride (upsample stages): no overlap, no caching needed.
    For kernel_size > stride (decoder blocks): cache last input sample and
    use overlap-add to produce exact output.

    Args:
        transconv_module: Qwen3TTSTokenizerV2CausalTransConvNet instance.
        x: [B, C_in, N] new input samples.
        cache: [B, C_in, 1] last input sample from previous call,
               or pre-allocated zeros for first call. None for no-overlap case.

    Returns:
        (output [B, C_out, N*stride], new_cache or None).
    """
    stride = transconv_module.conv.stride[0]
    kernel_size = transconv_module.conv.kernel_size[0]

    if kernel_size <= stride:
        # No overlap between consecutive inputs — just run normally
        raw = transconv_module.conv(x)
        left_pad = transconv_module.left_pad
        right_pad = transconv_module.right_pad
        if right_pad > 0:
            raw = raw[..., left_pad:-right_pad]
        elif left_pad > 0:
            raw = raw[..., left_pad:]
        return raw.contiguous(), None

    # kernel > stride: overlap exists, need caching
    N = x.shape[-1]

    x_with_cache = torch.cat([cache, x], dim=-1)  # [B, C, N+1]

    # Save last input sample for next call
    new_cache = x[:, :, -1:].clone()

    raw = transconv_module.conv(x_with_cache)

    # Take exactly N * stride new samples after the overlap region.
    output = raw[:, :, stride : stride + N * stride]

    return output.contiguous(), new_cache


# ── ConvNeXtBlock streaming ──────────────────────────────────────────


def _convnext_streaming(
    block,
    x: torch.Tensor,
    cache_key: str,
    state: StreamingDecoderState,
) -> torch.Tensor:
    """Process a ConvNeXtBlock in streaming mode.

    Only the depthwise conv (dwconv) has state; the rest is pointwise.
    """
    residual = x

    # dwconv: CausalConvNet with groups=dim, kernel=7
    x, new_cache = _conv_streaming(
        block.dwconv, x, state.conv_caches[cache_key]
    )
    state.conv_caches[cache_key] = new_cache

    # Pointwise path (stateless)
    x = x.permute(0, 2, 1)
    x = block.norm(x)
    x = block.pwconv1(x)
    x = block.act(x)
    x = block.pwconv2(x)
    x = block.gamma * x
    x = x.permute(0, 2, 1)

    return residual + x


# ── ResidualUnit streaming ───────────────────────────────────────────


def _residual_unit_streaming(
    unit,
    x: torch.Tensor,
    cache_prefix: str,
    state: StreamingDecoderState,
) -> torch.Tensor:
    """Process a DecoderResidualUnit in streaming mode.

    conv1 (dilated, kernel=7) needs caching; conv2 (kernel=1) does not.
    """
    residual = x

    x = unit.act1(x)
    x, c1 = _conv_streaming(
        unit.conv1, x, state.conv_caches[f"{cache_prefix}.c1"]
    )
    state.conv_caches[f"{cache_prefix}.c1"] = c1

    x = unit.act2(x)
    x, c2 = _conv_streaming(
        unit.conv2, x, state.conv_caches[f"{cache_prefix}.c2"]
    )
    state.conv_caches[f"{cache_prefix}.c2"] = c2

    return x + residual


# ── DecoderBlock streaming ───────────────────────────────────────────


def _decoder_block_streaming(
    block,
    x: torch.Tensor,
    cache_prefix: str,
    state: StreamingDecoderState,
) -> torch.Tensor:
    """Process a DecoderBlock in streaming mode.

    block.block[0] = SnakeBeta (stateless)
    block.block[1] = CausalTransConvNet (cached — kernel = 2*stride)
    block.block[2..4] = ResidualUnit (cached — dilated convs)
    """
    # SnakeBeta — pointwise, stateless
    x = block.block[0](x)

    # CausalTransConvNet — upsampling with overlap
    tc_key = f"{cache_prefix}.tc"
    x, tc = _transconv_streaming(
        block.block[1], x, state.transconv_caches[tc_key]
    )
    state.transconv_caches[tc_key] = tc

    # ResidualUnits (dilation 1, 3, 9)
    for j, ru in enumerate(block.block[2:]):
        x = _residual_unit_streaming(ru, x, f"{cache_prefix}.ru{j}", state)

    return x


# ── State initialization ─────────────────────────────────────────────


def init_streaming_state(
    decoder=None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> StreamingDecoderState:
    """Create a fresh streaming state with pre-allocated caches.

    When decoder is provided, walks the module tree and pre-allocates all
    conv/transconv caches as zero tensors. This eliminates `if cache is not None`
    branches which cause torch.compile guard failures.

    When decoder is None (legacy), returns empty state with lazy allocation.
    """
    if decoder is None:
        return StreamingDecoderState()

    B = 1  # Single-batch streaming
    state = StreamingDecoderState()

    # Pre-conv: CausalConvNet(codebook_dim → latent_dim, kernel=3, padding=2)
    if decoder.pre_conv.padding > 0:
        C_in = decoder.pre_conv.conv.in_channels
        state.conv_caches["pre_conv"] = torch.zeros(
            B, C_in, decoder.pre_conv.padding, device=device, dtype=dtype
        )

    # Upsample stages: TransConv(k=stride, no overlap) + ConvNeXtBlock(dwconv k=7)
    for i, blocks in enumerate(decoder.upsample):
        transconv = blocks[0]
        convnext = blocks[1]
        # TransConv: kernel == stride for upsample stages, no cache needed
        # ConvNeXtBlock dwconv: CausalConvNet with kernel=7, padding=6
        if convnext.dwconv.padding > 0:
            C_in = convnext.dwconv.conv.in_channels
            state.conv_caches[f"up.{i}.cnx"] = torch.zeros(
                B, C_in, convnext.dwconv.padding, device=device, dtype=dtype
            )

    # Decoder path
    # decoder[0]: CausalConvNet(latent_dim → decoder_dim, kernel=7, padding=6)
    entry_conv = decoder.decoder[0]
    if entry_conv.padding > 0:
        state.conv_caches["dec.0"] = torch.zeros(
            B, entry_conv.conv.in_channels, entry_conv.padding, device=device, dtype=dtype
        )

    # decoder[1..N-2]: DecoderBlocks
    num_decoder_blocks = len(decoder.decoder) - 3
    for i in range(num_decoder_blocks):
        block = decoder.decoder[1 + i]
        prefix = f"dec.{1 + i}"

        # TransConvNet: kernel = 2*stride, needs overlap cache
        tc = block.block[1]
        if tc.conv.kernel_size[0] > tc.conv.stride[0]:
            C_in = tc.conv.in_channels
            state.transconv_caches[f"{prefix}.tc"] = torch.zeros(
                B, C_in, 1, device=device, dtype=dtype
            )

        # ResidualUnits
        for j, ru in enumerate(block.block[2:]):
            # conv1: dilated, kernel=7
            if ru.conv1.padding > 0:
                state.conv_caches[f"{prefix}.ru{j}.c1"] = torch.zeros(
                    B, ru.conv1.conv.in_channels, ru.conv1.padding, device=device, dtype=dtype
                )
            # conv2: kernel=1, padding=0 — but pre-allocate anyway for consistency
            state.conv_caches[f"{prefix}.ru{j}.c2"] = torch.zeros(
                B, ru.conv2.conv.in_channels, max(ru.conv2.padding, 1), device=device, dtype=dtype
            ) if ru.conv2.padding > 0 else torch.zeros(0, device=device, dtype=dtype)

    # decoder[-1]: CausalConvNet(output_dim → 1, kernel=7, padding=6)
    final_conv = decoder.decoder[-1]
    if final_conv.padding > 0:
        state.conv_caches["dec.final"] = torch.zeros(
            B, final_conv.conv.in_channels, final_conv.padding, device=device, dtype=dtype
        )

    # KV cache: DynamicCache for Phase 1 (torch.compile handles it)
    state.kv_cache = DynamicCache()

    logger.info(
        "[streaming] pre-allocated state: %d conv caches, %d transconv caches",
        len(state.conv_caches),
        len(state.transconv_caches),
    )

    return state


def init_cudagraph_streaming_state(
    decoder,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_seq_len: int = 8000,
) -> CUDAGraphStreamingState:
    """Create streaming state for manual CUDA graph path (mode 2).

    Uses StaticStreamingKVCache with fixed-size buffers and integer-indexed
    cache lists for zero-overhead lookup.
    """
    B = 1
    state = CUDAGraphStreamingState()
    state.position_offset = torch.zeros(1, dtype=torch.long, device=device)

    conv_idx = 0
    transconv_idx = 0

    def _add_conv(key: str, module):
        nonlocal conv_idx
        if module.padding > 0:
            C_in = module.conv.in_channels
            state.conv_caches.append(
                torch.zeros(B, C_in, module.padding, device=device, dtype=dtype)
            )
        else:
            state.conv_caches.append(torch.zeros(0, device=device, dtype=dtype))
        state.conv_key_map[key] = conv_idx
        conv_idx += 1

    def _add_transconv(key: str, module):
        nonlocal transconv_idx
        if module.conv.kernel_size[0] > module.conv.stride[0]:
            C_in = module.conv.in_channels
            state.transconv_caches.append(
                torch.zeros(B, C_in, 1, device=device, dtype=dtype)
            )
        else:
            state.transconv_caches.append(torch.zeros(0, device=device, dtype=dtype))
        state.transconv_key_map[key] = transconv_idx
        transconv_idx += 1

    # Walk the exact same module tree order as decode_incremental
    _add_conv("pre_conv", decoder.pre_conv)

    for i, blocks in enumerate(decoder.upsample):
        _add_conv(f"up.{i}.cnx", blocks[1].dwconv)

    _add_conv("dec.0", decoder.decoder[0])

    num_decoder_blocks = len(decoder.decoder) - 3
    for i in range(num_decoder_blocks):
        block = decoder.decoder[1 + i]
        prefix = f"dec.{1 + i}"
        _add_transconv(f"{prefix}.tc", block.block[1])
        for j, ru in enumerate(block.block[2:]):
            _add_conv(f"{prefix}.ru{j}.c1", ru.conv1)
            _add_conv(f"{prefix}.ru{j}.c2", ru.conv2)

    _add_conv("dec.final", decoder.decoder[-1])

    # Static KV cache — derive shapes from actual projection weights
    # to avoid config mismatches (num_attention_heads vs num_key_value_heads)
    transformer = decoder.pre_transformer
    config = transformer.config
    first_layer = transformer.layers[0]
    # k_proj weight shape: [num_kv_heads * head_dim, hidden_size]
    kv_proj_shape = first_layer.self_attn.k_proj.weight.shape
    num_kv_heads = config.num_key_value_heads
    head_dim = kv_proj_shape[0] // num_kv_heads
    logger.info(
        "[streaming] KV cache: num_layers=%d, num_kv_heads=%d, head_dim=%d, max_seq_len=%d",
        config.num_hidden_layers, num_kv_heads, head_dim, max_seq_len,
    )
    state.kv_cache = StaticStreamingKVCache(
        num_layers=config.num_hidden_layers,
        batch_size=B,
        num_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )

    # Store decoder reference for eager fallback in decode_cudagraph
    state.decoder = decoder

    logger.info(
        "[streaming] CUDA graph state: %d conv caches, %d transconv caches, "
        "KV max_seq_len=%d",
        len(state.conv_caches),
        len(state.transconv_caches),
        max_seq_len,
    )

    return state


# ── Core decode function ──────────────────────────────────────────────


def _decode_incremental_inner(
    decoder,
    new_codes: torch.Tensor,
    state: StreamingDecoderState,
) -> torch.Tensor:
    """Inner decode function — the actual computation.

    Separated from decode_incremental() so torch.compile can wrap this
    without the @torch.inference_mode() decorator (which is incompatible
    with torch.compile's tracing).
    """
    # 1. Dequantize (stateless — codebook lookup + projection)
    hidden = decoder.quantizer.decode(new_codes)  # [B, 256, N]

    # 2. Pre-conv: CausalConvNet(256 → 1024, kernel=3)
    hidden, c = _conv_streaming(
        decoder.pre_conv, hidden, state.conv_caches["pre_conv"]
    )
    state.conv_caches["pre_conv"] = c

    # 3. Transformer with KV cache
    hidden = hidden.transpose(1, 2)  # [B, N, 1024]
    N = hidden.shape[1]

    cache_position = torch.arange(
        state.position_offset,
        state.position_offset + N,
        device=hidden.device,
    )

    result = decoder.pre_transformer(
        inputs_embeds=hidden,
        use_cache=True,
        past_key_values=state.kv_cache,
        cache_position=cache_position,
    )
    hidden = result.last_hidden_state  # [B, N, 1024]
    state.kv_cache = result.past_key_values
    state.position_offset += N

    hidden = hidden.permute(0, 2, 1)  # [B, 1024, N]

    # 4. Upsample stages: TransConv(stride=factor, kernel=factor) + ConvNeXtBlock
    for i, blocks in enumerate(decoder.upsample):
        transconv = blocks[0]
        convnext = blocks[1]
        hidden, tc = _transconv_streaming(
            transconv, hidden, state.transconv_caches.get(f"up.{i}")
        )
        if tc is not None:
            state.transconv_caches[f"up.{i}"] = tc
        hidden = _convnext_streaming(convnext, hidden, f"up.{i}.cnx", state)

    # 5. Decoder path
    wav = hidden

    # decoder[0]: CausalConvNet(1024 → decoder_dim, kernel=7)
    wav, c = _conv_streaming(
        decoder.decoder[0], wav, state.conv_caches["dec.0"]
    )
    state.conv_caches["dec.0"] = c

    # decoder[1..N-2]: DecoderBlocks (upsample 8x, 5x, 4x, 3x)
    num_decoder_blocks = len(decoder.decoder) - 3
    for i in range(num_decoder_blocks):
        wav = _decoder_block_streaming(
            decoder.decoder[1 + i], wav, f"dec.{1 + i}", state
        )

    # decoder[-2]: SnakeBeta (stateless)
    wav = decoder.decoder[-2](wav)

    # decoder[-1]: CausalConvNet(output_dim → 1, kernel=7)
    wav, c = _conv_streaming(
        decoder.decoder[-1], wav, state.conv_caches["dec.final"]
    )
    state.conv_caches["dec.final"] = c

    return wav.clamp(min=-1, max=1)


# ── torch.compile wrapper (Phase 1) ──────────────────────────────────


_compiled_decoder = None


def get_compiled_decoder():
    """Lazily compile the decode function with torch.compile."""
    global _compiled_decoder
    if _compiled_decoder is None:
        # Use "default" mode — "reduce-overhead" and "max-autotune" both
        # capture internal CUDA graphs that conflict with mutable conv cache
        # state passed between calls (cache tensors are graph inputs & outputs).
        # "default" still gets Inductor kernel fusion without the graph issues.
        logger.info("[streaming] compiling decode_incremental with torch.compile(mode='default')")
        _compiled_decoder = torch.compile(
            _decode_incremental_inner,
            mode="default",
            fullgraph=False,   # Allow graph breaks at DynamicCache
            dynamic=False,     # N is always 1 or 4
        )
    return _compiled_decoder


def warmup_compiled_decoder(decoder, state: StreamingDecoderState, device: str = "cuda"):
    """Warmup torch.compile for streaming decode shapes.

    Must be called with a fresh state that will be discarded after warmup.
    Returns nothing — the compiled function is cached globally.
    """
    compiled_fn = get_compiled_decoder()

    for N in [4, 1]:
        dummy = torch.zeros(1, 16, N, device=device, dtype=torch.long)
        with torch.inference_mode():
            # Mark step boundary so torch.compile's internal CUDA graph
            # management doesn't confuse cache tensors between runs
            torch.compiler.cudagraph_mark_step_begin()
            compiled_fn(decoder, dummy, state)
        torch.cuda.synchronize()

    logger.info("[streaming] torch.compile warmup done for N=1,4")


@torch.inference_mode()
def decode_incremental(
    decoder,
    new_codes: torch.Tensor,
    state: StreamingDecoderState,
    compiled: bool = False,
) -> torch.Tensor:
    """Decode only new codec frames using cached state.

    Args:
        decoder: Qwen3TTSTokenizerV2Decoder instance.
        new_codes: [B, 16, N] new codec frames (N = emit_every_frames).
        state: StreamingDecoderState from init_streaming_state().
        compiled: If True, use torch.compile'd version.

    Returns:
        [B, 1, N * total_upsample] new audio samples, clamped to [-1, 1].
        The state is mutated in-place.
    """
    if compiled and _compiled_decoder is not None:
        # Mark step boundary for torch.compile's internal CUDA graph
        # management — prevents cache tensor overwrite errors between calls
        torch.compiler.cudagraph_mark_step_begin()
        return _compiled_decoder(decoder, new_codes, state)
    return _decode_incremental_inner(decoder, new_codes, state)


# ── Manual CUDA graph path (Phase 2) ─────────────────────────────────


def _decode_graphable(
    decoder,
    new_codes: torch.Tensor,
    conv_caches: list[torch.Tensor],
    transconv_caches: list[torch.Tensor],
    kv_cache: StaticStreamingKVCache,
    position_offset: torch.Tensor,
) -> torch.Tensor:
    """Decode function written for CUDA graph compatibility.

    All cache accesses use integer indices (list indexing). The only
    Python-level branching is on static module properties (padding, stride)
    which are constant across graph replays.

    Conv/TransConv caches are mutated in-place via .copy_().
    KV cache is mutated via StaticStreamingKVCache.update().
    position_offset is mutated via .add_().
    """
    conv_idx = 0
    transconv_idx = 0

    def _conv_graph(conv_module, x):
        nonlocal conv_idx
        padding = conv_module.padding
        cache = conv_caches[conv_idx]
        if padding == 0:
            conv_idx += 1
            return conv_module.conv(x).contiguous()
        x_padded = torch.cat([cache, x], dim=-1)
        conv_caches[conv_idx].copy_(x_padded[:, :, -padding:])
        conv_idx += 1
        return conv_module.conv(x_padded).contiguous()

    def _transconv_graph(transconv_module, x):
        nonlocal transconv_idx
        stride = transconv_module.conv.stride[0]
        kernel_size = transconv_module.conv.kernel_size[0]
        if kernel_size <= stride:
            transconv_idx += 1
            raw = transconv_module.conv(x)
            left_pad = transconv_module.left_pad
            right_pad = transconv_module.right_pad
            if right_pad > 0:
                raw = raw[..., left_pad:-right_pad]
            elif left_pad > 0:
                raw = raw[..., left_pad:]
            return raw.contiguous()

        N = x.shape[-1]
        cache = transconv_caches[transconv_idx]
        x_with_cache = torch.cat([cache, x], dim=-1)
        transconv_caches[transconv_idx].copy_(x[:, :, -1:])
        transconv_idx += 1
        raw = transconv_module.conv(x_with_cache)
        return raw[:, :, stride : stride + N * stride].contiguous()

    # 1. Dequantize
    hidden = decoder.quantizer.decode(new_codes)

    # 2. Pre-conv
    hidden = _conv_graph(decoder.pre_conv, hidden)

    # 3. Transformer with static KV cache
    hidden = hidden.transpose(1, 2)
    N = hidden.shape[1]
    cache_position = torch.arange(N, device=hidden.device) + position_offset.squeeze()

    result = decoder.pre_transformer(
        inputs_embeds=hidden,
        use_cache=True,
        past_key_values=kv_cache,
        cache_position=cache_position,
    )
    hidden = result.last_hidden_state
    position_offset.add_(N)

    hidden = hidden.permute(0, 2, 1)

    # 4. Upsample stages
    for i, blocks in enumerate(decoder.upsample):
        transconv = blocks[0]
        convnext = blocks[1]
        # TransConv (no overlap for upsample stages, but we track idx)
        hidden = transconv.conv(hidden)
        left_pad = transconv.left_pad
        right_pad = transconv.right_pad
        if right_pad > 0:
            hidden = hidden[..., left_pad:-right_pad]
        elif left_pad > 0:
            hidden = hidden[..., left_pad:]
        hidden = hidden.contiguous()
        # ConvNeXtBlock dwconv
        residual = hidden
        hidden = _conv_graph(convnext.dwconv, hidden)
        hidden = hidden.permute(0, 2, 1)
        hidden = convnext.norm(hidden)
        hidden = convnext.pwconv1(hidden)
        hidden = convnext.act(hidden)
        hidden = convnext.pwconv2(hidden)
        hidden = convnext.gamma * hidden
        hidden = hidden.permute(0, 2, 1)
        hidden = residual + hidden

    # 5. Decoder path
    wav = hidden

    # decoder[0]: entry conv
    wav = _conv_graph(decoder.decoder[0], wav)

    # decoder[1..N-2]: DecoderBlocks
    num_decoder_blocks = len(decoder.decoder) - 3
    for i in range(num_decoder_blocks):
        block = decoder.decoder[1 + i]
        # SnakeBeta
        wav = block.block[0](wav)
        # TransConv
        wav = _transconv_graph(block.block[1], wav)
        # ResidualUnits
        for ru in block.block[2:]:
            res = wav
            wav = ru.act1(wav)
            wav = _conv_graph(ru.conv1, wav)
            wav = ru.act2(wav)
            wav = _conv_graph(ru.conv2, wav)
            wav = wav + res

    # decoder[-2]: SnakeBeta
    wav = decoder.decoder[-2](wav)

    # decoder[-1]: final conv
    wav = _conv_graph(decoder.decoder[-1], wav)

    return wav.clamp(min=-1, max=1)


def capture_streaming_cudagraphs(
    decoder,
    state: CUDAGraphStreamingState,
    device: str = "cuda",
    chunk_sizes: list[int] | None = None,
) -> dict:
    """Capture CUDA graphs for streaming decode at specific chunk sizes.

    Returns dict mapping chunk_size N → (graph, input_buf, output_buf).
    The state's caches are used as static buffers for graph capture.

    WARNING: After capture, the state should be reset before real inference
    (call state.kv_cache.reset() and re-init conv/transconv caches).
    """
    if chunk_sizes is None:
        chunk_sizes = [1, 2, 3, 4]

    graphs = {}
    graph_pool = None

    with torch.inference_mode():
        for N in reversed(chunk_sizes):  # Larger first for pool reuse
            input_buf = torch.zeros(1, 16, N, device=device, dtype=torch.long)

            # Warmup run
            output = _decode_graphable(
                decoder, input_buf,
                state.conv_caches, state.transconv_caches,
                state.kv_cache, state.position_offset,
            )
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=graph_pool):
                output_buf = _decode_graphable(
                    decoder, input_buf,
                    state.conv_caches, state.transconv_caches,
                    state.kv_cache, state.position_offset,
                )
            if graph_pool is None:
                graph_pool = graph.pool()

            graphs[N] = {
                "graph": graph,
                "input_buf": input_buf,
                "output_buf": output_buf,
            }
            torch.cuda.synchronize()
            logger.info("[streaming] captured CUDA graph for N=%d", N)

    logger.info("[streaming] CUDA graph capture done for chunk_sizes=%s", chunk_sizes)
    return graphs


def decode_cudagraph(
    new_codes: torch.Tensor,
    graphs: dict,
    state: CUDAGraphStreamingState,
) -> torch.Tensor:
    """Replay a captured CUDA graph for streaming decode.

    Falls back to eager _decode_graphable if no graph matches the chunk size.
    """
    N = new_codes.shape[2]
    entry = graphs.get(N)

    if entry is not None:
        entry["input_buf"].copy_(new_codes)
        entry["graph"].replay()
        return entry["output_buf"].clone()

    # Fallback to eager (shouldn't happen if chunk_sizes covers all emit sizes)
    logger.warning("[streaming] no CUDA graph for N=%d, using eager fallback", N)
    return _decode_graphable(
        state.decoder, new_codes,
        state.conv_caches, state.transconv_caches,
        state.kv_cache, state.position_offset,
    )
