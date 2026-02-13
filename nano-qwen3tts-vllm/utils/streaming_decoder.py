"""Cached streaming decoder for Qwen3-TTS codec.

Eliminates the ~20x redundant compute in the windowed decode approach
by maintaining conv layer state and Transformer KV cache across calls.

Instead of re-decoding the entire 80-frame window each time 4 new frames
arrive, this processes only the new frames through the full decoder
pipeline with cached state.

Usage:
    from streaming_decoder import StreamingDecoderState, init_streaming_state, decode_incremental

    state = init_streaming_state(decoder, device="cuda")
    while True:
        new_codes = ...  # [1, 16, N] where N = emit_every_frames
        wav_new = decode_incremental(decoder, new_codes, state)
        # wav_new: [1, 1, N*1920] new audio samples
"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

logger = logging.getLogger(__name__)


@dataclass
class StreamingDecoderState:
    """All streaming state for one decode session."""

    conv_caches: dict[str, torch.Tensor] = field(default_factory=dict)
    transconv_caches: dict[str, torch.Tensor] = field(default_factory=dict)
    kv_cache: DynamicCache | None = None
    position_offset: int = 0


# ── Conv1d streaming ─────────────────────────────────────────────────


def _conv_streaming(
    conv_module,
    x: torch.Tensor,
    cache: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Process a CausalConvNet layer in streaming mode.

    Instead of left-padding with zeros every call, we prepend the cached
    tail of the previous input. This produces identical output to the
    non-streaming path for all chunks after the first.

    Args:
        conv_module: Qwen3TTSTokenizerV2CausalConvNet instance.
        x: [B, C_in, N] new input samples.
        cache: [B, C_in, padding] from previous call, or None for first call.

    Returns:
        (output [B, C_out, N'], new_cache [B, C_in, padding] or None).
    """
    padding = conv_module.padding

    if padding == 0:
        # Pointwise conv (kernel=1) — no state needed
        return conv_module.conv(x).contiguous(), None

    if cache is not None:
        x_padded = torch.cat([cache, x], dim=-1)
    else:
        x_padded = F.pad(x, (padding, 0), mode="constant", value=0)

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

    Based on the Moshi/vox-serve streaming ConvTranspose1d pattern.

    Args:
        transconv_module: Qwen3TTSTokenizerV2CausalTransConvNet instance.
        x: [B, C_in, N] new input samples.
        cache: [B, C_in, 1] last input sample from previous call, or None.

    Returns:
        (output [B, C_out, N*stride], new_cache or None).
    """
    stride = transconv_module.conv.stride[0]
    kernel_size = transconv_module.conv.kernel_size[0]

    if kernel_size <= stride:
        # No overlap between consecutive inputs — just run normally
        raw = transconv_module.conv(x)
        # Apply the same trimming as the non-streaming forward
        left_pad = transconv_module.left_pad
        right_pad = transconv_module.right_pad
        if right_pad > 0:
            raw = raw[..., left_pad:-right_pad]
        elif left_pad > 0:
            raw = raw[..., left_pad:]
        return raw.contiguous(), None

    # kernel > stride: overlap exists, need caching
    N = x.shape[-1]

    if cache is not None:
        x_with_cache = torch.cat([cache, x], dim=-1)  # [B, C, N+1]
    else:
        x_with_cache = F.pad(x, (1, 0), mode="constant", value=0)

    # Save last input sample for next call
    new_cache = x[:, :, -1:].clone()

    # ConvTranspose1d on [B, C, N+1]
    # Raw output length: N * stride + kernel_size
    raw = transconv_module.conv(x_with_cache)

    # The first `stride` output samples overlap with the previous chunk's tail.
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
        block.dwconv, x, state.conv_caches.get(cache_key)
    )
    if new_cache is not None:
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
        unit.conv1, x, state.conv_caches.get(f"{cache_prefix}.c1")
    )
    if c1 is not None:
        state.conv_caches[f"{cache_prefix}.c1"] = c1

    x = unit.act2(x)
    x, c2 = _conv_streaming(
        unit.conv2, x, state.conv_caches.get(f"{cache_prefix}.c2")
    )
    if c2 is not None:
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
        block.block[1], x, state.transconv_caches.get(tc_key)
    )
    if tc is not None:
        state.transconv_caches[tc_key] = tc

    # ResidualUnits (dilation 1, 3, 9)
    for j, ru in enumerate(block.block[2:]):
        x = _residual_unit_streaming(ru, x, f"{cache_prefix}.ru{j}", state)

    return x


# ── Full incremental decode ─────────────────────────────────────────


def init_streaming_state() -> StreamingDecoderState:
    """Create a fresh streaming state for a new synthesis request.

    Caches are allocated lazily on first use — no need to pre-allocate.
    """
    return StreamingDecoderState()


@torch.inference_mode()
def decode_incremental(
    decoder,
    new_codes: torch.Tensor,
    state: StreamingDecoderState,
) -> torch.Tensor:
    """Decode only new codec frames using cached state.

    Walks the decoder module tree manually, calling each layer with
    the appropriate caching logic. Produces identical output to the
    non-streaming path (up to floating point) for chunks after the first.

    Args:
        decoder: Qwen3TTSTokenizerV2Decoder instance.
        new_codes: [B, 16, N] new codec frames (N = emit_every_frames).
        state: StreamingDecoderState from init_streaming_state() or
               previous decode_incremental() call.

    Returns:
        [B, 1, N * total_upsample] new audio samples, clamped to [-1, 1].
        The state is mutated in-place.
    """
    # 1. Dequantize (stateless — codebook lookup + projection)
    hidden = decoder.quantizer.decode(new_codes)  # [B, 512, N]

    # 2. Pre-conv: CausalConvNet(512 → 1024, kernel=3)
    hidden, c = _conv_streaming(
        decoder.pre_conv, hidden, state.conv_caches.get("pre_conv")
    )
    if c is not None:
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
        decoder.decoder[0], wav, state.conv_caches.get("dec.0")
    )
    if c is not None:
        state.conv_caches["dec.0"] = c

    # decoder[1..N-2]: DecoderBlocks (upsample 8x, 5x, 4x, 3x)
    num_decoder_blocks = len(decoder.decoder) - 3  # exclude entry conv, final snake, final conv
    for i in range(num_decoder_blocks):
        wav = _decoder_block_streaming(
            decoder.decoder[1 + i], wav, f"dec.{1 + i}", state
        )

    # decoder[-2]: SnakeBeta (stateless)
    wav = decoder.decoder[-2](wav)

    # decoder[-1]: CausalConvNet(output_dim → 1, kernel=7)
    wav, c = _conv_streaming(
        decoder.decoder[-1], wav, state.conv_caches.get("dec.final")
    )
    if c is not None:
        state.conv_caches["dec.final"] = c

    return wav.clamp(min=-1, max=1)
