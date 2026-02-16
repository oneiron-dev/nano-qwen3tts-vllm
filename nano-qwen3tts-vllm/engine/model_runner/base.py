import os
import pickle
import torch
import torch.distributed as dist
from safetensors.torch import load_file
import json
from typing import Optional
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nano_qwen3tts_vllm.config import Config
from nano_qwen3tts_vllm.engine.sequence import Sequence
from nano_qwen3tts_vllm.layers.sampler import Sampler
from nano_qwen3tts_vllm.utils.context import set_context, get_context, reset_context
from nano_qwen3tts_vllm.config import Qwen3TTSConfig
from nano_qwen3tts_vllm.models.qwen3_tts_talker import Qwen3TTSTalkerForCausalLM
from nano_qwen3tts_vllm.models.qwen3_tts_predictor import Qwen3TTSCodePredictorForCausalLM
import logging
logger = logging.getLogger(__name__)



MODEL_TYPE_MAPPING = {
    "talker": Qwen3TTSTalkerForCausalLM,
    "predictor": Qwen3TTSCodePredictorForCausalLM,
}


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.fi_wrapper = None

        if not dist.is_initialized():
            # If world_size==1, pick a free port automatically so multiple
            # independent server processes don't clash on the default port.
            port = self.config.distributed_port
            if self.world_size == 1:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    port = s.getsockname()[1]
            dist.init_process_group(
                "nccl",
                f"tcp://localhost:{port}",
                world_size=self.world_size,
                rank=rank,
            )
        torch.cuda.set_device(rank)
        # torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda")

    def post_init(self, rank: int):
        default_dtype = torch.get_default_dtype()
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def load_model(self, config: Config):
        ...

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager and hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([], input_embeds=torch.zeros(1, 8, self.model_config.hidden_size)) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = self.model_config
        torch_dtype = torch.bfloat16
        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * torch_dtype.itemsize

        if getattr(config, "process_gpu_memory_fraction", None) is not None:
            # Multi-process on one GPU: cap this process to a fraction of total (e.g. 0.5 for 2 processes)
            effective_total = total * config.process_gpu_memory_fraction
            used = torch.cuda.memory_allocated()
            raw_blocks = int(effective_total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        else:
            used = total - free
            raw_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes

        config.num_kvcache_blocks = max(1, raw_blocks)
        if raw_blocks <= 0:
            import warnings
            warnings.warn(
                f"KV cache allocation would be 0 (gpu_memory_utilization={config.gpu_memory_utilization}, "
                f"free~{total * config.gpu_memory_utilization - used - peak + current:.0f} bytes). Using 1 block."
            )
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        self._init_flashinfer()

    # ---- FlashInfer integration ----

    def _init_flashinfer(self):
        """Initialize FlashInfer (if available). Wrappers created later in _fi_create_wrappers()."""
        try:
            import flashinfer
            self._flashinfer = flashinfer
        except ImportError:
            self.fi_wrapper = None
            self._fi_wrappers = {}
            return

        hf_config = self.model_config
        self._fi_num_kv_heads = hf_config.num_key_value_heads // self.world_size
        self._fi_num_qo_heads = hf_config.num_attention_heads // self.world_size
        self._fi_head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        self._fi_max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size

        self._fi_wrappers = {}  # graph_bs -> wrapper (populated in _fi_create_wrappers)
        self._fi_buffers = {}   # graph_bs -> wrapper buffers (kept alive for graph replay)
        self._fi_debug = os.getenv("NANO_QWEN3TTS_FI_DEBUG", "0") == "1"
        self.fi_wrapper = None  # active wrapper (set in _fi_plan)

        # Collect attention modules for fast wrapper swapping
        self._fi_attn_modules = [m for m in self.model.modules() if hasattr(m, "fi_wrapper")]

        if self.enforce_eager:
            # Single eager wrapper (no fixed batch size constraint)
            self._fi_eager_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            self.fi_wrapper = self._flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self._fi_eager_workspace, kv_layout="NHD", use_cuda_graph=False,
            )
            for m in self._fi_attn_modules:
                m.fi_wrapper = self.fi_wrapper

        logger.info(
            f"[flashinfer] Initialized (eager={self.enforce_eager}, "
            f"qo_heads={self._fi_num_qo_heads}, kv_heads={self._fi_num_kv_heads}, "
            f"head_dim={self._fi_head_dim})"
        )

    def _fi_create_wrappers(self, graph_bs_list):
        """Create one FlashInfer wrapper per graph batch size for CUDA graph compat."""
        if not hasattr(self, "_flashinfer") or self.enforce_eager:
            return

        flashinfer = self._flashinfer
        max_num_blocks = self._fi_max_num_blocks

        # Store per-wrapper buffer references for direct runtime access.
        self._fi_buffers = {}

        for bs in graph_bs_list:
            max_pages = bs * max_num_blocks
            indptr_buf = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
            indices_buf = torch.zeros(max(max_pages, 1), dtype=torch.int32, device="cuda")
            lpl_buf = torch.ones(bs, dtype=torch.int32, device="cuda")

            # split_kv is enabled with CUDA graphs, so each wrapper needs its own float workspace.
            float_ws_bytes = (
                self._fi_num_qo_heads * bs * self._fi_head_dim * torch.bfloat16.itemsize
                + self._fi_num_qo_heads * bs * torch.float32.itemsize
                + 1024 * 1024
            )
            float_ws = torch.empty(float_ws_bytes, dtype=torch.uint8, device="cuda")

            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                float_ws, kv_layout="NHD", use_cuda_graph=True,
                paged_kv_indptr_buffer=indptr_buf,
                paged_kv_indices_buffer=indices_buf,
                paged_kv_last_page_len_buffer=lpl_buf,
            )
            self._fi_wrappers[bs] = wrapper
            self._fi_buffers[bs] = {
                "indptr": indptr_buf,
                "indices": indices_buf,
                "lpl": lpl_buf,
                "float_ws": float_ws,
            }

        # Scratch tensors for padding actual_bs -> graph_bs
        max_bs = max(graph_bs_list)
        self._fi_scratch_indptr = torch.zeros(max_bs + 1, dtype=torch.int32, device="cuda")
        self._fi_scratch_indices = torch.zeros(max_bs * max_num_blocks, dtype=torch.int32, device="cuda")
        self._fi_scratch_lpl = torch.ones(max_bs, dtype=torch.int32, device="cuda")
        self._fi_dummy_indptr_tail = torch.arange(1, max_bs + 1, dtype=torch.int32, device="cuda")

        logger.info(f"[flashinfer] Created {len(graph_bs_list)} per-bs wrappers with per-wrapper workspaces")

    def _fi_update_buffers(self, graph_bs, fi_indptr, fi_indices, fi_last_page_len):
        """Update FlashInfer page-table buffers without calling plan() on graph replay."""
        if not self._fi_wrappers:
            if self._fi_debug:
                logger.info("[fi_update] no wrappers, returning")
            return

        bufs = self._fi_buffers.get(graph_bs)
        if bufs is None:
            if self._fi_debug:
                logger.warning("[fi_update] no buffers for graph_bs=%d, available=%s", graph_bs, list(self._fi_buffers.keys())[:5])
            return

        actual_bs = fi_indptr.size(0) - 1
        total_real_pages = fi_indices.size(0)

        if self._fi_debug:
            logger.info(
                "[fi_update] graph_bs=%d actual_bs=%d pages=%d indptr=%s indices=%s lpl=%s",
                graph_bs, actual_bs, total_real_pages,
                fi_indptr.tolist(), fi_indices.tolist()[:8], fi_last_page_len.tolist()
            )

        if self._fi_debug:
            assert fi_indptr.size(0) == actual_bs + 1, "fi_indptr size mismatch"
            assert fi_last_page_len.size(0) == actual_bs, "fi_last_page_len size mismatch"
            assert actual_bs <= graph_bs, f"actual_bs ({actual_bs}) exceeds graph_bs ({graph_bs})"
            assert total_real_pages <= bufs["indices"].numel(), (
                f"FlashInfer indices overflow: pages={total_real_pages}, capacity={bufs['indices'].numel()}"
            )

        if graph_bs == actual_bs:
            bufs["indptr"].copy_(fi_indptr, non_blocking=True)
            if total_real_pages > 0:
                bufs["indices"][:total_real_pages].copy_(fi_indices, non_blocking=True)
            bufs["lpl"].copy_(fi_last_page_len, non_blocking=True)
            return

        num_dummy = graph_bs - actual_bs
        total_pages = total_real_pages + num_dummy

        if self._fi_debug:
            assert total_pages <= bufs["indices"].numel(), (
                f"FlashInfer indices overflow: pages={total_pages}, capacity={bufs['indices'].numel()}"
            )

        # Build padded indptr: real sequences first, then +1 page per dummy sequence.
        scratch_indptr = self._fi_scratch_indptr
        scratch_indptr[:actual_bs + 1].copy_(fi_indptr, non_blocking=True)
        scratch_indptr[actual_bs + 1:graph_bs + 1] = (
            total_real_pages + self._fi_dummy_indptr_tail[:num_dummy]
        )

        # Build padded indices with dummy page 0 for padded sequences.
        scratch_indices = self._fi_scratch_indices
        if total_real_pages > 0:
            scratch_indices[:total_real_pages].copy_(fi_indices, non_blocking=True)
        scratch_indices[total_real_pages:total_pages].fill_(0)

        # Build padded last-page lengths with 1 token for padded sequences.
        scratch_lpl = self._fi_scratch_lpl
        scratch_lpl[:actual_bs].copy_(fi_last_page_len, non_blocking=True)
        scratch_lpl[actual_bs:graph_bs].fill_(1)

        # Copy padded metadata into graph-captured wrapper buffers.
        bufs["indptr"].copy_(scratch_indptr[:graph_bs + 1], non_blocking=True)
        if total_pages > 0:
            bufs["indices"][:total_pages].copy_(scratch_indices[:total_pages], non_blocking=True)
        bufs["lpl"].copy_(scratch_lpl[:graph_bs], non_blocking=True)

    def _fi_plan(self, batch_size, fi_indptr, fi_indices, fi_last_page_len):
        """Call FlashInfer plan() for capture-time setup and eager decode only."""
        if not self._fi_wrappers and self.fi_wrapper is None:
            return

        if not self.enforce_eager:
            # CUDA graph mode: use per-bs wrapper
            wrapper = self._fi_wrappers.get(batch_size)
            if wrapper is None:
                return

            # Swap active wrapper on all attention layers.
            # During graph capture this binds each bs graph to the correct wrapper.run() launches.
            if self.fi_wrapper is not wrapper:
                self.fi_wrapper = wrapper
                for m in self._fi_attn_modules:
                    m.fi_wrapper = wrapper

            # Pad actual_bs to graph batch_size
            actual_bs = fi_indptr.size(0) - 1
            if batch_size > actual_bs:
                self._fi_scratch_indptr[:actual_bs + 1].copy_(fi_indptr)
                self._fi_scratch_indptr[actual_bs + 1:batch_size + 1] = self._fi_scratch_indptr[actual_bs]
                self._fi_scratch_lpl[:actual_bs].copy_(fi_last_page_len)
                self._fi_scratch_lpl[actual_bs:batch_size].fill_(1)
                fi_indptr = self._fi_scratch_indptr[:batch_size + 1]
                fi_last_page_len = self._fi_scratch_lpl[:batch_size]
        else:
            wrapper = self.fi_wrapper

        wrapper.plan(
            fi_indptr, fi_indices, fi_last_page_len,
            num_qo_heads=self._fi_num_qo_heads,
            num_kv_heads=self._fi_num_kv_heads,
            head_dim=self._fi_head_dim,
            page_size=self.block_size,
            q_data_type=torch.bfloat16,
        )

    def _build_fi_metadata(self, seqs):
        """Convert sequence block tables to FlashInfer CSR page-table format."""
        if not self._fi_wrappers and self.fi_wrapper is None:
            return None, None, None

        indptr = [0]
        indices = []
        last_page_len = []
        for seq in seqs:
            indices.extend(seq.block_table)
            indptr.append(len(indices))
            last_page_len.append(seq.last_block_num_tokens)

        fi_indptr = torch.tensor(indptr, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        fi_indices = torch.tensor(indices, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        fi_last_page_len = torch.tensor(last_page_len, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return fi_indptr, fi_indices, fi_last_page_len

    # ---- End FlashInfer ----

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        input_embeds = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            input_embeds.extend(seq.input_embeds[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        input_embeds = torch.cat([e if e.dim() > 1 else e.unsqueeze(0) for e in input_embeds], dim=0).to(dtype=torch.bfloat16)
        if input_embeds.device.type != "cuda":
            input_embeds = input_embeds.pin_memory().cuda(non_blocking=True)
        else:
            input_embeds = input_embeds.cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, input_embeds, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        fi_indptr, fi_indices, fi_last_page_len = self._build_fi_metadata(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables,
                    fi_indptr=fi_indptr, fi_indices=fi_indices, fi_last_page_len=fi_last_page_len)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, input_embeds: Optional[torch.Tensor] = None):
        model_input = input_embeds if input_embeds is not None else input_ids

        if is_prefill or self.enforce_eager or input_ids.size(0) > 512 or input_embeds is not None:
            if not is_prefill and (self._fi_wrappers or self.fi_wrapper is not None):
                context = get_context()
                self._fi_plan(input_ids.size(0), context.fi_indptr, context.fi_indices, context.fi_last_page_len)
            return self.model.compute_logits(self.model(model_input, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph_bs = next(x for x in self.graph_bs if x >= bs)
            graph = self.graphs[graph_bs]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            if self._fi_wrappers:
                self._fi_update_buffers(graph_bs, context.fi_indptr, context.fi_indices, context.fi_last_page_len)
            if self._fi_debug:
                logger.info("[run_model] graph replay: bs=%d graph_bs=%d fi_wrappers=%s", bs, graph_bs, bool(self._fi_wrappers))
            graph.replay()
            if self._fi_debug:
                torch.cuda.synchronize()
                logger.info("[run_model] graph replay done, outputs[:bs] norm=%.4f", graph_vars["outputs"][:bs].norm().item())
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_embeds = None
        if is_prefill:
            input_ids, input_embeds, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill, input_embeds)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # Create per-bs FlashInfer wrappers before graph capture
        self._fi_create_wrappers(self.graph_bs)

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            if self._fi_wrappers:
                dummy_indptr = torch.arange(bs + 1, dtype=torch.int32, device="cuda") * max_num_blocks
                dummy_indices = torch.zeros(bs * max_num_blocks, dtype=torch.int32, device="cuda")
                dummy_lpl = torch.full((bs,), self.block_size, dtype=torch.int32, device="cuda")
                self._fi_plan(bs, dummy_indptr, dummy_indices, dummy_lpl)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            if self._fi_wrappers:
                self._fi_plan(bs, dummy_indptr, dummy_indices, dummy_lpl)
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
