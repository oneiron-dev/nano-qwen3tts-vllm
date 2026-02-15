"""Talker model adapted for nano-vllm style continuous batching.

This module adapts Qwen3TTSTalkerForConditionalGeneration to work with nano-vllm's
architecture by:
1. Removing the nested code_predictor.generate() call
2. Simplifying forward() to only generate codebook 0
3. Adding methods compatible with nano-vllm's model runner
"""

import torch
from torch import nn
import torch.distributed as dist
from nano_qwen3tts_vllm.engine.model_runner.base import get_context


from nano_qwen3tts_vllm.layers.layernorm import Qwen3TTSRMSNorm
from nano_qwen3tts_vllm.models.qwen3_tts_share import Qwen3TTSDecoderLayer

class Qwen3TTSCodePredictorModel(nn.Module):
    def __init__(self, config, talker_config=None):
        super().__init__()
        self.vocab_size = config.vocab_size

        codec_embedding_dim = talker_config.hidden_size if talker_config else config.hidden_size

        self.layers = nn.ModuleList([Qwen3TTSDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, codec_embedding_dim) for _ in range(config.num_code_groups - 1)]
        )
        # Pre-stack codec embedding weights for tensor-op-based lookup
        self.register_buffer(
            "codec_embedding_weight",
            torch.stack([emb.weight for emb in self.codec_embedding], dim=0),
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
            
            
        hidden_states = self.norm(hidden_states)
        
        
        return hidden_states


class Qwen3TTSCodePredictorForCausalLM(nn.Module):
    def __init__(self, config, talker_config):
        super().__init__()
        self.model = Qwen3TTSCodePredictorModel(config, talker_config)
        self.vocab_size = config.vocab_size
        
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )
        # Pre-stack lm_head weights for tensor-op-based logit computation
        self.register_buffer(
            "lm_head_weight",
            torch.stack([head.weight for head in self.lm_head], dim=0),
        )

        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = torch.nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = torch.nn.Identity()
            
    def convert_state_dict(self, state_dict):
        """
        Convert state dict from original format to model format.

        Handles:
        1. Extracting code_predictor keys (remove "talker.code_predictor." prefix)
        2. No fusion: attention uses separate q_proj, k_proj, v_proj; MLP uses separate gate_proj, up_proj
        (Same approach as Qwen3TTSTalkerForCausalLM.convert_state_dict.)
        """
        transformed = {}

        for key, value in state_dict.items():
            if not key.startswith("talker.code_predictor."):
                continue
            key_without_prefix = key.replace("talker.code_predictor.", "", 1)
            transformed[key_without_prefix] = value

        return transformed
    
    def load_state_dict(self, state_dict, strict=True):
        state_dict = self.convert_state_dict(state_dict)

        # Synthesize stacked buffers from individual head/embedding weights
        # so strict loading succeeds (pretrained weights don't include these)
        num_heads = len(self.lm_head)
        if "lm_head_weight" not in state_dict:
            state_dict["lm_head_weight"] = torch.stack(
                [state_dict[f"lm_head.{i}.weight"] for i in range(num_heads)], dim=0
            )
        num_embs = len(self.model.codec_embedding)
        if "model.codec_embedding_weight" not in state_dict:
            state_dict["model.codec_embedding_weight"] = torch.stack(
                [state_dict[f"model.codec_embedding.{i}.weight"] for i in range(num_embs)], dim=0
            )

        super().load_state_dict(state_dict, strict=strict)
        
    def get_input_embeddings(self, input_ids, input_embeds, generation_steps):
        # Prefill path: return pre-computed embeddings as-is
        if input_embeds is not None and input_embeds.shape[1] > 1:
            generation_steps = input_embeds.shape[1] - 2  # hidden & layer 0
            input_embeds_final = [input_embeds]
            input_embeds_final = torch.stack(input_embeds_final)
            return input_embeds_final

        # Decode path: use stacked codec embedding weight for direct tensor lookup
        # (avoids Python nn.Embedding.__call__ dispatch per sequence)
        num_seqs = input_ids.size(0)
        if num_seqs == 1:
            step = int(generation_steps[0])
            emb_weight = self.model.codec_embedding_weight[step - 1]
            return emb_weight[input_ids[0]].unsqueeze(0)

        emb_idx = torch.tensor(
            [int(s) - 1 for s in generation_steps],
            device=input_ids.device, dtype=torch.long,
        )
        selected = self.model.codec_embedding_weight[emb_idx]
        return selected[torch.arange(num_seqs, device=input_ids.device), input_ids]
            
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)
        hidden_states = self.model(inputs_embeds, positions)
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        generation_steps: list[int],
    ) -> torch.Tensor:
        # hidden_states: (total_tokens, hidden_size) from model forward
        # We need one logit vector per sequence (last position) for the sampler: (num_seqs, vocab_size)
        num_seqs = len(generation_steps)
        hidden_states = hidden_states.view(num_seqs, -1, hidden_states.shape[-1])
        hidden_states = hidden_states[:, -1, :]  # [num_seqs, hidden_size]

        if num_seqs == 1:
            # Fast path: single sequence (production case)
            # Direct matmul with pre-stacked weight, no nn.Linear dispatch overhead
            logits = hidden_states @ self.lm_head_weight[generation_steps[0]].t()
        else:
            head_idx = torch.tensor(generation_steps, device=hidden_states.device, dtype=torch.long)
            head_weights = self.lm_head_weight[head_idx]  # [num_seqs, vocab_size, hidden_size]
            logits = torch.bmm(
                hidden_states.unsqueeze(1),
                head_weights.transpose(1, 2),
            ).squeeze(1)

        return logits

