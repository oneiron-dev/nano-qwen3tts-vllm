"""
Engine core: shared data structures for engine operations.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional
import torch


@dataclass
class EngineRequest:
    """Request data structure for engine operations."""
    action: Literal["add_request", "clear_request", "shutdown"]
    request_id: str
    inputs_embeds: Optional[torch.Tensor] = None
    sampling_params: Optional[Any] = None  # SamplingParams
