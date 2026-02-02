"""
Engine core processes: dedicated blocking loops for talker and predictor
engines. Each engine runs in its own process with a blocking loop that
checks for requests and executes steps when work is available, eliminating
async overhead.
"""

import queue
from dataclasses import dataclass
from typing import Any, Literal, Optional
import torch


@dataclass
class EngineRequest:
    """Request data structure for engine operations."""
    action: Literal["add_request", "clear_request"]
    request_id: str
    inputs_embeds: Optional[torch.Tensor] = None
    sampling_params: Optional[Any] = None  # SamplingParams


def talker_engine_core_loop(
    request_queue: queue.Queue,
    zmq_bridge_address: str,
    model_path: str,
    enforce_eager: bool = False,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.3,
) -> None:
    """
    Blocking loop for talker engine running in a dedicated process.

    Args:
        request_queue: Multiprocessing queue for receiving requests
        zmq_bridge_address: Address for ZMQ PUB socket
        model_path: Path to model weights
        enforce_eager: Whether to enforce eager execution
        tensor_parallel_size: Tensor parallel size
        gpu_memory_utilization: GPU memory utilization fraction
    """
    # Import inside process to avoid CUDA initialization issues
    from nano_qwen3tts_vllm.llm import TalkerLLM
    from nano_qwen3tts_vllm.zmq.output_bridge import ZMQOutputBridge

    # Initialize engine and ZMQ bridge inside this process
    talker_llm = TalkerLLM(
        model_path,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    zmq_bridge = ZMQOutputBridge(bind_address=zmq_bridge_address)

    print("[TalkerCore] Engine initialized, starting blocking loop")

    try:
        while True:
            # Check for new requests with short timeout to avoid
            # busy-waiting
            try:
                request: EngineRequest = request_queue.get(timeout=0.001)

                if request.action == "shutdown":
                    print("[TalkerCore] Received shutdown signal")
                    break
                elif request.action == "add_request":
                    talker_llm.add_request(
                        [request.inputs_embeds],
                        request.sampling_params,
                        request_id=request.request_id,
                    )
                elif request.action == "clear_request":
                    talker_llm.clear_request(request.request_id)

            except queue.Empty:
                pass  # No request, continue to check for work

            # Check if engine has work to do
            has_work = bool(
                talker_llm.scheduler.waiting or talker_llm.scheduler.running
            )

            if has_work:
                try:
                    _, _, outputs_all = talker_llm.step_with_outputs()

                    # Publish all outputs via ZMQ
                    for tup in outputs_all:
                        rid, seq_id, tids, hstates, is_fin = tup
                        zmq_bridge.publish_token(
                            "talker", rid, tids, hstates
                        )
                        if is_fin:
                            zmq_bridge.publish_done("talker", rid)

                except Exception as e:
                    print(f"[TalkerCore] Error during step: {e}")
                    raise

    finally:
        print("[TalkerCore] Shutting down")
        zmq_bridge.close()


def predictor_engine_core_loop(
    request_queue: queue.Queue,
    zmq_bridge_address: str,
    model_path: str,
    enforce_eager: bool = False,
    tensor_parallel_size: int = 1,
) -> None:
    """
    Blocking loop for predictor engine running in a dedicated process.

    Args:
        request_queue: Multiprocessing queue for receiving requests
        zmq_bridge_address: Address for ZMQ PUB socket
        model_path: Path to model weights
        enforce_eager: Whether to enforce eager execution
        tensor_parallel_size: Tensor parallel size
    """
    # Import inside process to avoid CUDA initialization issues
    from nano_qwen3tts_vllm.llm import PredictorLLM
    from nano_qwen3tts_vllm.zmq.output_bridge import ZMQOutputBridge

    # Initialize engine and ZMQ bridge inside this process
    predictor_llm = PredictorLLM(
        model_path,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tensor_parallel_size,
    )

    zmq_bridge = ZMQOutputBridge(bind_address=zmq_bridge_address)

    print("[PredictorCore] Engine initialized, starting blocking loop")

    try:
        while True:
            # Check for new requests with short timeout to avoid
            # busy-waiting
            try:
                request: EngineRequest = request_queue.get(timeout=0.001)

                if request.action == "shutdown":
                    print("[PredictorCore] Received shutdown signal")
                    break
                elif request.action == "add_request":
                    predictor_llm.add_request(
                        [request.inputs_embeds],
                        request.sampling_params,
                        request_id=request.request_id,
                    )
                elif request.action == "clear_request":
                    predictor_llm.clear_request(request.request_id)

            except queue.Empty:
                pass  # No request, continue to check for work

            # Check if engine has work to do
            has_work = bool(
                predictor_llm.scheduler.waiting or
                predictor_llm.scheduler.running
            )

            if has_work:
                try:
                    outputs, _ = predictor_llm.step()

                    # Publish outputs via ZMQ
                    for request_id, seq_id, token_ids in outputs:
                        zmq_bridge.publish_token(
                            "predictor", request_id, token_ids, None
                        )

                except Exception as e:
                    print(f"[PredictorCore] Error during step: {e}")
                    raise

    finally:
        print("[PredictorCore] Shutting down")
        zmq_bridge.close()
