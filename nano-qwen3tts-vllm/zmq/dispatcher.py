"""
ZMQ SUB dispatcher: subscribes to talker/ and predictor/, parses request_id from topic,
dispatches (engine_type, msg_type, payload_dict) to per-request_id asyncio queues.

Uses a dedicated thread for blocking ZMQ recv (so it is already waiting before the
first publish, avoiding slow-joiner). An asyncio task reads from the thread's inbox
and dispatches to request_queues.
"""

import asyncio
import logging
import queue
import threading
import time
from typing import Any

from nano_qwen3tts_vllm.zmq.output_bridge import deserialize_token_payload

logger = logging.getLogger(__name__)

try:
    import zmq
except ImportError:
    zmq = None


def _ensure_zmq():
    if zmq is None:
        raise ImportError("pyzmq is required. Install with: pip install pyzmq")


def _recv_thread(connect_address: str, inbox: queue.Queue) -> None:
    """Run in a dedicated thread: blocking recv, put (request_id, engine_type, msg_type, payload_dict, recv_time) in inbox."""
    _ensure_zmq()
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(connect_address)
    sub.setsockopt(zmq.SUBSCRIBE, b"talker/")
    sub.setsockopt(zmq.SUBSCRIBE, b"predictor/")
    try:
        while True:
            msg = sub.recv_multipart()
            t_recv = time.perf_counter()
            if len(msg) < 3:
                continue
            topic_b, msg_type_b, payload_b = msg[0], msg[1], msg[2]
            topic = topic_b.decode("utf-8")
            msg_type = msg_type_b.decode("utf-8")
            parts = topic.split("/", 1)
            engine_type = parts[0] if parts else ""
            request_id = parts[1] if len(parts) > 1 else ""
            if msg_type == "token":
                payload_dict = deserialize_token_payload(payload_b)
            else:
                payload_dict = {}
            inbox.put((request_id, engine_type, msg_type, payload_dict, t_recv))
    finally:
        sub.close()
        ctx.term()


async def run_dispatch_loop(
    inbox: queue.Queue,
    request_queues: dict[str, Any],
    queues_lock: asyncio.Lock,
) -> None:
    """Asyncio task: get from inbox (via executor) and put into request_queues[request_id].

    Uses non-blocking get with a fast poll loop to minimize latency --
    run_in_executor adds 0.5-1ms per call for the thread-pool round-trip.
    """
    dispatch_count = 0
    while True:
        # Drain ALL available messages in a tight loop before yielding.
        # With N CCUs, each cycle produces N talker + N predictor messages.
        # Processing them one-at-a-time with sleeps between each causes
        # zmq_to_dispatch delays of 15-30ms at high CCU counts.
        batch_processed = 0
        while True:
            try:
                item = inbox.get_nowait()
            except queue.Empty:
                break  # No more messages, exit inner loop
            if item is None:
                return  # Shutdown sentinel
            request_id, engine_type, msg_type, payload_dict, t_recv = item
            t_dispatch = time.perf_counter()
            async with queues_lock:
                q = request_queues.get(request_id)
            if q is not None:
                try:
                    q.put_nowait((engine_type, msg_type, payload_dict))
                    t_queued = time.perf_counter()
                    dispatch_count += 1
                    batch_processed += 1
                    if dispatch_count % 50 == 1:
                        logger.info(
                            f"[dispatch] #{dispatch_count} {engine_type}/{msg_type} "
                            f"req={request_id[:8]} "
                            f"zmq_to_dispatch={(t_dispatch - t_recv)*1000:.2f}ms "
                            f"dispatch_to_queue={(t_queued - t_dispatch)*1000:.2f}ms"
                        )
                except Exception:
                    pass

        if batch_processed == 0:
            await asyncio.sleep(0.0001)  # 0.1ms poll only when inbox is empty
        else:
            await asyncio.sleep(0)  # Yield to let other tasks run


def start_dispatcher_thread(connect_address: str) -> tuple[threading.Thread, queue.Queue]:
    """Start the ZMQ recv thread and return (thread, inbox). Caller must run run_dispatch_loop(inbox, ...) as asyncio task."""
    _ensure_zmq()
    inbox = queue.Queue()
    thread = threading.Thread(target=_recv_thread, args=(connect_address, inbox), daemon=True)
    thread.start()
    return thread, inbox
