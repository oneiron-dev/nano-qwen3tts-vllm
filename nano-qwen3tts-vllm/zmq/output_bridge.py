"""
ZMQ PUB bridge for publishing token outputs by (engine_type, request_id).

Message format (multipart):
  - Frame 0: topic (bytes) = "talker/<request_id>" or "predictor/<request_id>"
  - Frame 1: msg_type (bytes) = "token" or "done"
  - Frame 2: payload (bytes, serialized)

Payload for "token": msgpack-encoded dict with "token_ids" (list[int]) and
optionally "hidden_states" (numpy array serialized as bytes + shape + dtype).
Payload for "done": empty or msgpack dict with optional summary.
"""

import os
import socket
from typing import Any, Optional

import numpy as np

try:
    import zmq
except ImportError:
    zmq = None

try:
    import msgpack
except ImportError:
    msgpack = None


def _ensure_deps():
    if zmq is None:
        raise ImportError("pyzmq is required for ZMQ output bridge. Install with: pip install pyzmq")
    if msgpack is None:
        raise ImportError("msgpack is required for ZMQ output bridge. Install with: pip install msgpack")


def serialize_token_payload(token_ids: list[int], hidden_states: Optional[Any] = None) -> bytes:
    """Serialize token output for ZMQ. hidden_states: numpy array or torch tensor (will be converted to numpy)."""
    _ensure_deps()
    obj = {"token_ids": token_ids}
    if hidden_states is not None:
        if hasattr(hidden_states, "cpu"):
            arr = hidden_states.float().detach().cpu().numpy()
        else:
            arr = np.asarray(hidden_states)
        obj["hidden_states"] = arr.tobytes()
        obj["hidden_states_shape"] = list(arr.shape)
        obj["hidden_states_dtype"] = str(arr.dtype)
    return msgpack.packb(obj, use_bin_type=True)


def deserialize_token_payload(payload: bytes) -> dict:
    """Deserialize token payload from ZMQ. Returns dict with 'token_ids' and optionally 'hidden_states' (numpy)."""
    _ensure_deps()
    obj = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    if "hidden_states_shape" in obj:
        arr = np.frombuffer(obj["hidden_states"], dtype=obj["hidden_states_dtype"]).reshape(obj["hidden_states_shape"])
        obj["hidden_states"] = arr
        del obj["hidden_states_shape"]
        del obj["hidden_states_dtype"]
    return obj


def topic_for(engine_type: str, request_id: str) -> str:
    """Build ZMQ topic: e.g. talker/<request_id> or predictor/<request_id>."""
    return f"{engine_type}/{request_id}"


def find_available_port(start_port: int = 9555, max_attempts: int = 1000) -> int:
    """Find an available port starting from start_port.
    
    Args:
        start_port: Starting port number to check.
        max_attempts: Maximum number of ports to try.
    
    Returns:
        Available port number.
    
    Raises:
        RuntimeError: If no available port is found within max_attempts.
    """
    _ensure_deps()
    
    # Try using ZMQ to test ports, as regular socket check might not catch ZMQ-bound ports
    test_ctx = zmq.Context()
    
    try:
        for port in range(start_port, start_port + max_attempts):
            test_socket = None
            try:
                # Try to bind with ZMQ to see if port is actually available
                test_socket = test_ctx.socket(zmq.PUB)
                test_socket.setsockopt(zmq.LINGER, 0)
                test_address = f"tcp://127.0.0.1:{port}"
                test_socket.bind(test_address)
                # If we get here, port is available
                test_socket.close()
                return port
            except zmq.error.ZMQError:
                # Port is in use, try next one
                if test_socket:
                    try:
                        test_socket.close()
                    except:
                        pass
                continue
            except Exception:
                # Other error, try next port
                if test_socket:
                    try:
                        test_socket.close()
                    except:
                        pass
                continue
    finally:
        test_ctx.term()
    
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}")


def find_available_zmq_address(start_port: int = 9555, max_attempts: int = 1000) -> str:
    """Find an available ZMQ address starting from start_port.
    
    Args:
        start_port: Starting port number to check.
        max_attempts: Maximum number of ports to try.
    
    Returns:
        ZMQ address string (e.g., "tcp://127.0.0.1:9555").
    
    Raises:
        RuntimeError: If no available port is found within max_attempts.
    """
    port = find_available_port(start_port, max_attempts)
    return f"tcp://127.0.0.1:{port}"


class ZMQOutputBridge:
    """
    Publishes engine outputs over a ZMQ PUB socket.
    Topic = engine_type/request_id so subscribers can filter by request_id.
    """

    def __init__(self, bind_address: Optional[str] = None, auto_find_port: bool = True):
        """
        Initialize ZMQ output bridge.
        
        Args:
            bind_address: ZMQ bind address (e.g., "tcp://127.0.0.1:9555").
                         If None, uses QWEN_TTS_ZMQ_PUB env var or default "tcp://127.0.0.1:9555".
            auto_find_port: If True and bind_address port is in use, automatically find an available port.
        """
        _ensure_deps()
        self.bind_address = bind_address or os.environ.get("QWEN_TTS_ZMQ_PUB", "tcp://127.0.0.1:9555")
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        
        # Try to bind, and if port is in use and auto_find_port is True, find another port
        bind_success = False
        max_retries = 1 if not auto_find_port else 10  # Try up to 10 ports if auto_find_port is enabled
        
        for attempt in range(max_retries):
            try:
                self._socket.bind(self.bind_address)
                bind_success = True
                break
            except Exception as e:
                # Check for various forms of "address in use" errors
                error_str = str(e).lower()
                error_type = type(e).__name__
                
                # Check errno if available (EADDRINUSE = 98 on Linux)
                errno_match = False
                if hasattr(e, 'errno') and e.errno is not None:
                    errno_match = e.errno == 98 or (hasattr(zmq, 'EADDRINUSE') and e.errno == zmq.EADDRINUSE)
                
                # More permissive matching for address in use errors
                is_address_in_use = (
                    "address already in use" in error_str or
                    "address in use" in error_str or
                    ("addr" in error_str and ("already" in error_str or "in use" in error_str)) or
                    "eaddrinuse" in error_str or
                    errno_match or
                    isinstance(e, zmq.error.ZMQError)  # Any ZMQ error during bind is likely port-related
                )
                
                # If auto_find_port is enabled and this looks like an address-in-use error, try to find another port
                if auto_find_port and is_address_in_use and attempt < max_retries - 1:
                    # Extract port from address and find available port
                    original_address = self.bind_address
                    start_port = 9555
                    protocol = "tcp"
                    host = "127.0.0.1"
                    
                    if "://" in self.bind_address:
                        protocol, addr_part = self.bind_address.split("://", 1)
                        if ":" in addr_part:
                            host, port_str = addr_part.rsplit(":", 1)
                            try:
                                start_port = int(port_str)
                            except ValueError:
                                start_port = 9555
                        else:
                            host = addr_part if addr_part else "127.0.0.1"
                    else:
                        # No protocol, try to parse as host:port or just port
                        if ":" in self.bind_address:
                            host, port_str = self.bind_address.rsplit(":", 1)
                            try:
                                start_port = int(port_str)
                            except ValueError:
                                start_port = 9555
                        else:
                            try:
                                start_port = int(self.bind_address)
                            except ValueError:
                                start_port = 9555
                    
                    # Find available port (start from next port since current is in use)
                    # Try multiple strategies: next port range, then different port ranges
                    try:
                        # Strategy 1: Try next 500 ports
                        new_port = find_available_port(start_port + 1, max_attempts=500)
                    except RuntimeError:
                        try:
                            # Strategy 2: Try ports in a different range (10000-11000)
                            new_port = find_available_port(10000, max_attempts=1000)
                        except RuntimeError:
                            try:
                                # Strategy 3: Try random high ports (20000-25000)
                                import random
                                random_start = random.randint(20000, 24000)
                                new_port = find_available_port(random_start, max_attempts=5000)
                            except RuntimeError as final_error:
                                raise RuntimeError(
                                    f"Could not find any available port after trying multiple ranges. "
                                    f"Original port: {start_port}. Please free up some ports or specify a custom port via QWEN_TTS_ZMQ_PUB env var."
                                ) from final_error
                    
                    # Reconstruct address with new port (after successfully finding one)
                    new_address = f"{protocol}://{host}:{new_port}"
                    
                    # Close old socket and create new one
                    try:
                        self._socket.close()
                    except:
                        pass
                    self._socket = self._ctx.socket(zmq.PUB)
                    self._socket.setsockopt(zmq.LINGER, 0)
                    
                    # Update bind_address before retrying
                    self.bind_address = new_address
                    
                    # Show warning about port change
                    import warnings
                    warnings.warn(
                        f"Port {start_port} was in use, automatically switched to port {new_port}. "
                        f"Original address: {original_address}, New address: {new_address}",
                        UserWarning
                    )
                    
                    # Continue loop to try binding with new address
                    continue
                else:
                    # Port finding disabled, different error, or max retries reached - re-raise original
                    raise
        
        # If we get here and bind wasn't successful, something went wrong
        if not bind_success:
            raise RuntimeError(f"Failed to bind ZMQ socket after {max_retries} attempts")

    def publish_token(self, engine_type: str, request_id: str, token_ids: list[int], hidden_states: Optional[Any] = None):
        """Publish a token output. engine_type is 'talker' or 'predictor'."""
        import time as _time
        t0 = _time.perf_counter()
        topic = topic_for(engine_type, request_id)
        payload = serialize_token_payload(token_ids, hidden_states)
        t_ser = _time.perf_counter()
        self._socket.send_multipart([topic.encode("utf-8"), b"token", payload], flags=zmq.NOBLOCK)
        t_send = _time.perf_counter()
        if hidden_states is not None:
            import logging
            logging.getLogger(__name__).info(
                f"[zmq_pub] {engine_type}/{request_id[:8]} "
                f"serialize={(t_ser - t0)*1000:.2f}ms "
                f"send={(t_send - t_ser)*1000:.2f}ms "
                f"total={(t_send - t0)*1000:.2f}ms"
            )

    def publish_done(self, engine_type: str, request_id: str, payload: Optional[bytes] = None):
        """Publish a done message for this request."""
        topic = topic_for(engine_type, request_id)
        self._socket.send_multipart([topic.encode("utf-8"), b"done", payload or b""], flags=zmq.NOBLOCK)

    def close(self):
        self._socket.close()
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
