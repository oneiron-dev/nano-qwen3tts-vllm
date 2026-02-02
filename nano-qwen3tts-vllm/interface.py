import asyncio
import queue
import threading
import uuid
import torch
import torch.multiprocessing as mp
import soundfile as sf
import time
from nano_qwen3tts_vllm.utils.prompt import prepare_custom_voice_prompt
from nano_qwen3tts_vllm.processor import Qwen3TTSProcessor
from nano_qwen3tts_vllm.utils.generation import prepare_inputs
from nano_qwen3tts_vllm.llm import TalkerLLM, PredictorLLM
from nano_qwen3tts_vllm.sampling_params import SamplingParams
from nano_qwen3tts_vllm.engine.engine_core import EngineRequest


torch.manual_seed(42)
# Use Qwen3-TTS processor when available so tokenization matches Qwen3TTSModel.generate_custom_voice exactly
def _get_processor(model_path: str):
    try:
        from qwen_tts.core.models import Qwen3TTSProcessor as Qwen3TTSProcessorHF
        return Qwen3TTSProcessorHF.from_pretrained(model_path, fix_mistral_regex=True)
    except ImportError:
        return Qwen3TTSProcessor.from_pretrained(model_path, fix_mistral_regex=True)


class Qwen3TTSInterface:
    def __init__(self, model_path: str, enforce_eager: bool = False, tensor_parallel_size: int = 1, zmq_bridge=None):
        self.model_path = model_path
        self.enforce_eager = enforce_eager
        self.tensor_parallel_size = tensor_parallel_size
        self.zmq_bridge = zmq_bridge
        
        # For sync mode (non-ZMQ): initialize engines in main process
        if zmq_bridge is None:
            self.talker_llm = TalkerLLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.3)
            self.predictor_llm = PredictorLLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tensor_parallel_size)
            self.processor = _get_processor(model_path)
            self.model_config = self.talker_llm.model_runner.full_config
            
            self.text_embedding = self.talker_llm.model_runner.model.get_text_embeddings()
            self.input_embedding = self.talker_llm.model_runner.model.get_input_embeddings()
            self.text_projection = self.talker_llm.model_runner.model.text_projection
            
            self.predictor_input_embeddings = self.predictor_llm.model_runner.model.model.codec_embedding
        else:
            # For async/ZMQ mode: engines run in separate processes, we need embeddings for prep
            # Initialize a temporary engine just to get embeddings
            temp_talker = TalkerLLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.3)
            temp_predictor = PredictorLLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tensor_parallel_size)
            self.processor = _get_processor(model_path)
            self.model_config = temp_talker.model_runner.full_config
            
            self.text_embedding = temp_talker.model_runner.model.get_text_embeddings()
            self.input_embedding = temp_talker.model_runner.model.get_input_embeddings()
            self.text_projection = temp_talker.model_runner.model.text_projection
            self.predictor_input_embeddings = temp_predictor.model_runner.model.model.codec_embedding
            
            # Clean up temporary engines
            del temp_talker
            del temp_predictor
            torch.cuda.empty_cache()
            
            # Create multiprocessing queues for sending requests to engine processes
            mp_ctx = mp.get_context('spawn')
            self._talker_request_queue = mp_ctx.Queue()
            self._predictor_request_queue = mp_ctx.Queue()
            
            # Start engine core processes
            # Use separate ZMQ ports for talker and predictor to avoid conflicts
            from nano_qwen3tts_vllm.engine.engine_core import talker_engine_core_loop, predictor_engine_core_loop
            
            # Parse base address and create separate addresses for each engine
            base_addr = zmq_bridge.bind_address
            if ":" in base_addr:
                proto_host, port = base_addr.rsplit(":", 1)
                base_port = int(port)
                talker_addr = f"{proto_host}:{base_port}"
                predictor_addr = f"{proto_host}:{base_port + 1}"
            else:
                talker_addr = base_addr
                predictor_addr = base_addr
            
            self._talker_process = mp_ctx.Process(
                target=talker_engine_core_loop,
                args=(
                    self._talker_request_queue,
                    talker_addr,
                    model_path,
                ),
                kwargs={
                    'enforce_eager': enforce_eager,
                    'tensor_parallel_size': tensor_parallel_size,
                    'gpu_memory_utilization': 0.1,
                },
                daemon=False,
            )
            
            self._predictor_process = mp_ctx.Process(
                target=predictor_engine_core_loop,
                args=(
                    self._predictor_request_queue,
                    predictor_addr,
                    model_path,
                ),
                kwargs={
                    'enforce_eager': enforce_eager,
                    'tensor_parallel_size': tensor_parallel_size,
                    'gpu_memory_utilization': 0.1,
                },
                daemon=False,
            )
            
            # Store addresses for dispatcher
            self._talker_zmq_address = talker_addr
            self._predictor_zmq_address = predictor_addr
            
            self._talker_process.start()
            self._predictor_process.start()
            print(f"[Interface] Started talker process PID={self._talker_process.pid}")
            print(f"[Interface] Started predictor process PID={self._predictor_process.pid}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ZMQ path: asyncio queues for receiving outputs from ZMQ dispatcher
        self._request_queues: dict[str, asyncio.Queue] = {}
        self._queues_lock = asyncio.Lock()
        self._zmq_tasks: list[asyncio.Task] = []
        self._zmq_inbox: queue.Queue | None = None
        self._zmq_tasks_started = False
        # Serialize request prep (GPU work) so event loop can run while another request prepares.
        self._prep_lock = threading.Lock()

    async def start_zmq_tasks(self) -> None:
        """Start the ZMQ dispatcher (thread + asyncio task). Engine loops run in separate processes. Call once before generate_async when zmq_bridge is set."""
        if self.zmq_bridge is None:
            raise RuntimeError("start_zmq_tasks requires zmq_bridge to be set on the interface")
        if self._zmq_tasks_started:
            return
        self._zmq_tasks_started = True
        from nano_qwen3tts_vllm.zmq.dispatcher import start_dispatcher_thread, run_dispatch_loop
        
        # Start dispatcher threads for both talker and predictor addresses
        # (they publish on separate ports in separate processes)
        if hasattr(self, '_talker_zmq_address'):
            # New architecture: separate processes
            _, inbox_talker = start_dispatcher_thread(self._talker_zmq_address)
            _, inbox_predictor = start_dispatcher_thread(self._predictor_zmq_address)
            
            # Merge both inboxes into the main inbox
            import queue as sync_queue
            self._zmq_inbox = sync_queue.Queue()
            
            def merge_inboxes():
                """Merge messages from both inboxes into one."""
                import threading
                def forwarder(src, dst):
                    while True:
                        try:
                            msg = src.get()
                            if msg is None:
                                break
                            dst.put(msg)
                        except Exception:
                            break
                
                t1 = threading.Thread(target=forwarder, args=(inbox_talker, self._zmq_inbox), daemon=True)
                t2 = threading.Thread(target=forwarder, args=(inbox_predictor, self._zmq_inbox), daemon=True)
                t1.start()
                t2.start()
            
            merge_inboxes()
        else:
            # Old architecture: single process (for backwards compatibility)
            _, self._zmq_inbox = start_dispatcher_thread(self.zmq_bridge.bind_address)
        
        t1 = asyncio.create_task(run_dispatch_loop(self._zmq_inbox, self._request_queues, self._queues_lock))
        self._zmq_tasks.append(t1)
        # Give the recv thread time to connect and enter recv (avoid ZMQ slow-joiner)
        await asyncio.sleep(0.2)

    async def stop_zmq_tasks(self) -> None:
        """Stop ZMQ tasks and engine processes. Puts sentinel in inbox to unblock executor thread."""
        if not self._zmq_tasks:
            return
        
        # Send shutdown signal to engine processes if they exist
        if hasattr(self, '_talker_request_queue') and hasattr(self, '_talker_process'):
            shutdown_request = EngineRequest(action="shutdown", request_id="")
            self._talker_request_queue.put(shutdown_request)
            print(f"[Interface] Sent shutdown signal to talker process")
            
        if hasattr(self, '_predictor_request_queue') and hasattr(self, '_predictor_process'):
            shutdown_request = EngineRequest(action="shutdown", request_id="")
            self._predictor_request_queue.put(shutdown_request)
            print(f"[Interface] Sent shutdown signal to predictor process")
        
        # Unblock the thread blocked on inbox.get() in run_in_executor so shutdown_default_executor() can finish
        if self._zmq_inbox is not None:
            self._zmq_inbox.put(None)
        
        # Cancel asyncio tasks
        for t in self._zmq_tasks:
            t.cancel()
        await asyncio.gather(*self._zmq_tasks, return_exceptions=True)
        self._zmq_tasks.clear()
        self._zmq_inbox = None
        
        # Wait for engine processes to terminate
        if hasattr(self, '_talker_process'):
            self._talker_process.join(timeout=5.0)
            if self._talker_process.is_alive():
                print(f"[Interface] Talker process did not terminate, forcing...")
                self._talker_process.terminate()
                self._talker_process.join(timeout=2.0)
            print(f"[Interface] Talker process terminated")
            
        if hasattr(self, '_predictor_process'):
            self._predictor_process.join(timeout=5.0)
            if self._predictor_process.is_alive():
                print(f"[Interface] Predictor process did not terminate, forcing...")
                self._predictor_process.terminate()
                self._predictor_process.join(timeout=2.0)
            print(f"[Interface] Predictor process terminated")

    def generate_custom_voice(self, text: str, language: str = "English", speaker: str = "Vivian"):
        """Sync generator. Only valid when zmq_bridge is None. For ZMQ use generate_custom_voice_async()."""
        if self.zmq_bridge is not None:
            raise RuntimeError(
                "When using ZMQ bridge, use async API: await interface.start_zmq_tasks(); "
                "async for chunk in interface.generate_custom_voice_async(...)"
            )
        input_ids, instruct_ids, speakers, languages = prepare_custom_voice_prompt(
            text=text, language=language, speaker=speaker,
            processor=self.processor, device=self.device,
        )
        talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = prepare_inputs(
            config=self.model_config,
            input_ids=input_ids, instruct_ids=instruct_ids, speakers=speakers, languages=languages,
            non_streaming_mode=True,
            text_embedding=self.text_embedding, input_embedding=self.input_embedding,
            text_projection=self.text_projection, device=self.device,
        )
        yield from self._generate_caller_driven(
            talker_input_embeds, trailing_text_hiddens, tts_pad_embed,
            str(uuid.uuid4()),
            SamplingParams(temperature=1.0, max_tokens=1),
            SamplingParams(temperature=0.9, max_tokens=17),
        )

    async def generate_custom_voice_async(
        self, text: str, language: str = "English", speaker: str = "Vivian"
    ):
        """Async generator of codebook_id chunks. Requires zmq_bridge; call await start_zmq_tasks() first."""
        if self.zmq_bridge is None:
            raise RuntimeError("generate_custom_voice_async requires zmq_bridge")

        def _prep_in_thread() -> tuple:
            """Run prep in executor so event loop can run engine_loop; lock serializes GPU prep."""
            with self._prep_lock:
                input_ids, instruct_ids, speakers, languages = prepare_custom_voice_prompt(
                    text=text, language=language, speaker=speaker,
                    processor=self.processor, device=self.device,
                )
                return prepare_inputs(
                    config=self.model_config,
                    input_ids=input_ids, instruct_ids=instruct_ids, speakers=speakers, languages=languages,
                    non_streaming_mode=True,
                    text_embedding=self.text_embedding, input_embedding=self.input_embedding,
                    text_projection=self.text_projection, device=self.device,
                )

        loop = asyncio.get_event_loop()
        talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask = await loop.run_in_executor(
            None, _prep_in_thread
        )
        async for chunk in self.generate_async(
            talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask
        ):
            yield chunk

    def generate(self, inputs_embeds: torch.Tensor, trailing_text_hiddens: torch.Tensor, tts_pad_embed: torch.Tensor, talker_attention_mask: torch.Tensor, request_id: str | None = None):
        """Sync generator. Only valid when zmq_bridge is None."""
        if self.zmq_bridge is not None:
            raise RuntimeError("When using ZMQ bridge use generate_async() after await start_zmq_tasks()")
        request_id = request_id or str(uuid.uuid4())
        talker_sampling_params = SamplingParams(temperature=1.0, max_tokens=1)
        predictor_sampling_params = SamplingParams(temperature=0.9, max_tokens=17)
        yield from self._generate_caller_driven(
            inputs_embeds, trailing_text_hiddens, tts_pad_embed,
            request_id, talker_sampling_params, predictor_sampling_params,
        )

    async def generate_async(
        self,
        inputs_embeds: torch.Tensor,
        trailing_text_hiddens: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        talker_attention_mask: torch.Tensor,
        request_id: str | None = None,
    ):
        """Async generator of codebook_id chunks. ZMQ path; step() runs in separate engine processes. Call await start_zmq_tasks() first."""
        if self.zmq_bridge is None:
            raise RuntimeError("generate_async requires zmq_bridge")
        talker_sampling_params = SamplingParams(temperature=1.0, max_tokens=1)
        predictor_sampling_params = SamplingParams(temperature=0.9, max_tokens=17)
        request_id = request_id or str(uuid.uuid4())
        request_queue: asyncio.Queue = asyncio.Queue()
        async with self._queues_lock:
            self._request_queues[request_id] = request_queue
        try:
            next_talker_embeds = inputs_embeds
            if next_talker_embeds.dim() == 2:
                next_talker_embeds = next_talker_embeds.unsqueeze(0)
            generation_step = 0
            
            # Send add_request to talker engine process via multiprocessing queue
            talker_request = EngineRequest(
                action="add_request",
                request_id=request_id,
                inputs_embeds=next_talker_embeds,
                sampling_params=talker_sampling_params,
            )
            self._talker_request_queue.put(talker_request)

            while True:
                engine_type, msg_type, payload = await request_queue.get()
                if engine_type == "talker" and msg_type == "done":
                    # Send clear_request to talker engine process
                    clear_request = EngineRequest(
                        action="clear_request",
                        request_id=request_id,
                    )
                    self._talker_request_queue.put(clear_request)
                    break
                if engine_type == "talker" and msg_type == "token":
                    token_ids = payload["token_ids"]
                    hidden_states = payload.get("hidden_states")
                    last_id = token_ids[-1]
                    if last_id == 2150:
                        # Send clear_request to talker engine process
                        clear_request = EngineRequest(
                            action="clear_request",
                            request_id=request_id,
                        )
                        self._talker_request_queue.put(clear_request)
                        break
                    last_id_hidden = self.input_embedding(torch.tensor([last_id], device=self.device)).unsqueeze(0)
                    if hidden_states is not None:
                        h = torch.from_numpy(hidden_states.copy()).to(self.device)
                        if h.dim() == 1:
                            h = h.unsqueeze(0).unsqueeze(0)
                        else:
                            h = h.unsqueeze(0).unsqueeze(0)
                        last_hidden_state = h
                    else:
                        last_hidden_state = last_id_hidden.unsqueeze(0)
                    predictor_inputs_embeds = torch.cat((last_hidden_state, last_id_hidden), dim=1)
                    
                    # Send add_request to predictor engine process via multiprocessing queue
                    predictor_request = EngineRequest(
                        action="add_request",
                        request_id=request_id,
                        inputs_embeds=predictor_inputs_embeds,
                        sampling_params=predictor_sampling_params,
                    )
                    self._predictor_request_queue.put(predictor_request)
                    
                    _, _, payload2 = await request_queue.get()
                    pred_token_ids = payload2.get("token_ids", [])
                    codebook_ids = [last_id] + pred_token_ids
                    yield codebook_ids

                    codec_hiddens = torch.cat(
                        [last_id_hidden]
                        + [self.predictor_input_embeddings[i](torch.tensor([pred_token_ids[i]], device=self.device)).unsqueeze(0) for i in range(15)],
                        dim=1,
                    )
                    next_talker_embeds = codec_hiddens.sum(1, keepdim=True)
                    if generation_step < trailing_text_hiddens.shape[1]:
                        next_talker_embeds = next_talker_embeds + trailing_text_hiddens[:, generation_step].unsqueeze(1)
                    else:
                        next_talker_embeds = next_talker_embeds + tts_pad_embed
                    generation_step += 1
                    
                    # Send next add_request to talker engine process
                    talker_request = EngineRequest(
                        action="add_request",
                        request_id=request_id,
                        inputs_embeds=next_talker_embeds,
                        sampling_params=talker_sampling_params,
                    )
                    self._talker_request_queue.put(talker_request)
        finally:
            async with self._queues_lock:
                self._request_queues.pop(request_id, None)

    def _generate_caller_driven(
        self,
        inputs_embeds: torch.Tensor,
        trailing_text_hiddens: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        request_id: str,
        talker_sampling_params: SamplingParams,
        predictor_sampling_params: SamplingParams,
    ):
        generation_step = 0
        next_talker_embeds = inputs_embeds
        if next_talker_embeds.dim() == 2:
            next_talker_embeds = next_talker_embeds.unsqueeze(0)

        while True:
            self.talker_llm.add_request([next_talker_embeds], talker_sampling_params, request_id=request_id)
            _, _, outputs_all = self.talker_llm.step_with_outputs()
            if not outputs_all:
                self.talker_llm.clear_request(request_id)
                return

            match = next((o for o in outputs_all if o[0] == request_id), None)
            if match is None:
                continue
            _, _, token_ids, hidden_states, is_finished = match
            last_id = token_ids[-1]
            if last_id == 2150:
                self.talker_llm.clear_request(request_id)
                return

            last_id_hidden = self.input_embedding(torch.tensor([last_id], device=self.device)).unsqueeze(0)
            last_hidden_state = hidden_states.unsqueeze(0).unsqueeze(0)
            predictor_inputs_embeds = torch.cat((last_hidden_state, last_id_hidden), dim=1)
            predictor_outputs = self.predictor_llm.generate(
                [predictor_inputs_embeds.unsqueeze(0)],
                predictor_sampling_params,
                use_tqdm=False,
                request_id=request_id,
            )
            pred_token_ids = predictor_outputs[0]["token_ids"]
            codebook_ids = [last_id] + pred_token_ids
            yield codebook_ids

            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [self.predictor_input_embeddings[i](torch.tensor([pred_token_ids[i]], device=self.device)).unsqueeze(0) for i in range(15)],
                dim=1,
            )
            next_talker_embeds = codec_hiddens.sum(1, keepdim=True)
            if generation_step < trailing_text_hiddens.shape[1]:
                next_talker_embeds = next_talker_embeds + trailing_text_hiddens[:, generation_step].unsqueeze(1)
            else:
                next_talker_embeds = next_talker_embeds + tts_pad_embed
            generation_step += 1


if __name__ == "__main__":
    interface = Qwen3TTSInterface(model_path="/work/weights/qwen3tts")
    print("Warm up...")
    audio_codes = list(interface.generate_custom_voice(text="Hi there this is a test.", language="English", speaker="Vivian"))

    print("Generate...")
    start = time.time()
    audio_codes = list(interface.generate_custom_voice(text="Hi there, this is tsdocode, hope you are doing well.", language="English", speaker="Vivian"))
    end = time.time()

    
    
    
    