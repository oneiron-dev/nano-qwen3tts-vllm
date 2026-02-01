# nano-qwen3tts-vllm

Qwen3-TTS with vLLM-style optimizations for fast text-to-speech generation.

## Installation

```bash
git clone https://github.com/tsdocode/nano-qwen3tts-vllm.git
uv sync
# or
pip install -e .
```

Requires Python ≥3.10, PyTorch ≥2.10, `qwen-tts`, and `transformers`.

## Usage

### Basic TTS generation

```python
from nano_qwen3tts_vllm.inferface import Qwen3TTSInterface
from qwen_tts import Qwen3TTSTokenizer
import soundfile as sf

# Initialize with path to Qwen3-TTS model
interface = Qwen3TTSInterface(
    model_path="/path/to/qwen3tts",
    enforce_eager=False,   # use CUDA graphs when False
    tensor_parallel_size=1
)

# Generate audio
audio_codes = interface.generate_custom_voice(
    text="Hello, this is a test.",
    language="English",
    speaker="Vivian"
)

# Decode to waveform and save
tokenizer = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")
wavs, sr = tokenizer.decode([{"audio_codes": audio_codes}])
sf.write("output.wav", wavs[0], sr)
```

### Run from CLI

```bash
python -m nano_qwen3tts_vllm.inferface
```

This runs the example in `inferface.py` (ensure `model_path` points to your Qwen3-TTS checkpoint).

## Options

| Parameter | Description |
|-----------|-------------|
| `model_path` | Path to Qwen3-TTS model weights |
| `enforce_eager` | Disable CUDA graphs (useful for debugging) |
| `tensor_parallel_size` | Number of GPUs for tensor parallelism (1–8) |
| `text` | Input text to synthesize |
| `language` | Language (e.g. `"English"`) |
| `speaker` | Speaker name (e.g. `"Vivian"`) |

## Note
Current only support custom voice model
