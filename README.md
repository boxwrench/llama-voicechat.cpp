# llama-voicechat.cpp

<div align="center">

<b>NVIDIA NemotronLabs VoiceChat 11B, running on llama.cpp</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</div>

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that runs
[NVIDIA NemotronLabs VoiceChat 11B](https://huggingface.co/hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF)
end to end: speech in, speech out, on one duplex timeline, with tool calls.

Everything upstream llama.cpp does still works. The new part is one tool,
`llama-voicechat`, plus the converters that turn the published VoiceChat GGUF
into files llama.cpp can load.

## What this model is, in one paragraph

VoiceChat is not a captioner and not an ASR model with a TTS bolted on. It is a
single duplex speech-to-speech model that runs at a fixed 12.5 Hz: on every
80 ms frame the perception encoder's output is **added** to the embedding of the
token the model wrote on the previous frame, and out come three channels - the
text it is saying, a function channel carrying turn-taking and tool-call
markers, and the acoustic codes its own voice is made of. There is no chat
template and no place to replay a history into. A conversation is that timeline
continuing. `llama-mtmd-cli` appears to load the model and then quietly ignores
the audio, because it inserts audio as extra token positions instead of summing
it into the frame. That is the whole reason this fork exists.

The model answers in English only.

## Quick start

### 1. Get the binaries

Download from [Releases](https://github.com/sansamour/llama-voicechat.cpp/releases):

| file | what |
| --- | --- |
| `llama-voicechat-bin-win-cpu-x64.zip` | Windows x64, CPU. Needs an AVX2 CPU. |
| `llama-voicechat-bin-win-cuda-12.8-x64.zip` | Windows x64, CUDA 12. Self-contained. |
| `cudart-llama-bin-win-cuda-12.8-x64.zip` | CUDA 12.8 runtime dlls, unzip on top of the CUDA build |

The CUDA build ships device code for sm_86, sm_89 and sm_120 (RTX 30xx, 40xx and
50xx) and PTX from sm_50 up, so anything Maxwell or newer runs, with a JIT pause
on the first launch if it is not one of those three. Take the CUDA zip **and**
the cudart zip and unzip both into the same folder.

Or [build it yourself](#building-from-source).

### 2. Get the model

The published VoiceChat GGUF is one 6.19 GiB container holding five models, and
llama.cpp cannot load it as is - it carries NeMo tensor names and none of the KV
metadata llama.cpp needs. Either take the converted files, or convert them
yourself.

**Converted, ready to run:**

```bash
hf download hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-llamacpp-GGUF --local-dir voicechat
```

**Or convert them yourself** - see [From the source GGUF to the four files](#from-the-source-gguf-to-the-four-files).

Either way you end up with four files, which must sit in the same folder:

| file | size | what |
| --- | --- | --- |
| `nemotron_voicechat_11b-stt-llm-Q4_0.gguf` | 4.67 GiB | the language model, a stock `nemotron_h` |
| `nemotron_voicechat_11b-stt-llm-Q4_0-function-head.gguf` | 315 MiB | the turn-taking / tool-call head |
| `mmproj-voicechat-perception-Q4_0.gguf` | 435 MiB | the causal FastConformer speech encoder |
| `voicechat-tts-Q4_0.gguf` | 686 MiB | the speech generator and audio codec |

The function head is found by name next to the model, so do not rename it.

### 3. Run it

Ask a question from a wav and get a spoken answer back:

```bash
llama-voicechat -m voicechat/nemotron_voicechat_11b-stt-llm-Q4_0.gguf --mmproj voicechat/mmproj-voicechat-perception-Q4_0.gguf --tts voicechat/voicechat-tts-Q4_0.gguf --audio question.wav --tts-out answer.wav
```

Set `VC_NO_BARGE=1` and `VC_FORCE_BOS=1` for anything push-to-talk. Without them
the model barges in about a second into the clip, answers the first second of
the question, and every later turn on that timeline degenerates. This is the
single most important flag pair in the tool.

No microphone? The speech generator does not care where its text comes from, so
it can read a sentence out in the agent's own voice, which is how to make an
English test clip:

```bash
llama-voicechat -m voicechat/nemotron_voicechat_11b-stt-llm-Q4_0.gguf --mmproj voicechat/mmproj-voicechat-perception-Q4_0.gguf --tts voicechat/voicechat-tts-Q4_0.gguf --say "What is the weather in Hanoi right now?" --tts-out question.wav
```

Multi-turn, tool calls, one json object per line in and one event per line out:

```bash
llama-voicechat -m voicechat/nemotron_voicechat_11b-stt-llm-Q4_0.gguf --mmproj voicechat/mmproj-voicechat-perception-Q4_0.gguf --tts voicechat/voicechat-tts-Q4_0.gguf --serve
```

The full flag list, the event protocol, the environment variables and how each
stage works are in [tools/voicechat/README.md](tools/voicechat/README.md).

## From the source GGUF to the four files

Three converters, run against the same input file. They are pure numpy plus the
`gguf-py` already in this repo - no torch, no transformers.

```bash
pip install numpy
```

### 1. The source model

```bash
hf download hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF nemotron_voicechat_11b-Q4_0.gguf --local-dir .
```

### 2. The tokenizer

VoiceChat ships no tokenizer, so it comes from the base model it was trained
from. Only three small files are needed, not the weights:

```bash
hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2 config.json tokenizer.json tokenizer_config.json --local-dir ref_nano9b
```

### 3. Convert

```bash
python tools/voicechat/convert_voicechat_to_nemotron_h.py nemotron_voicechat_11b-Q4_0.gguf --ref-dir ref_nano9b -o nemotron_voicechat_11b-stt-llm-Q4_0.gguf
```

```bash
python tools/voicechat/convert_voicechat_perception_to_mmproj.py nemotron_voicechat_11b-Q4_0.gguf -o mmproj-voicechat-perception-Q4_0.gguf
```

```bash
python tools/voicechat/convert_voicechat_tts_to_gguf.py nemotron_voicechat_11b-Q4_0.gguf --ref-dir ref_nano9b -o voicechat-tts-Q4_0.gguf
```

The first command writes two files: the language model, and
`nemotron_voicechat_11b-stt-llm-Q4_0-function-head.gguf` beside it. The function
head is a second output projection that llama.cpp would reject inside a
`nemotron_h` file, so it is carried as a side gguf and loaded by name.

Nothing is requantized. A Q4_0 tensor in the source is copied out block for
block and stays bit-identical; only the handful of tensors llama.cpp stores in a
different layout are touched, and those are all F16 to begin with.

### On requantizing the source yourself

If you make your own quantization of the original NeMo checkpoint instead of
using the published Q4_0, keep every rank-1 tensor at F16: `A_log`, `D`,
`dt_bias`, and also `pos_bias_u` / `pos_bias_v` in the encoder. A rank-1
element's quantization error is never averaged away by a dot product, and on
`A = -exp(A_log)` that comes to up to a 1.93x error on the Mamba2 decay rate.
Keeping all of them F16 costs about 2 MiB on a 6 GiB file.

## Building from source

Needs CMake, Ninja and a C++17 compiler. On Windows, MSVC.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
```

```bash
cmake --build build --target llama-voicechat -j
```

For CUDA add `-DGGML_CUDA=ON`. On a CUDA 12 toolkit also add
`-DGGML_CUDA_CUB_3DOT2=ON`, which fetches CCCL 3.2; CUDA 13 bundles it already.
Blackwell (RTX 50xx) needs toolkit 12.8 or newer.

The released CUDA binaries are built by
[`.github/workflows/release-voicechat.yml`](.github/workflows/release-voicechat.yml),
which is the only workflow this fork runs. Every upstream llama.cpp workflow is
disabled here.

## What was changed against upstream

| path | what |
| --- | --- |
| `tools/voicechat/` | the whole tool: converters, the 12.5 Hz duplex loop, the speech generator, a numpy reference harness |
| `tools/mtmd/models/voicechat.cpp` | a new mtmd projector type: the causal FastConformer encoder |
| `.github/workflows/release-voicechat.yml` | the release build |

Everything else is upstream. The language model itself needed no C++ change -
VoiceChat's LLM is stock Nemotron-H, which llama.cpp already implements.

## Upstream llama.cpp

This fork tracks [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and
keeps its full history. All of the upstream documentation applies:

- [How to build](docs/build.md)
- [Supported backends](docs/build.md)
- [llama-server](tools/server/README.md)
- [llama-cli](tools/cli/README.md)
- [Models](docs/models.md)

Issues with llama.cpp itself belong upstream, not here.

## License

MIT, same as upstream llama.cpp. The VoiceChat model weights are NVIDIA's and
carry their own license.

## Acknowledgements

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml)
- [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/Speech), branch `nemotron-labs-voicechat`, which is the reference this implementation was settled against
