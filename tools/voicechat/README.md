# NemotronLabs VoiceChat in llama.cpp

This directory holds the work to run
[`hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF`](https://huggingface.co/hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF)
under llama.cpp.

That one file is the whole model. It is not loadable as is: the converters here
split it into the four files `llama-voicechat` actually wants. The root
[README](../../README.md) has the download-and-convert walkthrough; this file is
about how the model works and why the code looks the way it does.

## What the VoiceChat GGUF actually is

One container, five models, 1632 tensors:

| component | tensors | what it is |
|---|---|---|
| `stt_model.perception` | 640 | FastConformer encoder, 24 layers, d=1024, 8 rel-pos heads, plus mel featurizer |
| `stt_model.llm` | 337 | **NVIDIA-Nemotron-Nano-9B-v2**, unmodified: 56 blocks, `M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-` |
| `stt_model.rnnt_decoder` / `rnnt_joint` | 14 | RNN-T ASR head, 2-layer LSTM |
| `tts_model.tts_model` | 493 | 28-block transformer backbone, subword encoder, MoG head, gated audio/text fusion |
| `tts_model.audio_codec` | 142 | ConvNeXt encoder/decoder plus a 31-stage residual VQ |

Tensor names come from the NeMo checkpoint (`stt_model.llm.layers.0.mixer.in_proj.weight`),
and the file carries none of the KV metadata llama.cpp needs - no `block_count`,
no `ssm.*`, no tokenizer. The hyper-parameters are all in one JSON blob,
`voicechat.config.model`, which every converter here reads.

## Status

| stage | what | state |
|---|---|---|
| 1 | STT language model | done, generates text; function head wired for turn-taking |
| 2 | perception encoder, speech in | done, `llama-voicechat` answers questions about a wav |
| 3 | TTS backbone + codec, speech out | done, `--tts` produces a wav |
| 4 | sessions: multi-turn, tool calls | done, `--serve` holds one timeline and speaks json |

The model answers in English and only in English. Fed a question in any of ten
languages it either answers it in English or produces nothing useful; there is
no language setting to change that. See [Stage 4](#stage-4---one-timeline-many-turns)
for `--say`, which is the cheap way to make an English test clip without a mic.

## Stage 1 - the STT language model

The STT LLM is stock Nemotron-H, which llama.cpp already implements. Its layer map
matches `hybrid_override_pattern` in the reference config character for character,
and every shape agrees:

```
hidden 4480   intermediate 15680   vocab 131072
mamba: 128 heads x 80 = d_inner 10240, 8 groups, state 128, conv 4
       in_proj = 2*10240 + 2*8*128 + 128 = 22656
attn : 40 heads x 128 = 5120, 8 kv heads = 1024
```

So no C++ change is needed to run it. `convert_voicechat_to_nemotron_h.py` renames
the tensors, writes the metadata, and pulls the tokenizer from a local copy of
`nvidia/NVIDIA-Nemotron-Nano-9B-v2`:

```bash
hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2 config.json tokenizer.json tokenizer_config.json --local-dir ref_nano9b
```

```bash
python tools/voicechat/convert_voicechat_to_nemotron_h.py nemotron_voicechat_11b-Q4_0.gguf --ref-dir ref_nano9b -o nemotron_voicechat_11b-stt-llm-Q4_0.gguf
```

Quantized tensors are copied block for block - a Q4_0 input gives bit-identical
Q4_0 weights out, no requantization round trip. Only the tensors llama.cpp stores
differently are touched, and all four are F16 in the source:

| source | output | transform |
|---|---|---|
| `mixer.A_log` | `blk.N.ssm_a` | `A = -exp(A_log)`, reshaped to `{1, 128}` |
| `mixer.D` | `blk.N.ssm_d` | reshaped to `{1, 128}` |
| `mixer.conv1d.weight` | `blk.N.ssm_conv1d.weight` | squeeze the middle dim |
| `mixer.norm.weight` | `blk.N.ssm_norm.weight` | regrouped to `{d_inner/n_group, n_group}` |

Everything outside the LLM is dropped; the output is ~4.67 GiB from a 6.19 GiB input.

### Special tokens

VoiceChat retrains the base tokenizer's special tokens and records the new ones in
`stt.model.override_tokens`. They do not match `ref_nano9b/config.json`:

| | reference | VoiceChat |
|---|---|---|
| bos | 1 `<s>` | 1 `<s>` |
| eos | 12 | 2 `</s>` |
| pad | 0 | 12 `<SPECIAL_12>` |

The reference eos is VoiceChat's pad, so writing the base ids makes generation stop
on the first silent frame. The converter applies the override and logs it.

### The function head

The duplex model has a second output projection, `function_head` (same shape as
`lm_head`), trained with `use_function_head: true`. Its channel carries the
turn-taking / tool-call markers - `<SPECIAL_20>` (sotc, start of tool call),
`<SPECIAL_21>` (eotc, end of tool call), `<SPECIAL_22>` (eotr, end of tool
response), token text in between, and pad everywhere else. The NeMo inference
loop (`DuplexSTTModel._step_inference`) feeds the sampled function token back
into the next frame's input sum with weight 2.0:

```
input[t] = 1.0 * embed(text[t-1]) + 1.0 * perception(audio)[t] + 2.0 * embed(function[t-1])
```

llama.cpp rejects unknown tensors in a `nemotron_h` file, so the stage 1
converter writes the head as a side gguf, `<output>-function-head.gguf`, with
the fusion weights and token ids as KV. `llama-voicechat` picks it up by name
next to the model, asks the context for per-frame embeddings, runs the head as
a one-matmul ggml graph, and reports sotc/eotc/eotr events. Feeding the channel
back is what keeps long inputs stable: without it the model is slightly out of
distribution on every frame and eventually degenerates into repetition.

### Tokenizer caveat

The VoiceChat GGUF has no tokenizer, so it comes from the Nemotron-Nano-9B-v2 repo.
Its pre-tokenizer regex is character-for-character the one llama.cpp calls `tekken`
(see the comment above `LLAMA_VOCAB_PRE_TYPE_TEKKEN` in `src/llama-vocab.cpp`), so
that is what the converter writes. This is a match by inspection, not by a checksum
in `convert_hf_to_gguf_update.py` - worth confirming with `llama-tokenize` against
the HF tokenizer before trusting it on anything long.

## Stage 2 - speech in

```bash
python tools/voicechat/convert_voicechat_perception_to_mmproj.py nemotron_voicechat_11b-Q4_0.gguf -o mmproj-voicechat-perception-Q4_0.gguf
```

```bash
llama-voicechat -m nemotron_voicechat_11b-stt-llm-Q4_0.gguf --mmproj mmproj-voicechat-perception-Q4_0.gguf --audio input.wav
```

### The encoder

`stt_model.perception` is a NeMo `ConformerEncoder`, the same family as Parakeet,
which llama.cpp already runs as `PROJECTOR_TYPE_PARAKEET`. So this is a new
projector type, `voicechat`, whose graph in `models/voicechat.cpp` is the parakeet
graph with the four differences the config forces. It is built for a streaming
duplex model, so everything is causal:

1. `causal_downsampling: true`. The subsampling convs are NeMo `CausalConv2D`,
   which pads kernel-1 on the left and stride-1 on the right of **both** the time
   and the frequency axis. So 128 mel bins go 128 -> 65 -> 33 -> 17, not
   128 -> 64 -> 32 -> 16, and `pre_encode.out` is `Linear(256*17 = 4352, 1024)`.
   That 4352 is the check that this reading is right.
2. `conv_norm_type: layer_norm`. NeMo keeps the attribute name `batch_norm`
   whatever the norm is, so the tensors are still called `conv.batch_norm.*`, but
   they are a LayerNorm over channels and there are no running mean/var tensors.
3. `conv_context_size: causal`. The depthwise conv pads kernel-1 = 8 on the left
   and nothing on the right.
4. `att_context_style: chunked_limited` with `att_context_size: [70, 0]`. The
   chunk is `right + 1 = 1` frame, so this reduces to plain causal attention with
   a 70 frame left window. It lives entirely in the input mask.

The preprocessor is the same NeMo featurizer parakeet uses (preemphasis 0.97,
n_fft 512, 400 sample Hann window, hop 160, 128 mel, `log(x + 2^-24)`), with the
mel filterbank and the window shipped in the mmproj as tensors, except that
`normalize: "NA"` here, so the per-feature mean/variance step is skipped.

`use_bias: false`, so no linear in the encoder has a bias. The modality adapter is
an `IdentityConnector`, so the projector is a single `Linear(1024, 4480)` + bias.

Output: one 4480-wide vector per 80 ms, i.e. 12.5 Hz, matching the TTS frame rate.

### Why llama-mtmd-cli cannot drive this model

VoiceChat is a duplex speech-to-speech model, not a captioner. The LLM runs at a
fixed 12.5 Hz and consumes one embedding per frame:

```
input[t] = perception(audio)[t] + embed_tokens(text_out[t-1])
```

The audio embeddings are **added** to the token embeddings, not appended as extra
positions - `text_embedding_weight: 1.0` in the config is the weight on that sum.
`llama-mtmd-cli` inserts them as tokens instead, and the model then ignores them.
`llama-voicechat` runs the real loop: it takes the perception output from mtmd,
reads one row of `token_embd.weight` off disk per frame (llama.cpp exposes no
handle on it), sums them, and decodes one embedding at a time.

The text channel is mostly `<SPECIAL_12>`; a real token only appears on the frames
where the model speaks. On `tools/mtmd/test-2.mp3`, 219 frames (17.5 s), greedy:

```
The New York Times is not just a daily newspaper; it is a daily newspaper. ...
```

and with `--temp 0.7 --top-p 0.9`:

```
... August thirty one ... twenty eight New York Times, the Times from July ...
two thousand pages ... History of United ... The Times ... from J ...
```

so the encoder is clearly feeding real content into the LLM.

### Known gaps in stage 2

- `delay_text_channel_by: 2` and `delay_source_text_by: 15` are in the config and
  are not modelled. Plain autoregression (feed frame t's token at t+1) is assumed.
  The NeMo inference loop does the same, so this matches the reference.
- `pos_bias_u` / `pos_bias_v` are `{128, 8}`, so rank 2, so the published Q4_0
  quantized them, and the mmproj has to dequantize a 4-bit tensor. A
  requantization should add them to the skip list in
  `quantize/voicechat_q4_0_worker.cpp`; it costs 2 KiB per layer.
- The mel is zero-padded at the edges; NeMo's `torch.stft` reflects. Same as
  parakeet, small edge effect only.

## Stage 3 - speech out

```bash
python tools/voicechat/convert_voicechat_tts_to_gguf.py nemotron_voicechat_11b-Q4_0.gguf --ref-dir ref_nano9b -o voicechat-tts-Q4_0.gguf
```

```bash
llama-voicechat -m nemotron_voicechat_11b-stt-llm-Q4_0.gguf --mmproj mmproj-voicechat-perception-Q4_0.gguf --audio input.wav --tts voicechat-tts-Q4_0.gguf --tts-out out.wav
```

That writes 559 tensors, 685 MiB, with the hyper-parameters as real KV, plus a
baked character table for the subword encoder (which is why it needs --ref-dir).
The graph that consumes it lives in `voicechat-tts.cpp`, on plain ggml graphs
driven through a `ggml_backend_sched`: the text channel token of every frame
drives one backbone step, and the wav is decoded at the end of the run. The
device is `--tts-device` (or `VC_TTS_DEVICE`), defaulting to whatever `--device`
put the llm on and then to the first GPU there is. The scheduler carries a CPU
backend behind the chosen one, so any op the device does not implement falls
back instead of aborting the turn. The specification
below was read off `speech_generation.model` in the source config, confirmed
against every tensor shape, and settled against the NeMo `speechlm2` source
(branch `nemotron-labs-voicechat` of NVIDIA-NeMo/Speech, mirrored locally in
`D:\Jobs\voicechat\ref_speechlm2`).

### Parts

```
backbone        gemma3_text, 28 blocks, hidden 1152, 16 heads x 72, GQA off,
                intermediate 4608, sliding window 7500, RMSNorm q and k,
                pre+post norms on both attention and ffn
embed_subword   t5gemma encoder, 1 block, same widths. Pools the LLM's subword
                tokens into one 1152 vector per frame. Carries two per-vocab
                lookup tables: special_flags {131072} and is_continuation
                {131073}, plus a 3-row special embedding and a 2-row
                continuation embedding
fusion          gated audio/text mix: audio_proj and text_proj are 1152x1152
                with bias, a per-channel `gate` {1152}, a scalar
                `residual_scale` and a final RMSNorm
mog_head        3 gemma MLP blocks (pre_norm, mlp, post_norm) then a final norm,
                then four projections off the 1152 hidden:
                  logits {1152, 1024}   mixture weights
                  mus    {1152, 65536}  = 1024 components x low_rank 64
                  logs   {1152, 1}      one log std, floored at min_log_std -4
                  else   {1152, 512}
                mu_k = low_mat[k] @ mus(h)[k], low_mat is {1024, 512, 64}
codec           31 stage residual VQ. rvq_embs {512, 1024, 31} are the codebooks;
                prvq.mus_list.N {512, 1024} and _variance_list.N are the
                probabilistic quantizer's per stage means and variances
codec decoder   latent 512 -> waveform. Layer indices in the checkpoint:
                  0   ConvTranspose1d 512 -> 1536, k=9,  stride 9
                  1-3 ConvNeXt blocks at 1536 (dwconv k=7, LayerNorm, 1536->6144->1536)
                  4   ConvTranspose1d 1536 -> 768, k=7, stride 7
                  5-7 ConvNeXt blocks at 768
                  8   ConvTranspose1d 768 -> 384, k=7, stride 7
                  9-11 ConvNeXt blocks at 384
                  12  Conv1d 384 -> 18, k=1
                then ISTFT with n_fft 16, hop 4: 18 channels = 9 bins x (re, im).
                Total upsample 9*7*7*4 = 1764 = wav_to_token_ratio, so
                22050 / 1764 = 12.5 Hz, the same clock as the perception encoder.
```

`tools/mtmd/mtmd-audio.h` already has `mtmd_audio_streaming_istft`, which is the
right shape for the last step.

### The generation loop, settled against the NeMo source

The open questions from the first pass all have answers in
`ear_tts_model.py` / `duplex_ear_tts.py`:

1. **The MoG head predicts a residual, not the full latent.** Per frame the 31
   codes start fully masked and are revealed over 8 iterations with the schedule
   `ceil((1 - (i/8)^3)^(1/3) * 31)`, which unmasks 1, 1, 3, 4 and then 22 stages
   (the first three iterations do nothing). Each iteration runs
   `x = mlp_stack(embed_code(depthsum(code)) + h)` for the conditional and
   unconditional hidden state, combines them as `x_c + 0.2 * (x_c - x_u)`,
   samples a mixture component (top-p 0.95, gumbel-max), and forms
   `z = mu * exp(logs) + proj_else(x) + exp(logs) * noise * 0.001` with
   `mu = low_mat[k] @ proj_mus(x)[k]`. `z` is the summed embedding of the still
   masked stages, so it is residual-encoded greedily into the codebooks starting
   at the current stage.
2. **Discrete codes, not the continuous latent, are the state.** The next
   frame's input is `embed_code(depthsum(code))` and the codec decodes the
   summed codebook embeddings (`rvq_embs` is exactly the codec's
   `prvq.mus_list`, stacked). The probabilistic-VQ variances are unused at
   inference.
3. **Guidance is two rows through the whole backbone.** The unconditional row
   shares the code embeddings and replaces the text conditioning with
   `null_emb`. Both rows live in one kv cache, `guidance_scale` 0.2.
4. **The audio prompt needs no codec encoder.** The warmup sequence is one text
   eos frame plus 37 prompt frames; every pre-bos position's code embedding is
   REPLACED by the pre-baked `audio_prompt_latents.Aria` (already projected), so
   the only real code embedding in the warmup is the codec silence frame right
   before bos. `audio_prompt_projection_W` is only needed to bake latents for
   new speakers.

5. **The backbone input is NOT scaled by sqrt(1152), the subword encoder input
   is.** Both models get `inputs_embeds` because the reference deletes their
   embedding modules (`find_and_delete_module`), so whether the gemma
   `sqrt(hidden)` normalizer applies depends on where HF keeps it. In gemma3 it
   lives inside `Gemma3TextScaledWordEmbedding`, so it never runs and the
   backbone sees the fusion output at RMS 1; in the t5gemma encoder it is still
   applied in the forward pass, so the character embeddings do get scaled.
   Scaling the backbone input makes the residual stream ~34x larger than every
   block's contribution, which silently bypasses all 28 blocks: the MoG head
   then predicts a near-mean latent (norm ~13 instead of ~71 for silence) and
   the output is uniform low-level mush with no dynamic range. See
   [debugging](#debugging) for how to catch this.

Other details that the checkpoint alone did not say: every RMSNorm in the TTS
stack is the gemma `x * (1 + w)` kind (the converter folds the +1 in); the
backbone is HF `gemma3_text` defaults (rope 1e6 global / 1e4 local, 5+1 sliding
pattern, query_pre_attn_scalar 256); the
subword encoder is HF `t5gemma` defaults (rope 1e4, attention softcap 50,
bidirectional) over the token's characters, mean-pooled and projected, with
per-vocab continuation/special flag embeddings on top; on a text eos the
previous frame's codes are forced to the codec silence frame; and codes >= 1024
(control ids) are replaced by the silence frame before the codec.

Deviation from the reference: frame 0, which NeMo leaves as zero codes because
its loop starts at t=1, is emitted as the codec silence frame here.

The codec decoder runs in short causal chunks with discarded overlap so the
compute buffer stays bounded, and the 16 point ISTFT (periodic hann, hop 4,
envelope normalized, value range constrained) is done on the CPU.

Three other things stay on the CPU whatever `--tts-device` says, and they are
what a turn now spends most of its time on: the 31 stage residual RVQ search
(1024 candidates per stage, per frame), the nucleus filter and gumbel pick over
the 1024 mixture components, and the low rank mixture mean. They read weight
rows straight out of the gguf, which is why the host copy of the weights is
kept alongside the device mirror.

### Debugging

Two environment variables, both no-ops when unset:

- `VC_TTS_DUMP_CODES=<file>` writes the raw frame codes: `int32 n_frames`,
  `int32 n_quantizers`, then `n_frames * n_quantizers` codes, then one `int32`
  text token per frame.
- `VC_TTS_DEBUG=1` logs the MoG sampler per unmask iteration: the chosen
  component, `logs`, and the norms of the head input, the low-rank mean, the
  residual mean and `z`.

The two numbers worth watching are `|z|` and the latent norm of the emitted
codes, because they have known targets and localize a fault immediately:

| what | expected |
| --- | --- |
| `|z|` on the first unmask iteration (cnt=0) of a silent frame | ~70.8, and the codes come out equal to `codec.silence_tokens` for the first 6+ stages |
| latent norm of a speech frame | ~35-45 |
| decoded wav | true digital silence between utterances, spectral peak at 100-300 Hz |

Anything upstream of the codec that is off-distribution shows up as a small
`|z|` (the head falls back to its mean), a near-uniform mixture (top-p 0.95
keeping ~900 of 1024 components), and a decoded wav with ~13 dB of dynamic
range whose loudest band is 0-100 Hz rumble.

The codec decoder itself can be checked independently: encoding digital silence
through the codec encoder in the source gguf reproduces `codec.silence_tokens`
exactly after two frames of causal warmup, and the latent norm settles at 70.77.

Do not guess at a stage 3 fault. [`debug/`](debug/README.md) is an independent
numpy port of the whole stage 3 path, written from the NeMo source rather than
from `voicechat-tts.cpp`, reading the same gguf the runtime does. Any
disagreement between the two is a logic difference, not a weight difference, so
it bisects a bad wav down to one stage in a few minutes. It needs numpy and
scipy, no torch. The `sqrt(1152)` scaling bug above was found this way.

## Stage 4 - one timeline, many turns

VoiceChat has no text input channel and no place to replay a chat history into:
there is a single 12.5 Hz timeline, and the state left behind by the model's
answer is the state the next question starts from. So a conversation is not a
prompt that grows, it is that timeline continuing, and `--serve` is what keeps it
alive:

```bash
llama-voicechat -m nemotron_voicechat_11b-stt-llm-Q4_0.gguf   --mmproj mmproj-voicechat-perception-Q4_0.gguf   --tts voicechat-tts-Q4_0.gguf --serve
```

One json object per line in, one event per line out (stdout is json only in this
mode; every log line stays on stderr):

```
{"cmd":"system","text":"You are ..."}          before the first turn only
{"cmd":"turn","audio":"in.wav","out":"out.wav"}
{"cmd":"tool_response","text":"{...}"}         while a call is pending
{"cmd":"reset"} {"cmd":"ping"} {"cmd":"quit"}
```

```
ready system_start system turn_start assistant_text_delta function_delta
tool_call_start tool_call tool_response tool_response_end audio turn_end
progress warning error bye
```

Each turn is the user's audio, then decoding past it on a zero audio embedding
until the text channel falls quiet, then draining the tts until the speech
settles; `out` gets exactly that turn's frames, decoded with a few frames of
causal run-up so it does not start on a codec transient. The same code runs the
one-shot mode, where `--audio` may now be given several times - each clip is one
turn of the same conversation.

`--session-seconds` (default 180) caps the timeline; it sizes the tts kv cache
(~126 KiB per frame) and has to fit in the llm context.

### Where a turn ends

A turn does not end when the audio does. Two separate tails come after it, and
cutting either one short is heard immediately:

- the **text** channel keeps being decoded past the last audio frame, on a zero
  audio embedding, until 10 consecutive pad frames say the model has stopped
  talking. `--extra-decoding-seconds` (default 50) caps that, and `VC_QUIET=N`
  overrides the pad streak.
- the **speech** channel trails the text channel by tens of frames. The words
  are queued into the speech generator, not spoken as they arrive, so stopping
  the tts when the text goes quiet truncates the last few words. The tts is
  drained on pad frames instead, until the latent settles back to silence.

Both used to fail in the same direction. An eos emitted while words were still
queued wiped them; the queue is now flushed before eos goes out.

### The system prompt

VoiceChat has no text input channel, so there is nowhere to put a prompt except
the one channel the model has. `--system` (or `--system-file`) writes the prompt
into the perception channel, one token per frame, ahead of the audio - exactly
where the model's own text would go. It behaves like a real system prompt, and
it is not free: the model runs at 12.5 Hz, so every prompt token is 80 ms of
decode. A 270 token tool list is about 75 s before the first turn is heard.

### Barge-in, and why it has to be suppressed

The model is duplex, so nothing stops it from opening its own turn while the
user is still speaking. Left alone it does, usually about a second in, and the
damage does not stay in that turn:

1. it answers the first second of the question, not the question
2. it then repeats itself, because the audio it is still hearing does not match
   what it already said
3. every later turn on that timeline degenerates further, since the state the
   next turn starts from is the state this one left behind

Two environment variables fix it, and they are meant to be used together:

| var | what |
| --- | --- |
| `VC_NO_BARGE=1` | the agent may not open a turn (emit bos) while the user's audio is still playing |
| `VC_FORCE_BOS=1` | with the guard on, open the turn on the first frame past the audio |

`VC_NO_BARGE` alone only defers the problem: the model has already decided to
talk and will emit bos late and unpredictably. `VC_FORCE_BOS` is the reference's
`force_bos_positions` - it pins the turn boundary to the end of the clip, which
is what a push-to-talk driver wants anyway.

### Environment variables

All of them are no-ops when unset.

| var | what |
| --- | --- |
| `VC_NO_BARGE` | see above |
| `VC_FORCE_BOS` | see above |
| `VC_QUIET=N` | pad streak that ends a turn, default 10 frames |
| `VC_DUMP=1` | one diagnostic line per frame |
| `VC_PACE_DUMP=1` | one line per speech frame, for tts pacing |
| `VC_TTS_DEBUG=1` | MoG sampler, per unmask iteration |
| `VC_TTS_DUMP_CODES=<file>` | raw frame codes plus the text token per frame |

### Tool calls in one pass

The reference does function calling offline in two passes: pass 1 finds the call,
then the whole sequence is re-run with the call and the API response spliced into
the function channel at the right steps (`_expand_for_function_calling`). On a
live timeline that collapses to one pass, because the splice can just happen as
the frames go by:

1. the function head emits `sotc`; from here the user's audio is **frozen** - the
   reference inserts the call region as extra positions with a zero audio
   embedding, which is what "the model stops listening while it consults the
   tool" means when there is only one clock
2. the call text accumulates until `eotc`
3. the driver is asked for a result (`tool_call` event, then a `tool_response`
   command; `--tool-response` in one-shot mode) and it is spliced into the
   function channel one token per frame, overriding the head
4. one more frozen frame for the model's own `eotr`, then the audio resumes

The tts is not stepped at all over that region, so the gap is simply not in the
wav - the same thing the reference does afterwards with `splice_fc_audio_gap`.

The tool list itself belongs in the system prompt, in the base model's own
format (`<AVAILABLE_TOOLS>` / `<TOOLCALL>` / `<TOOL_RESPONSE>`, see
`ref_nano9b/tokenizer_config.json`), which the driver has to render itself.
It is not free: two tools come to ~270 tokens, i.e. ~270 frames, i.e. ~75 s of
decode once per session.

### Making test clips without a mic

The tts does not care where its text channel comes from, so it can read a
sentence out in the agent's voice. This model is English-only, and this is the
cheap way to get an English question to feed it:

```bash
llama-voicechat -m ... --mmproj ... --tts voicechat-tts-Q4_0.gguf   --say "Hello! What is the weather in Hanoi right now?" --tts-out q.wav
```

`--say-pace` is how many frames each token gets, and the default of 1 is the
right one. Spreading the text out looks like it should help - real speech is
slower than one token per 80 ms - but it does the opposite: the speech generator
wants its text channel dense, and every pad frame inserted between two tokens is
a frame where the backbone is conditioned on nothing. Raising it past 2 makes
the words slur rather than slow down.

### One that was hiding in plain sight

`common_params_parser_init` sets `params.sampling.temp = 0.2` for every
`LLAMA_EXAMPLE_MTMD` tool, and it does that *during* the parse - so the `temp = 0`
this tool set before calling it was thrown away, and the text channel was being
sampled at 0.2 with a random seed. Two identical runs would not match. The
default is now applied after the parse, so it is greedy unless `--temp` says
otherwise, which is what the reference does (`nemotron_voicechat.py`:
`temperature=0.0, top_p=1.0`).

## Note on the source file

The Q4_0 file this targets keeps every rank-1 tensor at F16 on purpose. Quantizing
`A_log`, `D` and `dt_bias` to 4 bits costs up to a 1.93x error on the Mamba2 decay
rate `A = -exp(A_log)`, because a rank-1 element's error is never averaged away by
a dot product. Keeping all of them F16 costs about 2 MiB on a 6 GiB file. If you
requantize this model yourself, keep that rule, and add `pos_bias_u` / `pos_bias_v`
to it.

## Files

| file | what |
|---|---|
| `vc_gguf.py` | minimal seek/read GGUF reader shared by the converters |
| `convert_voicechat_to_nemotron_h.py` | stage 1, STT LLM -> `nemotron_h` + function head side gguf |
| `convert_voicechat_perception_to_mmproj.py` | stage 2, perception -> mtmd audio projector |
| `convert_voicechat_tts_to_gguf.py` | stage 3, TTS + codec -> standalone gguf |
| `voicechat-cli.cpp` | `llama-voicechat`, the 12.5 Hz duplex loop + function channel |
| `voicechat-tts.cpp` | stage 3 runtime: backbone, MoG sampler, RVQ, codec, istft |
| `debug/` | numpy reference port of stage 3, with known-good target numbers |
| `../mtmd/models/voicechat.cpp` | the causal FastConformer graph |
