# NemotronLabs VoiceChat in llama.cpp

This directory holds the work to run
[`hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF`](https://huggingface.co/hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF)
under llama.cpp.

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
| 1 | STT language model | done, generates text |
| 2 | perception encoder, speech in | done, `llama-voicechat` answers questions about a wav |
| 3 | TTS backbone + codec, speech out | weights extracted, graph not written |

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

- **Turn taking is not wired up.** The real model has a second output head,
  `function_head` (4480 x 131072), whose channel carries `sotc`/`eotc`/`eotr`/`call`
  and decides when the agent starts and stops speaking. `llama-voicechat` samples
  the text channel on every frame instead, so long inputs degenerate into
  repetition once the model would have stopped. The stage 1 converter drops
  `function_head`; running it needs a second output projection.
- `delay_text_channel_by: 2` and `delay_source_text_by: 15` are in the config and
  are not modelled. Plain autoregression (feed frame t's token at t+1) is assumed.
- `pos_bias_u` / `pos_bias_v` are `{128, 8}`, so rank 2, so the published Q4_0
  quantized them, and the mmproj has to dequantize a 4-bit tensor. A
  requantization should add them to the skip list in
  `quantize/voicechat_q4_0_worker.cpp`; it costs 2 KiB per layer.
- The mel is zero-padded at the edges; NeMo's `torch.stft` reflects. Same as
  parakeet, small edge effect only.

## Stage 3 - speech out

```bash
python tools/voicechat/convert_voicechat_tts_to_gguf.py nemotron_voicechat_11b-Q4_0.gguf -o voicechat-tts-Q4_0.gguf
```

That writes 557 tensors, 681 MiB, with the hyper-parameters as real KV. **The ggml
graph that consumes it is not written.** What follows is the specification for it,
read off `speech_generation.model` in the source config and confirmed against every
tensor shape.

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

### What is not settled

The weights and the module shapes are certain. The generation loop is not, and
these are the open questions, in the order they matter:

1. How the MoG head's four outputs combine into a latent. `proj_else` (512 wide)
   is most likely a component-independent offset added to the sampled mean, but
   nothing in the config says so.
2. Whether the sampled continuous latent is fed to the codec decoder directly or
   is quantized through the 31 stage RVQ first. Both paths exist in the file:
   `embed_code` maps 512 -> 1152 for the next frame's input, and `rvq_embs` holds
   the codebooks. `num_delay_speech_tokens: 2` suggests the usual delay pattern
   over quantizer stages.
3. Classifier-free guidance. `inference_guidance_enabled: true`,
   `inference_guidance_scale: 0.2`, `p_uncond: 0.1` and a `null_emb` tensor mean
   two forward passes per frame, conditional and unconditional.
4. The audio prompt. `audio_prompt_latents.Aria` is `{1152, 37}` - 37 frames is
   the 3 s `audio_prompt_duration` at 12.5 Hz - and it is already in backbone
   space, so `use_audio_prompt_frozen_projection: true` means it goes through
   `audio_prompt_projection_W` and is prefixed to the sequence. Other speakers
   need the codec **encoder**, which this converter drops.

Answering 1 and 2 needs the NeMo `speechlm2` source, not more of this checkpoint.

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
| `convert_voicechat_to_nemotron_h.py` | stage 1, STT LLM -> `nemotron_h` |
| `convert_voicechat_perception_to_mmproj.py` | stage 2, perception -> mtmd audio projector |
| `convert_voicechat_tts_to_gguf.py` | stage 3, TTS + codec -> standalone gguf |
| `voicechat-cli.cpp` | `llama-voicechat`, the 12.5 Hz duplex loop |
| `../mtmd/models/voicechat.cpp` | the causal FastConformer graph |
