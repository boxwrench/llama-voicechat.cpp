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
and the file carries none of the KV metadata llama.cpp needs — no `block_count`,
no `ssm.*`, no tokenizer.

## Stage 1 — the STT language model (done)

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
hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    config.json tokenizer.json tokenizer_config.json --local-dir ref_nano9b

python tools/voicechat/convert_voicechat_to_nemotron_h.py \
    nemotron_voicechat_11b-Q4_0.gguf \
    --ref-dir ref_nano9b \
    -o nemotron_voicechat_11b-stt-llm-Q4_0.gguf
```

Quantized tensors are copied block for block — a Q4_0 input gives bit-identical
Q4_0 weights out, no requantization round trip. Only the tensors llama.cpp stores
differently are touched, and all four are F16 in the source:

| source | output | transform |
|---|---|---|
| `mixer.A_log` | `blk.N.ssm_a` | `A = -exp(A_log)`, reshaped to `{1, 128}` |
| `mixer.D` | `blk.N.ssm_d` | reshaped to `{1, 128}` |
| `mixer.conv1d.weight` | `blk.N.ssm_conv1d.weight` | squeeze the middle dim |
| `mixer.norm.weight` | `blk.N.ssm_norm.weight` | regrouped to `{d_inner/n_group, n_group}` |

Everything outside the LLM is dropped; the output is ~4.67 GiB from a 6.19 GiB input.

### Tokenizer caveat

The VoiceChat GGUF has no tokenizer, so it comes from the Nemotron-Nano-9B-v2 repo.
Its pre-tokenizer regex is character-for-character the one llama.cpp calls `tekken`
(see the comment above `LLAMA_VOCAB_PRE_TYPE_TEKKEN` in `src/llama-vocab.cpp`), so
that is what the converter writes. This is a match by inspection, not by a checksum
in `convert_hf_to_gguf_update.py` — worth confirming with `llama-tokenize` against
the HF tokenizer before trusting it on anything long.

## Stage 2 — audio in (not started)

The perception encoder has to run before the LLM and feed embeddings into it. In
llama.cpp terms that is an `mtmd` audio projector, the same shape of thing as the
Whisper/Ultravox encoders in `tools/mtmd`, not a new text architecture. The
FastConformer pieces that have no direct equivalent yet: depthwise conv modules
with batch norm, relative-position attention with `pos_bias_u`/`pos_bias_v`, and
the 5-stage conv subsampling front end.

## Stage 3 — audio out (not started)

TTS backbone plus the 31-stage residual VQ codec. `tools/tts` is the closest
existing pattern, but the MoG head and the ConvNeXt codec decoder are new work.

## Note on the source file

The Q4_0 file this targets keeps every rank-1 tensor at F16 on purpose. Quantizing
`A_log`, `D` and `dt_bias` to 4 bits costs up to a 1.93x error on the Mamba2 decay
rate `A = -exp(A_log)`, because a rank-1 element's error is never averaged away by
a dot product. Keeping all of them F16 costs about 2 MiB on a 6 GiB file. If you
requantize this model yourself, keep that rule.
