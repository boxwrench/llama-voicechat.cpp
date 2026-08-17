# Numpy reference harness

An independent, weight-identical numpy port of the stage 3 path, written from
the NeMo source in `ref_speechlm2`, not from `voicechat-tts.cpp`. It exists to
localize faults: every stage below has a target value that is checkable without
listening to anything.

| file | what it ports |
| --- | --- |
| `gg.py` | minimal gguf reader with F32/F16/Q4_0/I32 dequant |
| `codec.py` | `Latent2Wav` + RVQ dequant + `spec_to_wav` (the codec decoder) |
| `encoder.py` | `Wav2Latent` + `PreTrainedProbabilisticVQ.encode` (ground-truth codes from a real wav; reads the **source** gguf, which is the only file carrying the codec encoder) |
| `tts_np.py` | subword encoder, gated fusion, gemma3 backbone with kv cache, MoG head |

Needs numpy and scipy. No torch.

## Known-good targets

```python
import numpy as np, codec, encoder
c, e = codec.Codec(), encoder.Encoder()

# 1. the codec decoder turns the silence codes into digital silence
w = c.decode_codes(np.tile(c.silence.astype(np.int64), (40, 1)))
assert w.std() < 1e-5

# 2. the codec encoder round-trips: encode(zeros) == codec.silence_tokens
lat = e.encode_latents(np.zeros(1764 * 8))
codes, _ = e.quantize(lat, c.rvq)
assert (codes[3] == c.silence).all() and abs(np.linalg.norm(lat[3]) - 70.77) < 0.1
```

Real 22050 Hz speech round-trips at matching RMS with waveform correlation
~0.6 and latent norms ~43.

## Bisecting the runtime against it

`tts_np.TTS()` reads the same `voicechat-tts-*.gguf` the C++ runtime does, so
any disagreement is a logic difference rather than a weight difference. Drive it
with the text tokens from `VC_TTS_DUMP_CODES` and compare against
`VC_TTS_DEBUG=1`:

```python
m = tts_np.TTS()
m.warmup()
masked = np.full(31, m.n_codebook, np.int64)
_, _, hc, hu = m.step(text_pad_token, masked)
xg, logits, logs, mu_res = m.mog_head(hc, hu, masked)
# a silent frame: |mu_res| ~= 70.8 and cos(mu_res, silence latent) ~= 1.0
```

`scale_backbone_in` / `scale_cas_in` toggle the gemma `sqrt(hidden)` input
normalizer on each of the two transformers. The shipped values (`False` for the
backbone, `True` for the subword encoder) are the ones that reproduce the
silence latent exactly; this harness is how that was settled.
