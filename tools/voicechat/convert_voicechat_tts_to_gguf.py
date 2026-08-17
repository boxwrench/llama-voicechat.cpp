#!/usr/bin/env python3
"""
Extract the speech generator from a NemotronLabs VoiceChat GGUF.

This is stage 3: text (and LLM state) in, 22050 Hz audio out. It has four parts,
and this converter isolates all of them into one file with real metadata so the
C++ side does not have to carry the 6 GiB container around.

  1. backbone         28 block gemma3_text, hidden 1152, 16 heads of 72, GQA off,
                      q/k RMSNorm, pre and post norms on both attn and ffn
  2. embed_subword    1 block t5gemma encoder that pools the LLM's subword
                      tokens, plus bos/eos and subword-continuation embeddings
  3. mog_head         3 gemma MLP blocks, then a mixture of 1024 Gaussians over
                      the 512-wide codec latent. Each mean is low rank:
                      mu_k = low_mat[k] @ proj_mus(h)[k], low_mat is {1024,512,64}
  4. audio_codec      31 stage residual VQ plus a ConvNeXt decoder that upsamples
                      9x7x7 and ends in a 16 point ISTFT with hop 4
                      (9*7*7*4 = 1764 samples per frame = 12.5 Hz at 22050)

The codec ENCODER is dropped: generation only needs the decoder, and the one
built-in speaker prompt ships as a precomputed latent (audio_prompt_latents).

Conv weights need no reshaping. Torch Conv1d is (out, in, k) and ConvTranspose1d
is (in, out, k), which land in GGUF as {k, in, out} and {k, out, in}, exactly
what ggml_conv_1d and ggml_conv_transpose_1d want. The k=1 "pointwise" convs are
matmuls, so those lose their trailing 1.

The subword encoder is character-aware: NeMo's CharAwareSubwordEncoder splits
every LLM token into single characters, embeds those, runs the 1 block t5gemma
encoder over them and mean-pools. The character vocabulary is not stored in the
checkpoint; NeMo rebuilds it from the tokenizer (every single-character token,
densely reindexed in id order). With --ref-dir this converter bakes the same
mapping into the output as two i32 tensors, `tts.subword.char_ids` (flattened
sequences) and `tts.subword.char_offsets` (per token id start offsets).

Every RMSNorm in the TTS stack is the gemma kind, `x * (1 + w)`, and the +1 is
folded into the weights here, exactly as convert_hf_to_gguf.py does for gemma.
The codec's LayerNorms are standard and are copied as-is.

Usage
-----
    python tools/voicechat/convert_voicechat_tts_to_gguf.py \
        nemotron_voicechat_11b-Q4_0.gguf \
        --ref-dir ref_nano9b \
        -o voicechat-tts-Q4_0.gguf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gguf-py"))
sys.path.insert(0, str(Path(__file__).parent))
import gguf  # noqa: E402

from vc_gguf import GGML_BLOCK, GGUFSource  # noqa: E402

logger = logging.getLogger("voicechat-tts")

SRC = "tts_model."
TTS = "tts_model.tts_model."
CODEC = "tts_model.audio_codec."

# ConvNeXt stages: 3 blocks each at 1536, 768, 384 for the decoder.
# Layer indices in the checkpoint: resample convs at 0, 4, 8, head at 12.
DEC_RESAMPLE = (0, 4, 8)
DEC_HEAD = 12
DEC_BLOCKS = (1, 2, 3, 5, 6, 7, 9, 10, 11)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--ref-dir", type=Path, required=True,
                    help="directory with tokenizer.json from nvidia/NVIDIA-Nemotron-Nano-9B-v2, "
                         "used to bake the character-aware subword table")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    src = GGUFSource(args.input)
    logger.info("source: %s (%d tensors)", args.input, len(src.tensors))

    gen = json.loads(src.kv["voicechat.config.model"])["speech_generation"]
    cfg = gen["model"]
    tts_cfg = cfg["tts_config"]
    bb = tts_cfg["backbone_config"]
    mog = tts_cfg["mog_head_config"]
    codec = cfg["codec_config"]

    n_layer = bb["num_hidden_layers"]
    n_embd = bb["hidden_size"]

    w = gguf.GGUFWriter(path=None, arch="voicechat_tts")
    w.add_type(gguf.GGUFType.MODEL)
    w.add_name("NemotronLabs VoiceChat 11B speech generator")
    w.add_description(
        "TTS backbone, subword encoder, mixture-of-Gaussians latent head and 31 stage "
        "RVQ codec decoder from the NVIDIA NemotronLabs VoiceChat 11B GGUF."
    )

    # backbone
    w.add_block_count(n_layer)
    w.add_embedding_length(n_embd)
    w.add_feed_forward_length(bb["intermediate_size"])
    w.add_head_count(bb["num_attention_heads"])
    w.add_head_count_kv(bb["num_key_value_heads"])
    w.add_key_length(bb["head_dim"])
    w.add_value_length(bb["head_dim"])
    w.add_sliding_window(bb["sliding_window"])
    w.add_layer_norm_rms_eps(1e-6)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_0)

    # generator and codec hyper-parameters, under a private prefix
    kv = {
        "voicechat.tts.latent_size":           tts_cfg["latent_size"],
        "voicechat.tts.codebook_size":         tts_cfg["codebook_size"],
        "voicechat.tts.num_quantizers":        tts_cfg["num_quantizers"],
        "voicechat.tts.num_delay_speech_tokens": tts_cfg["num_delay_speech_tokens"],
        "voicechat.tts.mog.num_layers":        mog["num_layers"],
        "voicechat.tts.mog.low_rank":          mog["low_rank"],
        "voicechat.tts.mog.num_predictions":   mog["num_predictions"],
        "voicechat.codec.base_hidden_size":    codec["base_hidden_size"],
        "voicechat.codec.num_blocks":          codec["num_blocks"],
        "voicechat.codec.kernel_size":         codec["kernel_size"],
        "voicechat.codec.n_fft":               codec["n_fft"],
        "voicechat.codec.hop_length":          codec["hop_length"],
        "voicechat.codec.sample_rate":         gen["data"]["target_sample_rate"],
        "voicechat.codec.wav_to_token_ratio":  codec["wav_to_token_ratio"],
    }
    for k, v in kv.items():
        w.add_uint32(k, int(v))
    w.add_array("voicechat.codec.rates", [int(r) for r in codec["rates"]])
    w.add_array("voicechat.codec.channel_mult", [int(c) for c in codec["channel_mult"]])
    w.add_float32("voicechat.tts.mog.min_log_std", float(mog["min_log_std"]))
    w.add_float32("voicechat.tts.mog.eps", float(mog["eps"]))
    w.add_float32("voicechat.tts.guidance_scale", float(cfg["inference_guidance_scale"]))
    w.add_float32("voicechat.tts.noise_scale", float(cfg["inference_noise_scale"]))
    w.add_string("voicechat.tts.speaker", cfg.get("inference_speaker_name") or "Aria")

    # generation loop, from DuplexEARTTS._get_generation_config and the tts config
    w.add_uint32("voicechat.tts.num_iter", 8)
    w.add_float32("voicechat.tts.top_p", float(cfg.get("inference_top_p_or_k", 0.8)))
    w.add_float32("voicechat.tts.exponent", float(tts_cfg["exponent"]))
    w.add_bool("voicechat.tts.force_silence_on_eos", bool(cfg.get("inference_force_speech_silence_on_eos", True)))

    # text channel token ids, resolved below from the tokenizer; speech side is
    # codebook_size + 0/1/2 = pad/eos/bos (see DuplexEARTTS.speech_*_id)
    # backbone hyper-parameters that are HF gemma3_text defaults, not in the config
    w.add_rope_freq_base(1_000_000.0)
    w.add_float32("voicechat.tts.rope_local_theta", 10_000.0)
    w.add_uint32("voicechat.tts.swa_pattern", 6)
    w.add_uint32("voicechat.tts.query_pre_attn_scalar", 256)
    # subword encoder hyper-parameters that are HF t5gemma defaults
    w.add_float32("voicechat.tts.cas.rope_theta", 10_000.0)
    w.add_float32("voicechat.tts.cas.attn_softcap", 50.0)
    w.add_uint32("voicechat.tts.cas.query_pre_attn_scalar", 256)

    tok = json.loads((args.ref_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = dict(tok["model"]["vocab"])
    for a in tok.get("added_tokens", []):
        vocab[a["content"]] = a["id"]

    w.add_uint32("voicechat.tts.text_bos_id", vocab[cfg.get("bos_token") or "<s>"])
    w.add_uint32("voicechat.tts.text_eos_id", vocab[cfg.get("eos_token") or "</s>"])
    w.add_uint32("voicechat.tts.text_pad_id", vocab[cfg.get("pad_token") or "<SPECIAL_12>"])

    n_copied = 0
    n_conv = 0

    def copy_raw(src_name: str, dst_name: str) -> None:
        nonlocal n_copied
        t = src.take(src_name)
        if len(t["dims"]) != 2:
            raise SystemExit(f"{src_name}: copy_raw needs 2-D, got {t['dims']}")
        block, size = GGML_BLOCK[t["ty"]]
        row_elems = t["dims"][0]
        n_rows = t["elements"] // row_elems
        data = np.frombuffer(src.raw(t), dtype=np.uint8).reshape(n_rows, row_elems // block * size)
        w.add_tensor(dst_name, data, raw_shape=data.shape,
                     raw_dtype=getattr(gguf.GGMLQuantizationType, t["ty"]))
        n_copied += 1
        logger.debug("copy    %-62s -> %-34s %s %s", src_name, dst_name, t["ty"], t["dims"])

    def put(dst_name: str, arr: np.ndarray, src_name: str, dtype=np.float32) -> None:
        nonlocal n_conv
        arr = np.ascontiguousarray(arr, dtype=dtype)
        w.add_tensor(dst_name, arr)
        n_conv += 1
        logger.debug("convert %-62s -> %-34s %s %s", src_name, dst_name,
                     np.dtype(dtype).name, arr.shape)

    def put_src(dst_name: str, src_name: str, dtype=np.float32, squeeze=None, plus_one=False) -> None:
        a = src.f32(src_name)
        if squeeze is not None:
            a = a.squeeze(squeeze)
        if plus_one:
            a = a + 1.0
        put(dst_name, a, src_name, dtype)

    def put_i32(dst_name: str, src_name: str) -> None:
        a = src.i64(src_name).astype(np.int32)
        put(dst_name, a, src_name, dtype=np.int32)

    # -------------------------------------------------------------- backbone
    for il in range(n_layer):
        p = f"{TTS}backbone.layers.{il}."
        b = f"blk.{il}."
        copy_raw(p + "self_attn.q_proj.weight", b + "attn_q.weight")
        copy_raw(p + "self_attn.k_proj.weight", b + "attn_k.weight")
        copy_raw(p + "self_attn.v_proj.weight", b + "attn_v.weight")
        copy_raw(p + "self_attn.o_proj.weight", b + "attn_output.weight")
        put_src(b + "attn_q_norm.weight", p + "self_attn.q_norm.weight", plus_one=True)
        put_src(b + "attn_k_norm.weight", p + "self_attn.k_norm.weight", plus_one=True)
        copy_raw(p + "mlp.gate_proj.weight", b + "ffn_gate.weight")
        copy_raw(p + "mlp.up_proj.weight",   b + "ffn_up.weight")
        copy_raw(p + "mlp.down_proj.weight", b + "ffn_down.weight")
        put_src(b + "attn_norm.weight",      p + "input_layernorm.weight", plus_one=True)
        put_src(b + "post_attention_norm.weight", p + "post_attention_layernorm.weight", plus_one=True)
        put_src(b + "ffn_norm.weight",       p + "pre_feedforward_layernorm.weight", plus_one=True)
        put_src(b + "post_ffw_norm.weight",  p + "post_feedforward_layernorm.weight", plus_one=True)

    put_src("output_norm.weight", TTS + "backbone.norm.weight", plus_one=True)

    # frame level inputs into the backbone
    copy_raw(TTS + "embed_code.weight", "tts.embed_code.weight")
    put_src("tts.bos_emb",  TTS + "bos_emb")
    put_src("tts.null_emb", TTS + "null_emb")
    copy_raw(TTS + "audio_prompt_projection_W", "tts.audio_prompt_proj.weight")

    # ------------------------------------------------------- subword encoder
    sp = TTS + "embed_subword."
    sb = "tts.subword."
    copy_raw(sp + "embed_tokens.weight",   sb + "embed_tokens.weight")
    copy_raw(sp + "proj_embedding.weight", sb + "proj.weight")
    copy_raw(sp + "bos_eos_emb.special_emb.weight", sb + "special_emb.weight")
    copy_raw(sp + "subword_flag_emb.cont_emb.weight", sb + "cont_emb.weight")
    # per-vocab lookup tables: which LLM token is special, which continues a word
    put_i32(sb + "special_flags",  sp + "bos_eos_emb.special_flags")
    put_i32(sb + "is_continuation", sp + "subword_flag_emb.is_continuation")

    # character table (NeMo build_vocabs): the character vocabulary is every
    # single-character token, densely reindexed in token id order
    singles = {s: i for s, i in vocab.items() if len(s) == 1}
    char_vocab = {s: n for n, s in enumerate(sorted(singles, key=lambda s: singles[s]))}
    n_char_rows = src.take(sp + "embed_tokens.weight")["dims"][1]
    if len(char_vocab) + 1 != n_char_rows:
        raise SystemExit(f"char vocab has {len(char_vocab)} entries, "
                         f"embed_tokens has {n_char_rows} rows (want chars + 1 pad)")

    n_ids = max(vocab.values()) + 1
    id_to_str = {i: s for s, i in vocab.items()}
    offsets = np.zeros(n_ids + 1, dtype=np.int32)
    flat: list[int] = []
    for i in range(n_ids):
        flat.extend(char_vocab[c] for c in id_to_str.get(i, "") if c in char_vocab)
        offsets[i + 1] = len(flat)
    put(sb + "char_ids", np.asarray(flat, dtype=np.int32), "<tokenizer>", dtype=np.int32)
    put(sb + "char_offsets", offsets, "<tokenizer>", dtype=np.int32)
    logger.info("char table: %d chars, %d ids, %d char refs", len(char_vocab), n_ids, len(flat))

    ep = sp + "backbone.encoder."
    eb = sb + "blk.0."
    copy_raw(ep + "layers.0.self_attn.q_proj.weight", eb + "attn_q.weight")
    copy_raw(ep + "layers.0.self_attn.k_proj.weight", eb + "attn_k.weight")
    copy_raw(ep + "layers.0.self_attn.v_proj.weight", eb + "attn_v.weight")
    copy_raw(ep + "layers.0.self_attn.o_proj.weight", eb + "attn_output.weight")
    copy_raw(ep + "layers.0.mlp.gate_proj.weight", eb + "ffn_gate.weight")
    copy_raw(ep + "layers.0.mlp.up_proj.weight",   eb + "ffn_up.weight")
    copy_raw(ep + "layers.0.mlp.down_proj.weight", eb + "ffn_down.weight")
    put_src(eb + "attn_norm.weight",           ep + "layers.0.pre_self_attn_layernorm.weight", plus_one=True)
    put_src(eb + "post_attention_norm.weight", ep + "layers.0.post_self_attn_layernorm.weight", plus_one=True)
    put_src(eb + "ffn_norm.weight",            ep + "layers.0.pre_feedforward_layernorm.weight", plus_one=True)
    put_src(eb + "post_ffw_norm.weight",       ep + "layers.0.post_feedforward_layernorm.weight", plus_one=True)
    put_src(sb + "output_norm.weight",         ep + "norm.weight", plus_one=True)

    # ------------------------------------------------ gated audio/text fusion
    gp = TTS + "gated_fusion_audio_text."
    gb = "tts.fusion."
    copy_raw(gp + "audio_proj.weight", gb + "audio_proj.weight")
    put_src(gb + "audio_proj.bias",    gp + "audio_proj.bias")
    copy_raw(gp + "text_proj.weight",  gb + "text_proj.weight")
    put_src(gb + "text_proj.bias",     gp + "text_proj.bias")
    put_src(gb + "gate",               gp + "gate")
    put_src(gb + "norm.weight",        gp + "final_norm.weight", plus_one=True)
    put_src(gb + "residual_scale",     gp + "residual_scale")

    # ------------------------------------------------------------- mog head
    for i in range(mog["num_layers"]):
        p = f"{TTS}mog_head.mlp_stack.{i}."
        b = f"tts.mog.blk.{i}."
        copy_raw(p + "mlp.gate_proj.weight", b + "ffn_gate.weight")
        copy_raw(p + "mlp.up_proj.weight",   b + "ffn_up.weight")
        copy_raw(p + "mlp.down_proj.weight", b + "ffn_down.weight")
        put_src(b + "pre_norm.weight",  p + "pre_norm.weight", plus_one=True)
        put_src(b + "post_norm.weight", p + "post_norm.weight", plus_one=True)
    put_src("tts.mog.norm.weight", f"{TTS}mog_head.mlp_stack.{mog['num_layers']}.weight", plus_one=True)

    copy_raw(TTS + "mog_head.proj_logits.weight", "tts.mog.logits.weight")
    copy_raw(TTS + "mog_head.proj_mus.weight",    "tts.mog.mus.weight")
    copy_raw(TTS + "mog_head.proj_logs.weight",   "tts.mog.logs.weight")
    copy_raw(TTS + "mog_head.proj_else.weight",   "tts.mog.else.weight")
    # {low_rank, latent, num_predictions}; too big for a raw 2-D copy, so F16
    put_src("tts.mog.low_mat", TTS + "mog_head.low_mat", dtype=np.float16)

    # ------------------------------------------------------------------- rvq
    put_src("codec.rvq_embs", TTS + "rvq_embs", dtype=np.float16)
    for i in range(tts_cfg["num_quantizers"]):
        copy_raw(f"{CODEC}prvq.mus_list.{i}", f"codec.rvq.{i}.mus")
        put_src(f"codec.rvq.{i}.variance", f"{CODEC}prvq._variance_list.{i}.variance")

    # --------------------------------------------------------- codec decoder
    for i in DEC_RESAMPLE:
        # ConvTranspose1d, GGUF already holds {k, out, in}
        put_src(f"codec.dec.up.{i}.weight", f"{CODEC}decoder.layers.{i}.weight", dtype=np.float16)
    put_src(f"codec.dec.head.weight", f"{CODEC}decoder.layers.{DEC_HEAD}.weight", dtype=np.float16)

    for n, i in enumerate(DEC_BLOCKS):
        p = f"{CODEC}decoder.layers.{i}."
        b = f"codec.dec.blk.{n}."
        put_src(b + "dwconv.weight", p + "dwconv.weight", dtype=np.float16)
        put_src(b + "dwconv.bias",   p + "dwconv.bias")
        put_src(b + "norm.weight",   p + "norm.weight")
        put_src(b + "norm.bias",     p + "norm.bias")
        put_src(b + "pw1.weight",    p + "pwconv1.weight", dtype=np.float16, squeeze=-1)
        put_src(b + "pw1.bias",      p + "pwconv1.bias")
        put_src(b + "pw2.weight",    p + "pwconv2.weight", dtype=np.float16, squeeze=-1)
        put_src(b + "pw2.bias",      p + "pwconv2.bias")

    # ------------------------------------------------------- misc lookups
    put_i32("tts.control_codes",       SRC + "_control_codes")
    put_i32("codec.silence_tokens",    SRC + "codec_silence_tokens")
    speaker = cfg.get("inference_speaker_name") or "Aria"
    put_src("tts.audio_prompt_latents", f"{SRC}audio_prompt_latents.{speaker}")

    logger.info("tensors: %d copied verbatim, %d converted", n_copied, n_conv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w.open_output_file(args.output)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    logger.info("wrote %s (%.1f MiB)", args.output, os.path.getsize(args.output) / 1024 ** 2)


if __name__ == "__main__":
    main()
