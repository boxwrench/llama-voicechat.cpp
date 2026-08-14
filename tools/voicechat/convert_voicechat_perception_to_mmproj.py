#!/usr/bin/env python3
"""
Extract the perception encoder from a NemotronLabs VoiceChat GGUF and write it
as an mtmd audio projector (mmproj).

Why this exists
---------------
Stage 1 (convert_voicechat_to_nemotron_h.py) pulled the STT language model out
of the container. This is stage 2: the thing that turns a waveform into the
embeddings that language model expects.

`stt_model.perception` is a NeMo `ConformerEncoder` - a FastConformer, the same
family as Parakeet, which llama.cpp already runs as PROJECTOR_TYPE_PARAKEET. So
this converter targets a near-identical mmproj layout and the C++ side reuses
the parakeet graph shape. Four things differ, and they are why the projector
type is `voicechat` and not `parakeet`:

  1. causal_downsampling: the subsampling convs pad (2 left, 1 right) on BOTH
     the time and the frequency axis, so 128 mel bins become 17, not 16, and
     pre_encode.out is Linear(256*17=4352, 1024).
  2. conv_norm_type=layer_norm: the module still calls itself `batch_norm`
     (NeMo reuses the attribute name), but it is a LayerNorm over channels, so
     there are no running mean/var tensors.
  3. conv_context_size=causal: the depthwise conv pads kernel-1 on the left only.
  4. att_context_style=chunked_limited with att_context_size=[70, 0]: chunk size
     is 1, so this is plain causal attention with a 70 frame left window.

Also: normalize="NA" in the preprocessor config, so there is no per-feature mean
and variance normalization of the mel spectrogram, and use_bias=false, so no
linear in the encoder carries a bias.

Usage
-----
    python tools/voicechat/convert_voicechat_perception_to_mmproj.py \
        nemotron_voicechat_11b-Q4_0.gguf \
        -o mmproj-voicechat-perception-Q4_0.gguf

Quantized 2-D weights are copied block for block. Everything the graph needs as
F32 (norms, biases, the depthwise conv, the two attention position biases) is
converted; the conv2d kernels stay F16 because ggml_conv_2d wants them that way.

Note on pos_bias_u / pos_bias_v
-------------------------------
Those are {128, 8} and rank 2, so the published Q4_0 quantized them. They have
to be dequantized here for ggml_add, which means their 4 bit error is baked in.
It is small (they enter a 128 term dot product) but a requantization should keep
them F16; see quantize/voicechat_q4_0_worker.cpp.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gguf-py"))
sys.path.insert(0, str(Path(__file__).parent))
import gguf  # noqa: E402

from vc_gguf import GGML_BLOCK, GGUFSource  # noqa: E402

logger = logging.getLogger("voicechat-mmproj")

SRC = "stt_model.perception."

# read off the NeMo config embedded in the source file
# (voicechat.config.model -> stt.model.perception)
N_LAYER = 24
N_EMBD = 1024
N_HEAD = 8
N_FF = 4096
N_MEL = 128
SUBSAMPLING_FACTOR = 8
CONV_KERNEL_SIZE = 9
ATT_LEFT_CONTEXT = 70
PROJ_DIM = 4480
LAYER_NORM_EPS = 1e-5

# indices of the real convs inside pre_encode.conv; the gaps are ReLU
PRE_ENCODE_CONV_IDX = (0, 2, 3, 5, 6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="nemotron_voicechat_11b-*.gguf")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    src = GGUFSource(args.input)
    logger.info("source: %s (%d tensors)", args.input, len(src.tensors))

    w = gguf.GGUFWriter(path=None, arch="clip")

    w.add_type(gguf.GGUFType.MMPROJ)
    w.add_name("NemotronLabs VoiceChat 11B perception")
    w.add_description(
        "FastConformer perception encoder from the NVIDIA NemotronLabs VoiceChat 11B GGUF. "
        "24 layers, d_model 1024, causal, projected to the 4480-wide STT LLM embedding space."
    )

    w.add_clip_has_audio_encoder(True)
    w.add_clip_projector_type("voicechat")

    w.add_audio_embedding_length(N_EMBD)
    w.add_audio_feed_forward_length(N_FF)
    w.add_audio_block_count(N_LAYER)
    w.add_audio_head_count(N_HEAD)
    w.add_audio_projection_dim(PROJ_DIM)
    w.add_audio_attention_layernorm_eps(LAYER_NORM_EPS)
    w.add_audio_num_mel_bins(N_MEL)
    w.add_audio_subsampling_factor(SUBSAMPLING_FACTOR)
    w.add_audio_conv_kernel_size(CONV_KERNEL_SIZE)
    w.add_audio_window_size(ATT_LEFT_CONTEXT)

    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_0)

    n_copied = 0
    n_conv = 0

    def copy_raw(src_name: str, dst_name: str) -> None:
        """Move a 2-D tensor across untouched, quantization blocks and all."""
        nonlocal n_copied
        t = src.take(src_name)
        if len(t["dims"]) != 2:
            raise SystemExit(f"{src_name}: copy_raw needs a 2-D tensor, got {t['dims']}")
        block, size = GGML_BLOCK[t["ty"]]

        row_elems = t["dims"][0]
        n_rows = t["elements"] // row_elems
        data = np.frombuffer(src.raw(t), dtype=np.uint8).reshape(n_rows, row_elems // block * size)

        w.add_tensor(dst_name, data,
                     raw_shape=data.shape,
                     raw_dtype=getattr(gguf.GGMLQuantizationType, t["ty"]))
        n_copied += 1
        logger.debug("copy    %-58s -> %-30s %s %s", src_name, dst_name, t["ty"], t["dims"])

    def put(dst_name: str, arr: np.ndarray, src_name: str, dtype=np.float32) -> None:
        nonlocal n_conv
        arr = np.ascontiguousarray(arr, dtype=dtype)
        w.add_tensor(dst_name, arr)
        n_conv += 1
        logger.debug("convert %-58s -> %-30s %s %s", src_name, dst_name,
                     np.dtype(dtype).name, arr.shape)

    # ------------------------------------------------------------ featurizer
    # fb is {1, 128, 257} in numpy order; clip reads it flat as [mel][fft_bin]
    fb = src.f32(SRC + "preprocessor.featurizer.fb").reshape(N_MEL, -1)
    put("a.mel_filters", fb, SRC + "preprocessor.featurizer.fb")
    put("a.window", src.f32(SRC + "preprocessor.featurizer.window"),
        SRC + "preprocessor.featurizer.window")

    # ------------------------------------------------------- conv subsampling
    for i in PRE_ENCODE_CONV_IDX:
        p = f"{SRC}encoder.pre_encode.conv.{i}."
        # ggml_conv_2d / ggml_conv_2d_dw_direct want an F16 kernel, shape kept as is
        put(f"a.conv1d.{i}.weight", src.f32(p + "weight"), p + "weight", dtype=np.float16)
        # the bias is added to a {freq, time, channel} tensor, so it must broadcast
        # on ne2: numpy (C, 1, 1) -> ne {1, 1, C}
        bias = src.f32(p + "bias").reshape(-1, 1, 1)
        put(f"a.conv1d.{i}.bias", bias, p + "bias")

    copy_raw(SRC + "encoder.pre_encode.out.weight", "a.pre_encode.out.weight")
    put("a.pre_encode.out.bias", src.f32(SRC + "encoder.pre_encode.out.bias"),
        SRC + "encoder.pre_encode.out.bias")

    # ------------------------------------------------------------- projector
    copy_raw(SRC + "proj.weight", "mm.a.proj.weight")
    put("mm.a.proj.bias", src.f32(SRC + "proj.bias"), SRC + "proj.bias")

    # ---------------------------------------------------------------- layers
    for il in range(N_LAYER):
        p = f"{SRC}encoder.layers.{il}."
        b = f"a.blk.{il}."

        # self attention (no bias anywhere: use_bias=false)
        copy_raw(p + "self_attn.linear_q.weight", b + "attn_q.weight")
        copy_raw(p + "self_attn.linear_k.weight", b + "attn_k.weight")
        copy_raw(p + "self_attn.linear_v.weight", b + "attn_v.weight")
        copy_raw(p + "self_attn.linear_out.weight", b + "attn_out.weight")
        copy_raw(p + "self_attn.linear_pos.weight", b + "linear_pos.weight")

        # {d_head, n_head}, dequantized because ggml_add needs a float tensor
        put(b + "pos_bias_u", src.f32(p + "self_attn.pos_bias_u"), p + "self_attn.pos_bias_u")
        put(b + "pos_bias_v", src.f32(p + "self_attn.pos_bias_v"), p + "self_attn.pos_bias_v")

        # the four layer norms
        for s, d in (("norm_self_att", "ln1"),
                     ("norm_out", "ln2"),
                     ("norm_feed_forward1", "ffn_norm"),
                     ("norm_feed_forward2", "ffn_norm_1"),
                     ("norm_conv", "norm_conv")):
            put(b + d + ".weight", src.f32(p + s + ".weight"), p + s + ".weight")
            put(b + d + ".bias", src.f32(p + s + ".bias"), p + s + ".bias")

        # the two macaron feed forwards
        copy_raw(p + "feed_forward1.linear1.weight", b + "ffn_up.weight")
        copy_raw(p + "feed_forward1.linear2.weight", b + "ffn_down.weight")
        copy_raw(p + "feed_forward2.linear1.weight", b + "ffn_up_1.weight")
        copy_raw(p + "feed_forward2.linear2.weight", b + "ffn_down_1.weight")

        # convolution module: pointwise convs are k=1, so they are matmuls
        pw1 = src.f32(p + "conv.pointwise_conv1.weight").squeeze(-1)
        put(b + "conv_pw1.weight", pw1, p + "conv.pointwise_conv1.weight", dtype=np.float16)
        pw2 = src.f32(p + "conv.pointwise_conv2.weight").squeeze(-1)
        put(b + "conv_pw2.weight", pw2, p + "conv.pointwise_conv2.weight", dtype=np.float16)

        # depthwise conv runs through ggml_ssm_conv, which wants {kernel, channels} F32
        dw = src.f32(p + "conv.depthwise_conv.weight").squeeze(1)
        put(b + "conv_dw.weight", dw, p + "conv.depthwise_conv.weight")

        # named batch_norm in the checkpoint, but conv_norm_type=layer_norm
        put(b + "conv_norm.weight", src.f32(p + "conv.batch_norm.weight"),
            p + "conv.batch_norm.weight")
        put(b + "conv_norm.bias", src.f32(p + "conv.batch_norm.bias"),
            p + "conv.batch_norm.bias")

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
