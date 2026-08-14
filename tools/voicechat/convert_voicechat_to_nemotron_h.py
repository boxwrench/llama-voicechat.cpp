#!/usr/bin/env python3
"""
Extract the STT language model from a NemotronLabs VoiceChat GGUF and rewrite it
as a stock llama.cpp `nemotron_h` model.

Why this exists
---------------
`nemotron_voicechat_11b-*.gguf` is a container for five models: a FastConformer
perception encoder, the STT language model, an RNN-T head, a TTS backbone and an
audio codec. Its tensor names come straight from the NeMo checkpoint
(`stt_model.llm.layers.0.mixer.in_proj.weight`) and it carries none of the KV
metadata llama.cpp needs.

The STT language model itself is unmodified NVIDIA-Nemotron-Nano-9B-v2, which
llama.cpp already implements as `nemotron_h`. Its `hybrid_override_pattern`
matches the layer map of the VoiceChat file layer for layer. So the LLM half can
be run today, without touching llama.cpp, by renaming the tensors and writing
proper metadata.

Quantized tensors are copied block for block. Nothing is dequantized and
requantized, so a Q4_0 input yields bit-identical Q4_0 weights on the output.
The four Mamba2 tensors that llama.cpp stores in a different form (`A_log` -> A,
conv1d squeeze, ssm_norm regroup, and the A/D unsqueeze) are F16 in the source
and are converted through F32, exactly as `conversion/mamba.py` does.

The tokenizer is not in the VoiceChat file at all, so it is read from a local
copy of the nvidia/NVIDIA-Nemotron-Nano-9B-v2 repo:

    hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
        config.json tokenizer.json tokenizer_config.json \
        --local-dir ref_nano9b

Usage
-----
    python tools/voicechat/convert_voicechat_to_nemotron_h.py \
        nemotron_voicechat_11b-Q4_0.gguf \
        --ref-dir ref_nano9b \
        -o nemotron_voicechat_11b-stt-llm-Q4_0.gguf

What is dropped
---------------
Everything that is not the STT language model: perception encoder, RNN-T decoder
and joint, function head, TTS backbone and audio codec. Those need a real
`nemotron_voicechat` architecture in llama.cpp; this script is stage one.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gguf-py"))
import gguf  # noqa: E402

logger = logging.getLogger("voicechat-convert")


# ============================================================================
# Minimal GGUF reader
#
# gguf.GGUFReader would also work, but this file is 6 GiB and only a third of it
# is needed; a plain seek/read keeps the peak RSS at one tensor.
# ============================================================================

GGML_TYPE_NAME = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0", 27: "I64"}

# elements per block, bytes per block
GGML_BLOCK = {
    "F32": (1, 4),
    "F16": (1, 2),
    "I64": (1, 8),
    "Q4_0": (32, 18),
    "Q8_0": (32, 34),
}

_SCALAR = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
    5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8),
    12: ("<d", 8),
}


class GGUFSource:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "rb")

        if self._raw(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")

        self.version = self._u32()
        n_tensors = self._u64()
        n_kv = self._u64()

        self.kv: dict[str, Any] = {}
        for _ in range(n_kv):
            key = self._string()
            self.kv[key] = self._value(self._u32())

        self.tensors: dict[str, dict[str, Any]] = {}
        for _ in range(n_tensors):
            name = self._string()
            n_dims = self._u32()
            if n_dims > 4:
                raise SystemExit(f"{name}: ndim={n_dims}, header is desynced")
            dims = [self._u64() for _ in range(n_dims)]  # GGUF order: dims[0] is the row width
            ty = GGML_TYPE_NAME.get(self._u32())
            off = self._u64()

            n = 1
            for d in dims:
                n *= d

            self.tensors[name] = dict(name=name, dims=dims, ty=ty, off=off, elements=n)

        align = self.kv.get("general.alignment", 32)
        self.data_start = (self.f.tell() + align - 1) // align * align

    # -- primitives ----------------------------------------------------------

    def _raw(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"want {n} bytes, got {len(b)}")
        return b

    def _u32(self) -> int:
        return struct.unpack("<I", self._raw(4))[0]

    def _u64(self) -> int:
        return struct.unpack("<Q", self._raw(8))[0]

    def _string(self) -> str:
        return self._raw(self._u64()).decode("utf-8", "replace")

    def _value(self, t: int) -> Any:
        if t in _SCALAR:
            fmt, n = _SCALAR[t]
            return struct.unpack(fmt, self._raw(n))[0]
        if t == 8:
            return self._string()
        if t == 9:
            et = self._u32()
            n = self._u64()
            if et == 8:
                return [self._string() for _ in range(n)]
            fmt, sz = _SCALAR[et]
            return list(struct.unpack("<" + fmt[1] * n, self._raw(sz * n)))
        raise ValueError(f"unknown metadata value type {t}")

    # -- tensor access -------------------------------------------------------

    def nbytes(self, t: dict[str, Any]) -> int:
        if t["ty"] not in GGML_BLOCK:
            raise SystemExit(f"{t['name']}: unhandled dtype {t['ty']}")
        block, size = GGML_BLOCK[t["ty"]]
        return t["elements"] // block * size

    def raw(self, t: dict[str, Any]) -> bytes:
        self.f.seek(self.data_start + t["off"])
        return self._raw(self.nbytes(t))

    def f32(self, t: dict[str, Any]) -> np.ndarray:
        """Read an F16/F32 tensor as F32 in numpy order (reverse of GGUF dims)."""
        if t["ty"] == "F16":
            a = np.frombuffer(self.raw(t), dtype=np.float16).astype(np.float32)
        elif t["ty"] == "F32":
            a = np.frombuffer(self.raw(t), dtype=np.float32)
        else:
            raise SystemExit(f"{t['name']}: expected F16/F32, got {t['ty']}")
        return a.reshape(tuple(reversed(t["dims"])))


# ============================================================================
# Vocab
# ============================================================================

# The VoiceChat GGUF carries no tokenizer. Nemotron-Nano-9B-v2 uses a byte-level
# BPE whose pre-tokenizer regex is character-for-character the one llama.cpp
# calls "tekken" (see the comment above LLAMA_VOCAB_PRE_TYPE_TEKKEN in
# src/llama-vocab.cpp). Verify with tools/tokenize against the HF tokenizer
# before trusting long-context output.
TOKENIZER_PRE = "tekken"


def load_vocab(ref_dir: Path) -> dict[str, Any]:
    tok = json.loads((ref_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tok_cfg = json.loads((ref_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    cfg = json.loads((ref_dir / "config.json").read_text(encoding="utf-8"))

    model = tok["model"]
    if model["type"] != "BPE":
        raise SystemExit(f"expected a BPE tokenizer, got {model['type']}")

    vocab_size = cfg["vocab_size"]
    reverse = {i: t for t, i in model["vocab"].items()}
    added = {a["id"]: a for a in tok.get("added_tokens", [])}

    tokens: list[str] = []
    toktypes: list[int] = []

    for i in range(vocab_size):
        if i in added:
            tokens.append(added[i]["content"])
            toktypes.append(
                gguf.TokenType.CONTROL if added[i]["special"] else gguf.TokenType.USER_DEFINED
            )
        elif i in reverse:
            tokens.append(reverse[i])
            toktypes.append(gguf.TokenType.NORMAL)
        else:
            # holes in the id space still need an entry
            tokens.append(f"[PAD{i}]")
            toktypes.append(gguf.TokenType.UNUSED)

    merges: list[str] = []
    for m in model.get("merges", []):
        merges.append(m if isinstance(m, str) else " ".join(m))

    return dict(
        tokens=tokens,
        toktypes=toktypes,
        merges=merges,
        bos_id=cfg.get("bos_token_id"),
        eos_id=cfg.get("eos_token_id"),
        pad_id=cfg.get("pad_token_id"),
        add_bos=bool(tok_cfg.get("add_bos_token", False)),
        add_eos=bool(tok_cfg.get("add_eos_token", False)),
        chat_template=tok_cfg.get("chat_template"),
        cfg=cfg,
    )


# ============================================================================
# Tensor mapping
# ============================================================================

SRC_LLM = "stt_model.llm."


def classify_layers(src: GGUFSource, n_layer: int) -> tuple[list[int], list[int], list[int]]:
    """Return (mamba, attn, mlp) layer indices, read off the tensors themselves."""
    mamba, attn, mlp = [], [], []

    for i in range(n_layer):
        p = f"{SRC_LLM}layers.{i}.mixer."
        if p + "A_log" in src.tensors:
            mamba.append(i)
        elif p + "q_proj.weight" in src.tensors:
            attn.append(i)
        elif p + "up_proj.weight" in src.tensors:
            mlp.append(i)
        else:
            raise SystemExit(f"layer {i}: cannot tell what mixer this is")

    return mamba, attn, mlp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="VoiceChat GGUF (F16 or quantized)")
    ap.add_argument("--ref-dir", type=Path, required=True,
                    help="directory with config.json + tokenizer.json + tokenizer_config.json "
                         "from nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    src = GGUFSource(args.input)
    logger.info("source: %s", args.input)
    logger.info("  arch=%s tensors=%d", src.kv.get("general.architecture"), len(src.tensors))

    voc = load_vocab(args.ref_dir)
    cfg = voc["cfg"]

    # ---------------------------------------------------------------- hparams

    n_layer = cfg["num_hidden_layers"]
    n_embd = cfg["hidden_size"]
    n_ff = cfg["intermediate_size"]
    n_head = cfg["num_attention_heads"]
    n_head_kv = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]

    d_state = cfg["ssm_state_size"]
    d_conv = cfg["conv_kernel"]
    n_group = cfg["mamba_num_groups"]
    n_ssm_head = cfg["mamba_num_heads"]
    d_inner = n_ssm_head * cfg["mamba_head_dim"]

    mamba, attn, mlp = classify_layers(src, n_layer)
    pattern = "".join("M" if i in mamba else "*" if i in attn else "-" for i in range(n_layer))

    ref_pattern = cfg.get("hybrid_override_pattern")
    if ref_pattern and ref_pattern != pattern:
        raise SystemExit(
            "layer map does not match the reference config:\n"
            f"  file : {pattern}\n"
            f"  ref  : {ref_pattern}\n"
            "This GGUF is not the LLM the reference config describes."
        )

    logger.info("layer map: %s", pattern)
    logger.info("  mamba=%d attn=%d mlp=%d", len(mamba), len(attn), len(mlp))

    # sanity: in_proj width must agree with the ssm hparams
    d_in_proj = 2 * d_inner + 2 * n_group * d_state + n_ssm_head
    got = src.tensors[f"{SRC_LLM}layers.{mamba[0]}.mixer.in_proj.weight"]["dims"][1]
    if got != d_in_proj:
        raise SystemExit(f"in_proj width {got} != expected {d_in_proj} from the ssm hparams")

    # ---------------------------------------------------------------- writer

    w = gguf.GGUFWriter(path=None, arch="nemotron_h")

    w.add_type(gguf.GGUFType.MODEL)
    w.add_name("NemotronLabs VoiceChat 11B STT LLM")
    w.add_description(
        "STT language model extracted from the NVIDIA NemotronLabs VoiceChat 11B GGUF. "
        "Perception, RNN-T, TTS and audio codec are not included."
    )

    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_vocab_size(cfg["vocab_size"])

    # A layer is recurrent iff both n_head_kv and n_ff are 0 for it, so those two
    # are per-layer arrays. head_count stays a scalar: llama.cpp derives
    # n_embd_head_k from n_head() (layer 0, a mamba block here) before reading
    # attention.key_length, and a 0 there makes every attn_q come out as {n_embd, 0}.
    w.add_feed_forward_length([n_ff if i in mlp else 0 for i in range(n_layer)])
    w.add_head_count(n_head)
    w.add_head_count_kv([n_head_kv if i in attn else 0 for i in range(n_layer)])
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)

    # both are required by llama_model_nemotron_h::load_arch_hparams
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_layer_norm_eps(cfg["layer_norm_epsilon"])

    w.add_ssm_conv_kernel(d_conv)
    w.add_ssm_inner_size(d_inner)
    w.add_ssm_state_size(d_state)
    w.add_ssm_group_count(n_group)
    w.add_ssm_time_step_rank(n_ssm_head)  # llama.cpp stores the mamba2 head count here

    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_0)

    # ---------------------------------------------------------------- vocab

    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre(TOKENIZER_PRE)
    w.add_token_list(voc["tokens"])
    w.add_token_types(voc["toktypes"])
    w.add_token_merges(voc["merges"])
    if voc["bos_id"] is not None:
        w.add_bos_token_id(voc["bos_id"])
    if voc["eos_id"] is not None:
        w.add_eos_token_id(voc["eos_id"])
    if voc["pad_id"] is not None:
        w.add_pad_token_id(voc["pad_id"])
    w.add_add_bos_token(voc["add_bos"])
    w.add_add_eos_token(voc["add_eos"])
    if voc["chat_template"]:
        w.add_chat_template(voc["chat_template"])

    # ---------------------------------------------------------------- tensors

    n_copied = 0
    n_converted = 0

    def take(src_name: str) -> dict[str, Any]:
        t = src.tensors.get(src_name)
        if t is None:
            raise SystemExit(f"missing tensor in source: {src_name}")
        return t

    def copy_raw(src_name: str, dst_name: str) -> None:
        """Move a tensor across untouched, quantization blocks and all."""
        nonlocal n_copied
        t = take(src_name)
        block, size = GGML_BLOCK[t["ty"]]

        row_elems = t["dims"][0]
        n_rows = t["elements"] // row_elems
        data = np.frombuffer(src.raw(t), dtype=np.uint8).reshape(n_rows, row_elems // block * size)

        w.add_tensor(dst_name, data,
                     raw_shape=data.shape,
                     raw_dtype=getattr(gguf.GGMLQuantizationType, t["ty"]))
        n_copied += 1
        logger.debug("copy    %-52s -> %-28s %s %s", src_name, dst_name, t["ty"], t["dims"])

    def put_f32(dst_name: str, arr: np.ndarray, src_name: str) -> None:
        nonlocal n_converted
        w.add_tensor(dst_name, np.ascontiguousarray(arr, dtype=np.float32))
        n_converted += 1
        logger.debug("convert %-52s -> %-28s F32 %s", src_name, dst_name, arr.shape)

    # embeddings and output head
    copy_raw(f"stt_model.embed_tokens.weight", "token_embd.weight")
    copy_raw(f"stt_model.lm_head.weight", "output.weight")
    put_f32("output_norm.weight", src.f32(take(f"{SRC_LLM}norm_f.weight")), f"{SRC_LLM}norm_f.weight")

    for i in range(n_layer):
        p = f"{SRC_LLM}layers.{i}."
        b = f"blk.{i}."

        # every block carries the input norm, whatever its mixer is
        put_f32(b + "attn_norm.weight", src.f32(take(p + "norm.weight")), p + "norm.weight")

        if i in mamba:
            copy_raw(p + "mixer.in_proj.weight", b + "ssm_in.weight")
            copy_raw(p + "mixer.out_proj.weight", b + "ssm_out.weight")

            # torch conv1d is [d_inner+2*n_group*d_state, 1, d_conv]; llama.cpp drops the middle
            conv = src.f32(take(p + "mixer.conv1d.weight"))
            put_f32(b + "ssm_conv1d.weight", conv.squeeze(), p + "mixer.conv1d.weight")
            put_f32(b + "ssm_conv1d.bias", src.f32(take(p + "mixer.conv1d.bias")),
                    p + "mixer.conv1d.bias")

            put_f32(b + "ssm_dt.bias", src.f32(take(p + "mixer.dt_bias")), p + "mixer.dt_bias")

            # A_log is stored; llama.cpp wants A = -exp(A_log), unsqueezed to {1, n_head}
            a_log = src.f32(take(p + "mixer.A_log"))
            put_f32(b + "ssm_a", (-np.exp(a_log)).reshape(n_ssm_head, 1), p + "mixer.A_log")

            d = src.f32(take(p + "mixer.D"))
            put_f32(b + "ssm_d", d.reshape(n_ssm_head, 1), p + "mixer.D")

            # the mixer norm is grouped: {d_inner/n_group, n_group}
            norm = src.f32(take(p + "mixer.norm.weight"))
            put_f32(b + "ssm_norm.weight", norm.reshape(n_group, d_inner // n_group),
                    p + "mixer.norm.weight")

        elif i in attn:
            copy_raw(p + "mixer.q_proj.weight", b + "attn_q.weight")
            copy_raw(p + "mixer.k_proj.weight", b + "attn_k.weight")
            copy_raw(p + "mixer.v_proj.weight", b + "attn_v.weight")
            copy_raw(p + "mixer.o_proj.weight", b + "attn_output.weight")

        else:
            copy_raw(p + "mixer.up_proj.weight", b + "ffn_up.weight")
            copy_raw(p + "mixer.down_proj.weight", b + "ffn_down.weight")

    logger.info("tensors: %d copied verbatim, %d converted to F32", n_copied, n_converted)

    dropped = [n for n in src.tensors if not (n.startswith(SRC_LLM)
                                              or n in ("stt_model.embed_tokens.weight",
                                                       "stt_model.lm_head.weight"))]
    logger.info("dropped %d tensors (perception / rnnt / function head / tts / codec)", len(dropped))

    # ---------------------------------------------------------------- write

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w.open_output_file(args.output)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    size = os.path.getsize(args.output)
    logger.info("wrote %s (%.3f GiB)", args.output, size / 1024 ** 3)


if __name__ == "__main__":
    main()
