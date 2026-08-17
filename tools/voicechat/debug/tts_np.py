"""Numpy replica of the VoiceChat TTS forward path, written from the NeMo
reference (rvq_ear_tts_model.py / ear_tts_model.py / duplex_ear_tts.py).

Reads the same voicechat-tts gguf the C++ runtime uses, so any disagreement is a
logic difference, not a weight difference.
"""
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gg import GGUF
from scipy.special import erf

TTS_GGUF = r'D:\Jobs\voicechat\models\voicechat-tts-Q4_0.gguf'


def rms_norm(x, w, eps=1e-6):
    """gemma RMSNorm; the converter folded the +1 into w."""
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def rope(x, pos, theta, head_dim):
    """NeoX/HF rotate_half applied to x [T, H, D]."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(half) * 2.0 / head_dim))
    ang = pos[:, None] * inv[None, :]                 # [T, half]
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class TTS:
    def __init__(self, path=TTS_GGUF):
        g = GGUF(path)
        self.g = g
        t = g.tensor
        self.n_layer = g.kv['voicechat_tts.block_count']
        self.n_embd = g.kv['voicechat_tts.embedding_length']
        self.n_head = g.kv['voicechat_tts.attention.head_count']
        self.head_dim = g.kv['voicechat_tts.attention.key_length']
        self.eps = 1e-6
        self.n_latent = g.kv['voicechat.tts.latent_size']
        self.n_quant = g.kv['voicechat.tts.num_quantizers']
        self.n_codebook = g.kv['voicechat.tts.codebook_size']
        self.low_rank = g.kv['voicechat.tts.mog.low_rank']
        self.n_pred = g.kv['voicechat.tts.mog.num_predictions']
        self.min_log_std = g.kv['voicechat.tts.mog.min_log_std']
        self.guidance = g.kv['voicechat.tts.guidance_scale']
        self.rope_g = g.kv['voicechat_tts.rope.freq_base']
        self.rope_l = g.kv['voicechat.tts.rope_local_theta']
        self.swa = g.kv['voicechat.tts.swa_pattern']
        self.scale = g.kv['voicechat.tts.query_pre_attn_scalar'] ** -0.5
        self.cas_scale = g.kv['voicechat.tts.cas.query_pre_attn_scalar'] ** -0.5
        self.cas_softcap = g.kv['voicechat.tts.cas.attn_softcap']
        self.cas_rope = g.kv['voicechat.tts.cas.rope_theta']
        self.text_eos = g.kv['voicechat.tts.text_eos_id']
        self.text_pad = g.kv['voicechat.tts.text_pad_id']

        self.layers = [dict(
            wq=t(f'blk.{i}.attn_q.weight'), wk=t(f'blk.{i}.attn_k.weight'),
            wv=t(f'blk.{i}.attn_v.weight'), wo=t(f'blk.{i}.attn_output.weight'),
            qn=t(f'blk.{i}.attn_q_norm.weight'), kn=t(f'blk.{i}.attn_k_norm.weight'),
            an=t(f'blk.{i}.attn_norm.weight'), pan=t(f'blk.{i}.post_attention_norm.weight'),
            fn=t(f'blk.{i}.ffn_norm.weight'), pfn=t(f'blk.{i}.post_ffw_norm.weight'),
            gate=t(f'blk.{i}.ffn_gate.weight'), up=t(f'blk.{i}.ffn_up.weight'),
            down=t(f'blk.{i}.ffn_down.weight')) for i in range(self.n_layer)]
        self.out_norm = t('output_norm.weight')

        self.embed_code = t('tts.embed_code.weight')          # (1152, 512)
        self.bos = t('tts.bos_emb')
        self.null = t('tts.null_emb')
        self.prompt = t('tts.audio_prompt_latents')[0]        # (37, 1152)
        self.n_prompt = self.prompt.shape[0]

        self.rvq = t('codec.rvq_embs')                        # (31, 1024, 512)
        self.silence = t('codec.silence_tokens').astype(np.int64)

        # fusion
        self.fa_w, self.fa_b = t('tts.fusion.audio_proj.weight'), t('tts.fusion.audio_proj.bias')
        self.ft_w, self.ft_b = t('tts.fusion.text_proj.weight'), t('tts.fusion.text_proj.bias')
        self.f_norm = t('tts.fusion.norm.weight')
        gate = t('tts.fusion.gate')
        res = 1.0 / (1.0 + np.exp(-t('tts.fusion.residual_scale')[0]))
        s = 1.0 / (1.0 + np.exp(-gate))
        self.g_a, self.g_t = s * res, (1.0 - s) * res

        # subword encoder (t5gemma, 1 encoder layer)
        b = 'tts.subword.blk.0.'
        self.cas = dict(
            wq=t(b + 'attn_q.weight'), wk=t(b + 'attn_k.weight'), wv=t(b + 'attn_v.weight'),
            wo=t(b + 'attn_output.weight'), an=t(b + 'attn_norm.weight'),
            pan=t(b + 'post_attention_norm.weight'), fn=t(b + 'ffn_norm.weight'),
            pfn=t(b + 'post_ffw_norm.weight'), gate=t(b + 'ffn_gate.weight'),
            up=t(b + 'ffn_up.weight'), down=t(b + 'ffn_down.weight'))
        self.cas_embed = t('tts.subword.embed_tokens.weight')   # (257, 1152)
        self.cas_proj = t('tts.subword.proj.weight')            # (1152, 1152)
        self.cas_norm = t('tts.subword.output_norm.weight')
        self.cont_emb = t('tts.subword.cont_emb.weight')        # (2, 1152)
        self.spec_emb = t('tts.subword.special_emb.weight')     # (3, 1152)
        self.cont_flag = t('tts.subword.is_continuation')
        self.spec_flag = t('tts.subword.special_flags')
        self.char_ids = t('tts.subword.char_ids')
        self.char_off = t('tts.subword.char_offsets')

        # mog head
        self.mog = [dict(pre=t(f'tts.mog.blk.{i}.pre_norm.weight'),
                         post=t(f'tts.mog.blk.{i}.post_norm.weight'),
                         gate=t(f'tts.mog.blk.{i}.ffn_gate.weight'),
                         up=t(f'tts.mog.blk.{i}.ffn_up.weight'),
                         down=t(f'tts.mog.blk.{i}.ffn_down.weight')) for i in range(3)]
        self.mog_norm = t('tts.mog.norm.weight')
        self.p_logits = t('tts.mog.logits.weight')     # (1024, 1152)
        self.p_logs = t('tts.mog.logs.weight')         # (1, 1152)
        self.p_else = t('tts.mog.else.weight')         # (512, 1152)
        self.p_mus = t('tts.mog.mus.weight')           # (65536, 1152)
        self.low_mat = t('tts.mog.low_mat')            # (1024, 512, 64)

        self.cache_k = [None] * (2 * self.n_layer)
        self.cache_v = [None] * (2 * self.n_layer)
        self.pos = 0
        self.cond_cache = {}
        # The gemma sqrt(hidden) input normalizer. The reference deletes both
        # models' embedding modules and feeds inputs_embeds, so it only applies
        # where HF keeps it in the forward pass: not in gemma3 (it lives in
        # Gemma3TextScaledWordEmbedding), still there in the t5gemma encoder.
        # Flip either one to see the runtime's old failure mode.
        self.scale_backbone_in = False
        self.scale_cas_in = True

    # ------------------------------------------------------------------ pieces
    def attn_block(self, L, x, pos_ids, theta, scale, softcap, causal, cache_i=None):
        """x [T, E]. With cache_i the k/v are appended to that layer's cache."""
        H, D = self.n_head, self.head_dim
        T = x.shape[0]
        cur = rms_norm(x, L['an'], self.eps)
        q = (cur @ L['wq'].T).reshape(T, H, D)
        k = (cur @ L['wk'].T).reshape(T, H, D)
        v = (cur @ L['wv'].T).reshape(T, H, D)
        if L.get('qn') is not None:
            q = rms_norm(q, L['qn'], self.eps)
            k = rms_norm(k, L['kn'], self.eps)
        q = rope(q, pos_ids, theta, D)
        k = rope(k, pos_ids, theta, D)
        if cache_i is not None:
            if self.cache_k[cache_i] is None:
                self.cache_k[cache_i], self.cache_v[cache_i] = k, v
            else:
                self.cache_k[cache_i] = np.concatenate([self.cache_k[cache_i], k], 0)
                self.cache_v[cache_i] = np.concatenate([self.cache_v[cache_i], v], 0)
            k, v = self.cache_k[cache_i], self.cache_v[cache_i]
        S = k.shape[0]
        att = np.einsum('thd,shd->hts', q, k) * scale
        if softcap > 0:
            att = np.tanh(att / softcap) * softcap
        if causal:
            m = np.triu(np.ones((T, S), bool), k=S - T + 1)
            att = np.where(m[None], -np.inf, att)
        att = att - att.max(-1, keepdims=True)
        att = np.exp(att)
        att /= att.sum(-1, keepdims=True)
        o = np.einsum('hts,shd->thd', att, v).reshape(T, H * D)
        o = rms_norm(o @ L['wo'].T, L['pan'], self.eps)
        x = x + o
        f = rms_norm(x, L['fn'], self.eps)
        f = (gelu_tanh(f @ L['gate'].T) * (f @ L['up'].T)) @ L['down'].T
        return x + rms_norm(f, L['pfn'], self.eps)

    def frame_cond(self, tok):
        if tok in self.cond_cache:
            return self.cond_cache[tok]
        cond = np.zeros(self.n_embd, np.float32)
        c0, c1 = self.char_off[tok], self.char_off[tok + 1]
        if c1 > c0:
            ids = self.char_ids[c0:c1]
            x = self.cas_embed[ids] * (math.sqrt(self.n_embd) if self.scale_cas_in else 1.0)
            x = self.attn_block(self.cas, x, np.arange(len(ids)), self.cas_rope,
                                self.cas_scale, self.cas_softcap, causal=False)
            x = rms_norm(x, self.cas_norm, self.eps)
            cond = (x.mean(0) @ self.cas_proj.T).astype(np.float32)
        cond = cond + self.cont_emb[self.cont_flag[tok]] + self.spec_emb[self.spec_flag[tok]]
        self.cond_cache[tok] = cond
        return cond

    def depthsum(self, code):
        z = np.zeros(self.n_latent, np.float32)
        for q in range(self.n_quant):
            c = code[q]
            if 0 <= c < self.n_codebook:
                z += self.rvq[q][c]
        return z

    def backbone(self, code_emb, cond, pos_ids):
        """code_emb/cond [T, E] for one CFG half. Returns [T, E]."""
        a = (code_emb / self.n_quant) @ self.fa_w.T + self.fa_b
        tt = cond @ self.ft_w.T + self.ft_b
        x = rms_norm(a * self.g_a + tt * self.g_t, self.f_norm, self.eps)
        if self.scale_backbone_in:
            x = x * math.sqrt(self.n_embd)
        return x

    def backbone_step(self, code_emb, cond, half):
        """half 0 = conditional, 1 = unconditional. Uses caches 0..27 / 28..55."""
        T = code_emb.shape[0]
        pos_ids = np.arange(self.pos, self.pos + T)
        x = self.backbone(code_emb, cond, pos_ids)
        for il in range(self.n_layer):
            theta = self.rope_g if (il + 1) % self.swa == 0 else self.rope_l
            x = self.attn_block(self.layers[il], x, pos_ids, theta, self.scale, 0.0,
                                causal=True, cache_i=il + half * self.n_layer)
        return rms_norm(x, self.out_norm, self.eps)

    def mog_head(self, h_cond, h_uncond, code):
        ce = self.embed_code @ self.depthsum(code)
        x = np.stack([ce + h_cond, ce + h_uncond])
        for B in self.mog:
            y = rms_norm(x, B['pre'], self.eps)
            y = (gelu_tanh(y @ B['gate'].T) * (y @ B['up'].T)) @ B['down'].T
            x = x + rms_norm(y, B['post'], self.eps)
        x = rms_norm(x, self.mog_norm, self.eps)
        xg = x[0] + self.guidance * (x[0] - x[1])
        return xg, self.p_logits @ xg, max(float((self.p_logs @ xg)[0]), self.min_log_std), self.p_else @ xg

    # ------------------------------------------------------------------- driver
    def warmup(self):
        T = self.n_prompt
        code_emb = np.zeros((T, self.n_embd), np.float32)
        code_emb[:T - 1] = self.prompt[:T - 1]
        code_emb[T - 1] = self.embed_code @ self.depthsum(self.silence) + self.bos
        cond = np.zeros((T, self.n_embd), np.float32)
        flags = self.cont_emb[self.cont_flag[self.text_pad]] + self.spec_emb[self.spec_flag[self.text_pad]]
        for t in range(T):
            if t == T - 2:
                cond[t] = self.frame_cond(self.text_pad)
            elif t == T - 1:
                cond[t] = self.frame_cond(self.text_eos)
            else:
                cond[t] = flags
        hc = self.backbone_step(code_emb, cond, 0)
        hu = self.backbone_step(code_emb, np.tile(self.null, (T, 1)), 1)
        self.pos += T
        return code_emb[-1], cond[-1], hc[-1], hu[-1]

    def step(self, tok, prev_code):
        if tok == self.text_eos:
            prev_code = self.silence
        ce = self.embed_code @ self.depthsum(prev_code)
        cond = self.frame_cond(tok)
        hc = self.backbone_step(ce[None], cond[None], 0)[0]
        hu = self.backbone_step(ce[None], self.null[None], 1)[0]
        self.pos += 1
        return ce, cond, hc, hu
