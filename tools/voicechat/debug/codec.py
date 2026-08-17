"""Reference (numpy) implementation of the VoiceChat RVQ codec decoder.

Mirrors ref_speechlm2/ear_tts_vae_codec.py: Latent2Wav + spec_to_wav.
Everything is done in one shot (no chunking) so it is an independent check of
the chunked ggml path in voicechat-tts.cpp.
"""
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gg import GGUF

GGUF_PATH = r'D:\Jobs\voicechat\models\voicechat-tts-Q4_0.gguf'


from scipy.special import erf as _erf


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


class Codec:
    def __init__(self, path=GGUF_PATH):
        g = GGUF(path)
        self.g = g
        self.n_fft = g.kv['voicechat.codec.n_fft']
        self.hop = g.kv['voicechat.codec.hop_length']
        self.rates = list(reversed(g.kv['voicechat.codec.rates']))   # decoder order
        self.sr = g.kv['voicechat.codec.sample_rate']
        self.n_quant = g.kv['voicechat.tts.num_quantizers']
        self.n_codebook = g.kv['voicechat.tts.codebook_size']

        # rvq_embs ggml ne (512, 1024, 31) -> numpy (31, 1024, 512)
        self.rvq = g.tensor('codec.rvq_embs')
        self.silence = g.tensor('codec.silence_tokens')

        # ConvTranspose1d weights, ggml ne (K, Cout, Cin) -> numpy (Cin, Cout, K)
        self.up = [g.tensor(f'codec.dec.up.{i}.weight') for i in (0, 4, 8)]
        # head: ggml ne (1, 384, 18) -> numpy (18, 384, 1)
        self.head = g.tensor('codec.dec.head.weight')[:, :, 0]        # (18, 384)

        self.blk = []
        for i in range(9):
            b = f'codec.dec.blk.{i}.'
            self.blk.append(dict(
                dw_w=g.tensor(b + 'dwconv.weight')[:, 0, :],          # (C, K)
                dw_b=g.tensor(b + 'dwconv.bias'),
                nw=g.tensor(b + 'norm.weight'),
                nb=g.tensor(b + 'norm.bias'),
                w1=g.tensor(b + 'pw1.weight'),                        # (inter, C)
                b1=g.tensor(b + 'pw1.bias'),
                w2=g.tensor(b + 'pw2.weight'),                        # (C, inter)
                b2=g.tensor(b + 'pw2.bias'),
            ))

    # ---------------------------------------------------------------- pieces
    def depthsum(self, codes):
        """codes [T, n_quant] -> latents [T, 512]. Out of range ids embed to 0."""
        T = codes.shape[0]
        z = np.zeros((T, self.rvq.shape[2]), dtype=np.float32)
        for q in range(self.n_quant):
            c = codes[:, q]
            ok = (c >= 0) & (c < self.n_codebook)
            z[ok] += self.rvq[q][c[ok]]
        return z

    @staticmethod
    def convnext(x, B):
        """x (C, L) -> (C, L), causal depthwise conv + layernorm over channels."""
        C, L = x.shape
        K = B['dw_w'].shape[1]
        xp = np.concatenate([np.zeros((C, K - 1), np.float32), x], axis=1)
        y = np.zeros((C, L), np.float32)
        for j in range(K):
            y += xp[:, j:j + L] * B['dw_w'][:, j:j + 1]
        y += B['dw_b'][:, None]
        m = y.mean(0, keepdims=True)
        s = y - m
        y = s / np.sqrt((s * s).mean(0, keepdims=True) + 1e-6)
        y = y * B['nw'][:, None] + B['nb'][:, None]
        y = B['w1'] @ y + B['b1'][:, None]
        y = _gelu(y)
        y = B['w2'] @ y + B['b2'][:, None]
        return x + y

    @staticmethod
    def conv_transpose(x, w, stride):
        """x (Cin, L), w (Cin, Cout, K) with K == stride -> (Cout, L*stride)."""
        Cin, L = x.shape
        _, Cout, K = w.shape
        assert K == stride, (K, stride)
        # each input frame maps to its own block of `stride` outputs
        y = np.einsum('il,iok->olk', x, w)          # (Cout, L, K)
        return y.reshape(Cout, L * K)

    def decode_latents(self, lat):
        """lat (T, 512) -> spec channels (18, T*441)."""
        x = lat.T.astype(np.float32)                # (512, T)
        blk = 0
        for s in range(3):
            x = self.conv_transpose(x, self.up[s], self.rates[s])
            for _ in range(3):
                x = self.convnext(x, self.blk[blk])
                blk += 1
        return self.head @ x                        # (18, L)

    def spec_to_wav(self, spec, constrain=True):
        n_fft, hop = self.n_fft, self.hop
        nb = n_fft // 2 + 1
        Ts = spec.shape[1]
        mag = 100.0 * np.exp(-np.logaddexp(0.0, math.log(100.0) - spec[:nb]))
        ph = spec[nb:]
        c = mag * np.exp(1j * ph)
        c[0] = mag[0] * np.cos(ph[0])
        c[-1] = mag[-1] * np.cos(ph[-1])
        frames = np.fft.irfft(c, n=n_fft, axis=0)   # (n_fft, Ts)
        win = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))
        if constrain:
            frames = np.clip(frames, -win[:, None], win[:, None])
        frames = frames * win[:, None]
        full = (Ts - 1) * hop + n_fft
        wav = np.zeros(full)
        env = np.zeros(full)
        w2 = win * win
        for t in range(Ts):
            wav[t * hop:t * hop + n_fft] += frames[:, t]
            env[t * hop:t * hop + n_fft] += w2
        pad = (n_fft - hop) // 2
        wav = wav[pad:full - pad] / env[pad:full - pad]
        return wav

    def decode_codes(self, codes):
        return self.spec_to_wav(self.decode_latents(self.depthsum(codes)))


def write_wav(path, x, sr=22050):
    import wave
    p = np.clip(x, -1, 1)
    pcm = np.rint(p * 32767).astype('<i2')
    w = wave.open(path, 'wb')
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(pcm.tobytes())
    w.close()
