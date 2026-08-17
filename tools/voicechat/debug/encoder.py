"""Numpy port of Wav2Latent + PreTrainedProbabilisticVQ.encode (ground truth codes)."""
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gg import GGUF
from codec import Codec, _gelu

SRC = r'C:\Users\Admin\AI\models\nemotron_voicechat_11b-Q4_0.gguf'
E = 'tts_model.audio_codec.encoder.layers.'


class Encoder:
    def __init__(self, path=SRC):
        g = GGUF(path)
        self.g = g
        self.proj_in = g.tensor(E + '0.weight')[:, :, 0]        # (384, 18)
        self.down = [g.tensor(E + f'{i}.weight') for i in (4, 8, 12)]   # (in, out, K)
        self.blk = []
        for i in (1, 2, 3, 5, 6, 7, 9, 10, 11):
            b = E + f'{i}.'
            self.blk.append(dict(
                dw_w=g.tensor(b + 'dwconv.weight')[:, 0, :],
                dw_b=g.tensor(b + 'dwconv.bias'),
                nw=g.tensor(b + 'norm.weight'),
                nb=g.tensor(b + 'norm.bias'),
                w1=g.tensor(b + 'pwconv1.weight')[:, :, 0],
                b1=g.tensor(b + 'pwconv1.bias'),
                w2=g.tensor(b + 'pwconv2.weight')[:, :, 0],
                b2=g.tensor(b + 'pwconv2.bias'),
            ))
        self.rates = [7, 7, 9]
        self.n_fft, self.hop = 16, 4

    def stft(self, wav):
        n_fft, hop = self.n_fft, self.hop
        pl = (n_fft - hop) // 2
        pr = (n_fft - hop) - pl
        w = np.pad(wav.astype(np.float64), (pl, pr))
        win = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))
        nfr = (len(w) - n_fft) // hop + 1
        fr = np.stack([w[i * hop:i * hop + n_fft] * win for i in range(nfr)], axis=1)
        return np.fft.rfft(fr, n=n_fft, axis=0)          # (9, nfr)

    @staticmethod
    def down_conv(x, w, stride):
        """x (Cin, L), w (Cout, Cin, K), K == stride, no padding -> (Cout, L//K)."""
        Cin, L = x.shape
        Cout, _, K = w.shape
        assert K == stride
        n = L // K
        xr = x[:, :n * K].reshape(Cin, n, K)
        return np.einsum('ilk,oik->ol', xr, w)

    def encode_latents(self, wav):
        spec = self.stft(wav)
        x = np.concatenate([spec.real, spec.imag], axis=0).astype(np.float32)   # (18, L)
        x = self.proj_in @ x
        b = 0
        for s in range(3):
            for _ in range(3):
                x = Codec.convnext(x, self.blk[b]); b += 1
            x = self.down_conv(x, self.down[s], self.rates[s])
        return x.T                                        # (T, 512)

    @staticmethod
    def quantize(z, rvq, depth=31):
        T = z.shape[0]
        r = z.astype(np.float32).copy()
        codes = np.zeros((T, depth), dtype=np.int64)
        for i in range(depth):
            mus = rvq[i]                                   # (1024, 512)
            d = (mus * mus).sum(1)[None, :] - 2.0 * (r @ mus.T)
            c = d.argmin(1)
            codes[:, i] = c
            r = r - mus[c]
        return codes, r
