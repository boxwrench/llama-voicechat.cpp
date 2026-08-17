"""Minimal GGUF reader with F32/F16/Q4_0/I32 dequantisation (numpy)."""
import struct
import numpy as np

FMT = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i',
       6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}

TYPE_F32, TYPE_F16, TYPE_Q4_0, TYPE_I32 = 0, 1, 2, 26


class GGUF:
    def __init__(self, path):
        self.f = open(path, 'rb')
        f = self.f
        rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        magic, ver = rd('<4sI')
        assert magic == b'GGUF', magic
        nt, nkv = rd('<QQ')

        def rstr():
            (l,) = rd('<Q')
            return f.read(l).decode('utf-8', 'replace')

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                (et,) = rd('<I')
                (n,) = rd('<Q')
                return [rstr() if et == 8 else rd(FMT[et])[0] for _ in range(n)]
            return rd(FMT[t])[0]

        self.kv = {}
        for _ in range(nkv):
            k = rstr()
            (t,) = rd('<I')
            self.kv[k] = rval(t)

        self.info = {}
        for _ in range(nt):
            name = rstr()
            (nd,) = rd('<I')
            dims = rd('<' + 'Q' * nd)
            (ty,) = rd('<I')
            (off,) = rd('<Q')
            self.info[name] = (dims, ty, off)

        align = self.kv.get('general.alignment', 32)
        pos = f.tell()
        self.data_start = pos + (-pos) % align

    def raw(self, name):
        dims, ty, off = self.info[name]
        n = 1
        for d in dims:
            n *= d
        if ty == TYPE_F32:
            nb = n * 4
        elif ty == TYPE_F16:
            nb = n * 2
        elif ty == TYPE_I32:
            nb = n * 4
        elif ty == TYPE_Q4_0:
            assert dims[0] % 32 == 0
            nb = n // 32 * 18
        else:
            raise ValueError(f'unsupported type {ty} for {name}')
        self.f.seek(self.data_start + off)
        return dims, ty, self.f.read(nb)

    def tensor(self, name):
        """Returns the tensor with numpy (C-order) shape = reversed ggml ne."""
        dims, ty, buf = self.raw(name)
        n = 1
        for d in dims:
            n *= d
        if ty == TYPE_F32:
            a = np.frombuffer(buf, dtype='<f4')
        elif ty == TYPE_F16:
            a = np.frombuffer(buf, dtype='<f2').astype(np.float32)
        elif ty == TYPE_I32:
            a = np.frombuffer(buf, dtype='<i4')
        elif ty == TYPE_Q4_0:
            b = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 18)
            d = b[:, :2].copy().view('<f2').astype(np.float32)         # [nb,1]
            q = b[:, 2:]                                                # [nb,16]
            lo = (q & 0x0F).astype(np.int8) - 8
            hi = (q >> 4).astype(np.int8) - 8
            a = (np.concatenate([lo, hi], axis=1).astype(np.float32) * d).reshape(-1)
        else:
            raise ValueError(ty)
        assert a.size == n, (name, a.size, n)
        return a.reshape(tuple(reversed(dims)))
