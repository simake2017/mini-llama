#!/usr/bin/env python3
# 独立仲裁者：纯 numpy 从 GGUF 原始字节实现 TinyLlama 前向（float64 累加），
# 不依赖 mini-llama 任何代码。裁决 ours-vs-llama.cpp 发散归属。
# 用法: arbiter.py model.gguf 1,1576
import struct, sys
import numpy as np

GT_MAP = {0:'f32', 1:'f16', 2:'q4_0', 3:'q8_0'}

def f16_to_f32(h):
    h = h.astype(np.uint32)
    sign = (h & 0x8000) << 16
    exp = ((h >> 10) & 0x1F).astype(np.int64)
    mant = (h & 0x3FF).astype(np.float64)
    f = np.where(exp == 0, mant * 2.0**-24,
        np.where(exp == 31, np.inf, (1 + mant / 1024.0) * 2.0**(exp - 15)))
    return (f * np.where(h & 0x8000, -1.0, 1.0)).astype(np.float64)

class Gguf:
    def __init__(self, path):
        b = open(path, 'rb').read()
        self.b = b
        off = 4  # magic
        ver = struct.unpack_from('<I', b, off)[0]; off += 4
        # v1: u32 ×2；v2/v3: u64 ×2
        if ver == 1:
            n_tensors, n_kv = struct.unpack_from('<II', b, off); off += 8
        else:
            n_tensors, n_kv = struct.unpack_from('<QQ', b, off); off += 16
        self.kv = {}
        for _ in range(n_kv):
            k, off = self._str(b, off)
            t = struct.unpack_from('<I', b, off)[0]; off += 4
            v, off = self._value(b, off, t)
            self.kv[k] = v
        self.tensors = {}
        for _ in range(n_tensors):
            name, off = self._str(b, off)
            nd = struct.unpack_from('<I', b, off)[0]; off += 4
            dims = struct.unpack_from(f'<{nd}Q', b, off); off += 8 * nd
            gtype = struct.unpack_from('<I', b, off)[0]; off += 4
            toff = struct.unpack_from('<Q', b, off)[0]; off += 8
            self.tensors[name] = (dims, gtype, toff)
        self.data_base = (off + 31) // 32 * 32  # 对齐到 32

    @staticmethod
    def _str(b, off):
        n = struct.unpack_from('<Q', b, off)[0]; off += 8
        return b[off:off+n].decode('utf-8'), off + n

    @classmethod
    def _value(cls, b, off, t):
        if t == 4: return struct.unpack_from('<I', b, off)[0], off + 4
        if t == 5: return struct.unpack_from('<i', b, off)[0], off + 4
        if t == 6: return struct.unpack_from('<f', b, off)[0], off + 4
        if t == 7: return bool(b[off]), off + 1
        if t == 8: return cls._str(b, off)
        if t == 10: return struct.unpack_from('<Q', b, off)[0], off + 8
        if t == 9:  # array
            et = struct.unpack_from('<I', b, off)[0]; off += 4
            n = struct.unpack_from('<Q', b, off)[0]; off += 8
            if et == 8: return [cls._str(b, off := off)[0] for _ in range(n)] if False else cls._str_array(b, off, n)
            sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8}[et]
            fmt = {0:'b',1:'B',2:'h',3:'H',4:'i',5:'i',6:'f',7:'?',10:'q'}[et]
            arr = list(struct.unpack_from(f'<{n}{fmt}', b, off)); off += n * sz
            return arr, off
        raise ValueError(f'kv type {t}')

    @classmethod
    def _str_array(cls, b, off, n):
        out = []
        for _ in range(n):
            s, off = cls._str(b, off)
            out.append(s)
        return out, off

    def raw(self, name):
        dims, gtype, toff = self.tensors[name]
        # 张量 offset 相对数据区起点，且每个张量起点按 32 对齐累计
        # 简化：从 data_base + align(rel) 取；尺寸按类型算
        return dims, gtype, self.data_base + toff

    def dequant(self, name):
        dims, gtype, off = self.raw(name)
        n_elem = int(np.prod(dims))
        b = self.b
        if gtype == 0:  # f32
            return np.frombuffer(b, dtype='<f4', count=n_elem, offset=off).astype(np.float64).reshape(dims, order='F')
        if gtype == 1:  # f16
            h = np.frombuffer(b, dtype='<u2', count=n_elem, offset=off)
            return f16_to_f32(h).reshape(dims, order='F')
        if gtype == 8:  # q8_0: 块 32 = (f16 d, i8 quants×32)，与 ggml block_q8_0 一致
            nblk = n_elem // 32
            raw = np.frombuffer(b, dtype=np.uint8, count=nblk * 34, offset=off)
            raw = raw.reshape(nblk, 34)
            d16 = raw[:, 0].astype(np.uint32) | (raw[:, 1].astype(np.uint32) << 8)
            d = f16_to_f32(d16)
            q = raw[:, 2:34].astype(np.int8).astype(np.float64)
            return (q * d[:, None]).reshape(n_elem).reshape(dims, order='F')
        raise ValueError(f'{name} type {gtype}')

def rmsnorm(x, w, eps):
    ms = np.mean(x * x)
    return x / np.sqrt(ms + eps) * w

def rope(x, pos, theta):
    # x: (n_head, head_dim)。相邻元素配对 (2i, 2i+1)，与 llama.cpp master 的
    # LLAMA_ROPE_TYPE_NORM 一致（LLM_ARCH_LLAMA 的 rope 类型，见 llama-model.cpp）
    h = x.shape[-1]
    half = h // 2
    i = np.arange(half)
    freq = theta ** (-2.0 * i / h)
    ang = pos * freq
    c, s = np.cos(ang), np.sin(ang)
    x0, x1 = x[..., 0::2].copy(), x[..., 1::2].copy()
    x[..., 0::2] = x0 * c - x1 * s
    x[..., 1::2] = x0 * s + x1 * c
    return x

def main():
    g = Gguf(sys.argv[1])
    toks = [int(t) for t in sys.argv[2].split(',')]
    hp = g.kv
    n_layer = hp['llama.block_count']
    d = hp['llama.embedding_length']
    nh = hp['llama.attention.head_count']
    nkv = hp['llama.attention.head_count_kv']
    hd = d // nh
    nff = hp['llama.feed_forward_length']
    eps = hp['llama.attention.layer_norm_rms_epsilon']
    theta = hp['llama.rope.freq_base']
    nv = hp['llama.vocab_size']

    emb = g.dequant('token_embd.weight')        # (d, nv) GGUF 布局
    out_w = g.dequant('output.weight') if 'output.weight' in g.tensors else emb
    out_norm = g.dequant('output_norm.weight')

    seq = len(toks)
    xs = [emb[:, t].copy() for t in toks]       # 每位置残差流
    K = [[] for _ in range(n_layer)]            # 每层 K 历史 (nkv, hd) 列表
    V = [[] for _ in range(n_layer)]
    for l in range(n_layer):
        wq = g.dequant(f'blk.{l}.attn_q.weight')
        wk = g.dequant(f'blk.{l}.attn_k.weight')
        wv = g.dequant(f'blk.{l}.attn_v.weight')
        wo = g.dequant(f'blk.{l}.attn_output.weight')
        an = g.dequant(f'blk.{l}.attn_norm.weight')
        fn = g.dequant(f'blk.{l}.ffn_norm.weight')
        wg = g.dequant(f'blk.{l}.ffn_gate.weight')
        wu = g.dequant(f'blk.{l}.ffn_up.weight')
        wd = g.dequant(f'blk.{l}.ffn_down.weight')
        for p in range(seq):
            h = rmsnorm(xs[p], an, eps)
            q = wq.T @ h                        # (d,)
            k = wk.T @ h                        # (kvd,)
            v = wv.T @ h
            q = rope(q.reshape(nh, hd), p, theta).reshape(-1)
            k = rope(k.reshape(nkv, hd), p, theta).reshape(-1)
            K[l].append(k); V[l].append(v)
            att = np.zeros(d)
            scale = 1.0 / np.sqrt(hd)
            for hq in range(nh):
                hkv = hq // (nh // nkv)
                qh = q[hq*hd:(hq+1)*hd]
                sc = np.array([qh @ K[l][j][hkv*hd:(hkv+1)*hd] for j in range(p+1)]) * scale
                sc = np.exp(sc - sc.max()); sc /= sc.sum()
                oh = np.zeros(hd)
                for j in range(p+1):
                    oh += sc[j] * V[l][j][hkv*hd:(hkv+1)*hd]
                att[hq*hd:(hq+1)*hd] = oh
            xs[p] = xs[p] + wo.T @ att
            h = rmsnorm(xs[p], fn, eps)
            gate = wg.T @ h
            up = wu.T @ h
            gate = gate / (1 + np.exp(-gate)) * up
            xs[p] = xs[p] + wd.T @ gate
    x = rmsnorm(xs[-1], out_norm, eps)
    logits = out_w.T @ x
    idx = np.argsort(-logits)[:4]
    for i, t in enumerate(idx):
        print(f'top{i} id={t} logit={logits[t]:.4f}')
    for i in [29892,29900,29889,29929,29906,13,29896,1734,937,5828,4148,29871,9750,2000,263,278,297,451,472,385,529]:
        print(f'id={i} {logits[i]:.4f}')

main()
