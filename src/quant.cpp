// 量化反量化实现：逐行移植 ggml/src/ggml-quants.c 的公式

#include "quant.h"

#include "model.h"   // f16_to_f32

namespace minillama {

// ---------------------------------------------------------------------------
// Q8_0: { f16 d; int8 qs[32] }，34 字节 / 32 元素
// value = qs[j] * d
// ---------------------------------------------------------------------------
void dequant_row_q8_0(const uint8_t * p, float * y, size_t n) {
    for (size_t i = 0; i < n / QK8_0; i++) {
        const float d = f16_to_f32(*reinterpret_cast<const uint16_t *>(p));
        p += 2;
        for (int j = 0; j < QK8_0; j++) {
            y[i * QK8_0 + j] = static_cast<int8_t>(*p++) * d;
        }
    }
}

// ---------------------------------------------------------------------------
// K-quants 共用的 scales 解包（6bit × 16 压缩进 12 字节）
// 【对照】ggml-quants.c get_scale_min_k4，逐行一致
// ---------------------------------------------------------------------------
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// Q4_K: { f16 d; f16 dmin; uint8 scales[12]; uint8 qs[128] }，144 字节 / 256 元素
// value = d * sc * nibble - dmin * m
// ---------------------------------------------------------------------------
void dequant_row_q4_k(const uint8_t * p, float * y, size_t n) {
    for (size_t i = 0; i < n / QK_K; i++) {
        const uint8_t * scales = p + 4;          // 12 字节压缩 scales
        const uint8_t * q      = p + 4 + 12;     // qs[128] 紧跟其后
        const float d   = f16_to_f32(*reinterpret_cast<const uint16_t *>(p));
        const float min = f16_to_f32(*reinterpret_cast<const uint16_t *>(p + 2));

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4)  - m2;
            q += 32;
            is += 2;
        }
        p += 144;
    }
}

// ---------------------------------------------------------------------------
// Q5_K: 同 Q4_K 结构 + qh[32] 存每元素第 5 比特，176 字节 / 256 元素
// ---------------------------------------------------------------------------
void dequant_row_q5_k(const uint8_t * p, float * y, size_t n) {
    for (size_t i = 0; i < n / QK_K; i++) {
        const uint8_t * scales = p + 4;
        const uint8_t * ql = p + 4 + 12;            // qs[128]
        const uint8_t * qh = p + 4 + 12 + 128;       // qh[32]
        const float d   = f16_to_f32(*reinterpret_cast<const uint16_t *>(p));
        const float min = f16_to_f32(*reinterpret_cast<const uint16_t *>(p + 2));

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4)  + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        p += 176;
    }
}

// ---------------------------------------------------------------------------
// Q6_K: { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; f16 d }，210 字节 / 256 元素
// value = d * sc * (6bit 量化 - 32)   —— 有符号量化，无 dmin
// ---------------------------------------------------------------------------
void dequant_row_q6_k(const uint8_t * p, float * y, size_t n) {
    for (size_t i = 0; i < n / QK_K; i++) {
        const float d = f16_to_f32(*reinterpret_cast<const uint16_t *>(p + 128 + 64 + 16));

        const uint8_t * ql = p;
        const uint8_t * qh = p + 128;
        const int8_t  * sc = reinterpret_cast<const int8_t *>(p + 128 + 64);

        for (int nn = 0; nn < QK_K; nn += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = static_cast<int8_t>((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = static_cast<int8_t>((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
        p += 210;
    }
}

} // namespace minillama
