// 量化反量化（单元 5）：Q8_0 / Q6_K / Q5_K / Q4_K 块格式 → FP32
//
// 对照真实 llama.cpp: ggml/src/ggml-quants.c 的 dequantize_row_q* 系列。
// 以下实现逐行移植同一公式，变量名保持一致便于对照。
//
// 【概念】K-quants（Q4_K 等）的块结构（256 元素 = 144 字节，压缩率 7.1x）:
//   ┌────────┬────────┬──────────────────┬─────────────────┐
//   │ d (f16)│dmin(f16)│ scales[12]       │ qs[128]         │
//   │ 块缩放 │ 块偏移  │ 16 个子块缩放/偏移│ 256 个 4bit 量化 │
//   └────────┴────────┴──────────────────┴─────────────────┘
//    value[i] = d * sc[子块(i)] * nibble(i) - dmin * m[子块(i)]
//   scales 12 字节装 16 个 6bit 数，打包/解包见 get_scale_min_k4 ——
//   这是本文件里唯一"魔幻"的位操作，对照 ggml 同名函数。

#pragma once

#include <cstddef>
#include <cstdint>

namespace minillama {

// 每种类型每块元素数
constexpr int QK8_0  = 32;
constexpr int QK_K   = 256;

// 反量化一行（连续 n 个元素，n 必须是块大小的整数倍），写入 y
void dequant_row_q8_0(const uint8_t * p, float * y, size_t n);
void dequant_row_q6_k(const uint8_t * p, float * y, size_t n);
void dequant_row_q5_k(const uint8_t * p, float * y, size_t n);
void dequant_row_q4_k(const uint8_t * p, float * y, size_t n);

} // namespace minillama
