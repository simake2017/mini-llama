// ============================================================================
// ① 定位：量化反量化的公共接口 —— 把 GGUF 里成块的量化字节还原成 FP32 的一行
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:553:dequantize_row_q8_0
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:1529:dequantize_row_q4_K
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:1731:dequantize_row_q5_K
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:1939:dequantize_row_q6_K
//    llama.cpp @ 88b47a755  ggml/src/ggml-common.h:89:QK_K / :251:QK8_0（每块元素数）
//
// ③ 动机：
//    【概念】量化 = 固定数量元素打一块、块内共享一组缩放系数；推理侧只做一件事：
//       把块里的整数还原成浮点。所以接口的形状必然是「一行、连续 n 个元素」。
//    【简化】只保留 Q8_0 / Q4_K / Q5_K / Q6_K 四支：四支已覆盖主流推理模型（Q4_K_M
//       等混合量化 = 张量级类型选择，加载器天然支持）；承重判据见 04-对齐约束.md §2.1 组 E。
//    【对照】上游签名收结构体指针（const block_q4_K *），本实现收裸字节指针
//       （const uint8_t * p）—— 字节序列相同，差别只在「同一偏移要写几遍」。
//
// ④ demo + 多方样例：
//    🟢 块尺寸（src/gguf.cpp 的 ggml_block_params）：Q8_0 = 34 字节 / 32 元素，
//       Q4_K = 144 / 256，Q5_K = 176 / 256，Q6_K = 210 / 256
//    🟡 真实模型：TinyLlama 的 token_embd 是 Q8_0，首行首块反量化后 min = -1.51396e-05、
//       max = 1.46627e-05（lines/numeric/dequant_line.cpp 第 6 跳）
//    🔴 n 不是块大小整数倍 → 尾部 (n % 块大小) 个元素**根本没被写**（实测 y[32] 仍是调用方留下的旧值）；上游靠 ggml-quants.c:556 的 assert 拦
//    调用方：src/model.cpp 的 dequant_column 按张量类型分派到这四个函数
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>

namespace minillama {

// 每种类型每块元素数（对照 ggml-common.h:89 的 QK_K 与 :251 的 QK8_0）。
// 这两个常量是「块」这个概念的全部：元素数决定切几块，块字节数由 ggml_block_params 给。
constexpr int QK8_0  = 32;
constexpr int QK_K   = 256;

// ============================================================================
// ① 定位：Q8_0 反量化 —— 每 32 个 int8 共享一个 f16 缩放系数 d，还原成 FP32
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:553:dequantize_row_q8_0
//    llama.cpp @ 88b47a755  ggml/src/ggml-common.h:251-255:block_q8_0（d + int8 qs[32]）
//
// ③ 动机：
//    【概念】调用契约：p 指向块首字节（2 字节 f16 的 d + 32 字节 int8 的 qs），
//       n 必须是 32 的整数倍，y 接住 n 个 FP32。量化位本身带符号，零点的位置由
//       符号本身给出来 —— 所以 Q8_0 不带偏移项，这正是 K-quants 要多带一个 dmin 的原因。
//    【陷阱】n 不整除 32 时末块不会被处理：上游靠 ggml-quants.c:556 的 assert 拦，
//       Release 下 NDEBUG 会把它预处理掉；本仓库改在加载期按块大小整除来拦。
//
// ④ demo + 多方样例：
//    🟢 合成块 d = 1.0、qs[j] = j + 1 → 还原值 1..32（lines/numeric/dequant_line.cpp 第 2 跳）
//    🟡 两块连续：块 1 的 d = 2.0、块 2 的 d = 4.0 → y[0] = 2、y[31] = 64、
//       y[32] = 4、y[63] = 128（块步进 34 字节）
//    🔴 n = 33 → 只有 1 个整块被处理，尾部 1 个元素根本没被写（实测 y[32] 仍是调用方
//       留下的旧值），返回值不报错
//    调用方：src/model.cpp 的 dequant_column
// ============================================================================
void dequant_row_q8_0(const uint8_t * p, float * y, size_t n);

// ============================================================================
// ① 定位：Q6_K 反量化 —— 有符号 6bit 量化（-32~31），无 dmin，块内切 16 个 16 元素子块
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:1939:dequantize_row_q6_K
//    llama.cpp @ 88b47a755  ggml/src/ggml-common.h:362-367:block_q6_K（ql[128] / qh[64] / scales[16] / d）
//
// ③ 动机：
//    【概念】调用契约：p 指向 210 字节的块首字节，n 必须是 256 的整数倍。布局与
//       Q4_K / Q5_K 相反 —— d 与 scales 排在块尾（scales 在 p + 192、d 在 p + 208），
//       量化位在最前面。
//    【陷阱】6bit 值是有符号的：解出来的 0~63 要减 32 才是真正的量化值（-32~31），
//       没有 dmin —— 对称量化不需要偏移项，零点的位置由「减 32」直接给出来。
//
// ④ demo + 多方样例：
//    🟢 ql = qh = 0 → 6bit 值 0，减 32 → -32；sc = 2、d = 0.25 → 256 个元素全 -16.0
//       （tests/test_quant.cpp:159-160）
//    🟡 改 ql[0] = 0x21、qh[0] = 0x01 → y[0] = -7.5，**同时** y[64] 变成 -15.0 ——
//       后者是 ql[0] 的 hi nibble 改了元素 64 的 q3，不是 bug（tests/test_quant.cpp:162-166）
//    🔴 ql 一个字节横跨两行、qh 一个字节横跨四行（4 组 2 bit 分别给 l / l+32 / l+64 / l+96）；
//       只验第 0 个元素，hi nibble 与 qh 的 4 组 2 bit 里有 3 组从未被验过
//       （docs/05-quantization.md §17.3）
//    调用方：src/model.cpp 的 dequant_column
// ============================================================================
void dequant_row_q6_k(const uint8_t * p, float * y, size_t n);

// ============================================================================
// ① 定位：Q5_K 反量化 —— 比 Q4_K 多一个「第 5 比特」，它存在独立的 qh[32] 里
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:1731:dequantize_row_q5_K
//    llama.cpp @ 88b47a755  ggml/src/ggml-common.h:344-355:block_q5_K（d / dmin / scales[12] / qh[32] / qs[128]）
//
// ③ 动机：
//    【概念】调用契约与 Q4_K 同构（n 是 256 的整数倍），块多 32 字节：低 4 bit 在
//       qs[128]（一字节两个元素），第 5 bit 另存 qh[32]（一字节 8 个比特、每个比特管
//       一个元素的高位，32 字节正好覆盖全部 256 个元素）。量化位不连续，就得拆两个
//       数组存、读时再跨数组拼回来。
//    【陷阱】本实现的块内字段顺序是 scales → qs → qh，上游是 scales → qh → qs ——
//       已知偏离，已登记为待拍板项（docs/05-quantization.md §15 / §17.9）。
//
// ④ demo + 多方样例：
//    🟢 qh 全 0 → 第 5 比特不生效，结果与 Q4_K 同构（lines/numeric/dequant_line.cpp 第 4 跳，
//       实测 min = max = 1）
//    🟡 qh[l] 第 0 比特 = 1 → 元素 l < 32 为 (l % 16) + 16、l >= 32 为 0
//       （tests/test_quant.cpp:135-141）
//    🔴 同一段字节两种解法：本实现取 qh 于 p + 144，上游在 p + 16。拿上游导出的 Q5_K
//       GGUF 加载时，第 5 比特会与低 4 bit 串位（docs/05-quantization.md §17.9）
//    调用方：src/model.cpp 的 dequant_column
// ============================================================================
void dequant_row_q5_k(const uint8_t * p, float * y, size_t n);

// ============================================================================
// ① 定位：Q4_K 反量化 —— 256 元素 / 144 字节的块，还原成 FP32（非对称量化，带 dmin）
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:1529:dequantize_row_q4_K
//    llama.cpp @ 88b47a755  ggml/src/ggml-common.h:327-337:block_q4_K（d / dmin / scales[12] / qs[128]）
//
// ③ 动机：
//    【概念】调用契约：p 指向 144 字节的块首字节（f16 d + f16 dmin + 12 字节压缩的
//       scales + 128 字节的 qs），n 必须是 256 的整数倍。块内切 8 个 32 元素子块，
//       各带自己的 scale / min —— 子块越小，共享一套缩放系数越浪费比特。
//    【陷阱】qs 的起点是 p + 16（d + dmin + scales 之后），不是 p + 4；偏移写错读出来
//       是「形状对、值全错」。上游由 block_q4_K 的字段顺序兜住，本实现靠人保证。
//
// ④ demo + 多方样例：
//    🟢 合成块（qs 全 1、sc = 1、m = 0）→ 256 个元素全 1.0（lines/numeric/dequant_line.cpp 第 3 跳）
//    🟡 逐元素对拍：d = 2.0、dmin = 1.0、scales[0] = 1 时，元素 l < 32 应为
//       2 * (l % 16) - 1、元素 l >= 32 应为 0（tests/test_quant.cpp:110-116）
//    🔴 qs 的 128 个字节各供 2 个元素（lo / hi nibble 各 1 个）：只验前 32 个元素等于
//       没验剩下 224 个 —— 合成块里 qs[32..127] 全 0 时元素 32..255 全是 0，测试照样通过
//    调用方：src/model.cpp 的 dequant_column
// ============================================================================
void dequant_row_q4_k(const uint8_t * p, float * y, size_t n);

} // namespace minillama
