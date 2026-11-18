// ============================================================================
// ① 定位：量化反量化内核——把 GGUF 里的量化字节（Q8_0 / Q4_K / Q5_K / Q6_K）逐块还原成 FP32
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:553:dequantize_row_q8_0
//                          ggml/src/ggml-quants.c:1529:dequantize_row_q4_K
//                          ggml/src/ggml-quants.c:1731:dequantize_row_q5_K
//                          ggml/src/ggml-quants.c:1939:dequantize_row_q6_K
//                          ggml/src/ggml-quants.c:880:get_scale_min_k4（K-quants 的 6bit scale 解包）
//    llama.cpp @ 88b47a755  ggml/src/ggml-common.h:251-255:block_q8_0 / :327-337:block_q4_K
//                          / :344-355:block_q5_K / :362-367:block_q6_K（块布局的定义处）
//
// ③ 动机：
//    【概念】量化不是逐元素压缩，而是「固定数量元素打一块、块内共享一组缩放系数」。
//       推理侧只做一件事：把块里的整数还原成浮点。上游把布局写进结构体（字段顺序由
//       编译器保证），本仓库用裸指针 + 偏移常量（顺序由人保证）——所以偏移必须成对出现。
//    【简化】只保留 4 种主流格式、只做反量化。量化（写侧）在上游是离线工具 llama-quantize
//       的职责，推理进程只读不写；砍掉它，「量化 = 块格式」这个核心概念仍完整成立
//       （承重判据见 docs/constraints/04-对齐约束.md §2.1 组 E）。
//
// ④ demo + 多方样例：
//    🟢 块字节数 / 块元素数（src/gguf.cpp 的 ggml_block_params 可查）：Q8_0 = 34 / 32，
//       Q4_K = 144 / 256，Q5_K = 176 / 256，Q6_K = 210 / 256 —— 与上游 type_traits 一致。
//    🟡 跑法：MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_dequant
//       第 2 跳验 Q8_0，第 3~5 跳用合成块验 Q4_K / Q5_K / Q6_K，第 7 跳验 matvec 接入。
//    🔴 两个块格式的坑：Q4_K 的 qs 起点是 p + 16（不是 p + 4）；Q5_K 的 qh / qs 字段顺序
//       与上游相反（本实现 qs→qh，上游 qh→qs）——见 docs/05-quantization.md §17.1 / §17.9。
// ============================================================================

#include "quant.h"

#include "model.h"   // f16_to_f32

namespace minillama {

// ============================================================================
// ① 定位：Q8_0 反量化——每 32 个 int8 共享一个 f16 缩放系数 d，还原成 FP32
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:553:dequantize_row_q8_0
//    llama.cpp @ 88b47a755  ggml/src/ggml-common.h:251-255:block_q8_0（d + int8 qs[32] = 34 字节）
//
// ③ 动机：
//    【概念】Q8_0 是「每块一个缩放系数」的整数量化：value = qs[j] * d。
//       量化位本身有符号（int8，-128~127），零点的位置由符号本身给出来 —— 所以它
//       不需要偏移项，这正是 K-quants 要额外带一个 dmin 的原因。
//    【对照】上游用结构体 x[i].d / x[i].qs 取字段，本仓库用裸指针逐字节推进；字节
//       序列完全相同，差别在「偏移写错时谁会发现」——上游由 static_assert 在编译期
//       兜住（ggml-common.h:256 的块大小断言），本仓库只能在加载期 / 前向期暴露。
//
// ④ demo + 多方样例：
//    🟢 合成块 d = 1.0、qs[j] = j + 1 → 还原值 1..32（lines/numeric/dequant_line.cpp 第 2 跳）
//    🟡 真实模型：token_embd 第 0 行第 1 块反量化后 min <= max（同一文件第 6 跳）
//    🔴 n 不是 32 的整数倍 → 最后一块读过头，把下一个张量的头两个字节当成 d（f16 位模式
//       解释成天文数字或 NaN）。上游靠 ggml-quants.c:556 的 assert 拦，但 Release 下
//       NDEBUG 会把它预处理掉；本仓库在加载期按块大小整除来拦（src/model.cpp 的张量校验）。
//    调用方：src/model.cpp 的 dequant_column 按张量类型分派到本函数
// ============================================================================
void dequant_row_q8_0(const uint8_t * p, float * y, size_t n) {
    for (size_t i = 0; i < n / QK8_0; i++) {
        const float d = f16_to_f32(*reinterpret_cast<const uint16_t *>(p));
        p += 2;
        for (int j = 0; j < QK8_0; j++) {
            y[i * QK8_0 + j] = static_cast<int8_t>(*p++) * d;
        }
    }
}

// ============================================================================
// ① 定位：K-quants 的 6bit scale/min 解包——把 12 字节里压着的 16 个 6bit 数拆回 8 组 (scale, min)
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:880:get_scale_min_k4（解包侧，两路分支逐行一致）
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:1553:quantize_row_q4_K_impl（打包侧，位布局闭环：
//                          1595-1597 行把多出的 2 bit 塞回 scales[j-4] / scales[j] 的高位）
//
// ③ 动机：
//    【陷阱】16 个数 × 6bit = 96 bit = 12 字节，代价是「一个字节不是一个数」：
//       j < 4 时 6bit 落在字节内（q[j] & 63 / q[j+4] & 63）；
//       j >= 4 时 6bit 横跨两个字节 —— scale 的低 4 bit 取 q[j+4]、高 2 bit 从 q[j-4] 的
//       第 6-7 位借；min 的低 4 bit 取 q[j+4] 的第 4-7 位、高 2 bit 从 q[j] 的第 6-7 位借。
//       只实现前一路，子块 4..7 的 scale/min 全错，而前 4 个子块权重最大、误差被掩盖。
//    【对照】上游把打包写在量化侧、解包写在 get_scale_min_k4，两侧共用同一份位布局约定；
//       本仓库只有解包侧，打包侧在离线工具里 —— 于是这份布局约定只存在于这一处函数体。
//
// ④ demo + 多方样例：
//    🟢 q = {1,1,1,1, 0,0,0,0, 1,1,1,1}：j=0 → sc = q[0]&63 = 1、m = q[4]&63 = 0；
//       j=4 → sc = (q[8]&0xF) | ((q[0]>>6)<<4) = 1、m = (q[8]>>4) | ((q[4]>>6)<<4) = 0
//    🔴 只验前 4 个子块等于没验：子块 4..7 走的是另一条解包路径
//       （docs/05-quantization.md §17.2）。两路都必须测。
//    调用方：src/quant.cpp 的 dequant_row_q4_k / dequant_row_q5_k 每块各调 8 次
//       （8 个子块各 1 次，每次同时取出该子块的 sc 与 m）
// ============================================================================
// 私有辅助：只被同文件的两个 K-quant 内核调用，所以取 static inline；函数名与上游同名，
// 便于逐行对照（上游也是 static inline —— 打包侧与解包侧必须在同一编译单元里闭环）。
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// ============================================================================
// ① 定位：Q4_K 反量化——256 元素 / 144 字节的块，还原成 FP32（非对称量化：带 dmin）
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:1529:dequantize_row_q4_K
//    llama.cpp @ 88b47a755  ggml/src/ggml-common.h:327-337:block_q4_K（d / dmin / scales[12] / qs[128]）
//
// ③ 动机：
//    【陷阱】Q4_K 是**无符号 4bit + 减 dmin** 的非对称量化：nibble 只有 0~15，块内最小值
//       靠 dmin * m 单独减掉，所以公式是 d * sc * nibble - dmin * m，不是 d * sc * nibble。
//       当成对称量化读，最小的那几个子块会整体偏一个 dmin * m。
//    【概念】块内切 8 个 32 元素的子块，每个子块有自己的 scale / min（6bit 打包进
//       scales[12]）。子块越小，共享一套缩放系数越浪费比特 —— K-quants 的比特就是这么省的。
//    【对照】上游 q = x[i].qs 一次取到 128 字节，本仓库写死 p + 4 + 12 —— 偏移常量必须
//       成对出现：读 scales 处是 12，算 qs 起点处也必须是 12（docs/05-quantization.md §17.1）。
//
// ④ demo + 多方样例：
//    🟢 合成块 d = 2.0、dmin = 1.0、scales[0] = 1、scales[4] = 1、qs[l] = l：
//       元素 0..31 → 2.0 * (l % 16) - 1.0，元素 32..255 → 0（tests/test_quant.cpp:97-118）
//    🟡 真实模型走的是同一段内核，只是块的字节来自 GGUF 数据区（src/model.cpp 的 dequant_column）
//    🔴 一个 qs 字节的 lo / hi nibble 各供 32 个元素，只验前 32 个元素等于没验 hi 半边
//       （docs/05-quantization.md §17.4）；且 qs 的起点是 p + 16，不是 p + 4。
// ============================================================================
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

// ============================================================================
// ① 定位：Q5_K 反量化——比 Q4_K 多一个「第 5 比特」，它存在独立的 qh[32] 里
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:1731:dequantize_row_q5_K
//    llama.cpp @ 88b47a755  ggml/src/ggml-common.h:344-355:block_q5_K（d / dmin / scales[12] / qh[32] / qs[128]）
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:1825-1826:quantize_row_q5_K_impl（写侧：
//                          uint8_t * qh = y[i].qh; uint8_t * ql = y[i].qs;）
//
// ③ 动机：
//    【概念】每个元素的量化值是 5 bit：低 4 bit 在 qs[128]（一字节两个元素），第 5 bit
//       另存 qh[32]（一字节八个元素，每字节 2 bit 表示一个元素的高位）。量化位不连续，
//       就得拆成两个数组存，读的时候再跨两个数组把同一个元素拼回来。
//    【陷阱】上游 block_q5_K 的字段顺序是 scales → qh → qs，本实现是 scales → qs → qh
//       （qh 取 p + 4 + 12 + 128）。**这是已知偏离**：字节序与上游相反，拿上游导出的
//       Q5_K GGUF 加载时，第 5 比特与低 4 bit 会互相串位。已登记为待拍板项
//       （docs/05-quantization.md §15 / §17.9），改法是调换两个指针的取址。
//    【对照】上游写侧与读侧共用一份结构体定义（字段顺序由编译器保证），本仓库没有这份
//       共享定义、同一偏移在两处各写一遍 —— 这正是上面那条偏离的成因。
//
// ④ demo + 多方样例：
//    🟢 合成块 d = 1.0、dmin = 0、scales[0] = 1、qs[l] = l、qh[l] 第 0 比特 = 1：
//       元素 0..31 → (l % 16) + 16.0（第 5 比特生效），元素 32..255 → 0
//       （tests/test_quant.cpp:124-143）
//    🟡 qh 全 0 时第 5 比特不生效，结果与 Q4_K 同构（lines/numeric/dequant_line.cpp 第 4 跳）
//    🔴 本实现的 qh 在 p + 144、上游在 p + 16：同一段字节两种解法，必须先定「以哪边为准」
//       再动代码（见【陷阱】段）
// ============================================================================
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

// ============================================================================
// ① 定位：Q6_K 反量化——有符号 6bit 量化（-32~31），无 dmin，块内切 16 个 16 元素的子块
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:1939:dequantize_row_q6_K
//    llama.cpp @ 88b47a755  ggml/src/ggml-common.h:362-367:block_q6_K（ql[128] / qh[64] / scales[16] / d）
//
// ③ 动机：
//    【陷阱】Q6_K 是**有符号**量化：解出来的 6 bit 值是 0~63，减 32 才是真正的量化值
//       （-32~31），公式是 d * sc * (v - 32)。没有 dmin —— 对称量化不需要偏移项，
//       零点的位置由「减 32」直接给出来。把它当无符号读，整块会整体偏 32 个量化级。
//    【概念】块内切 16 个 16 元素的子块（Q4_K / Q5_K 是 8 个 32 元素的子块）——切分维度
//       不同是 K-quants 的常态，测试预期必须按切分维度重算，不能按字节顺序线性推。
//    【陷阱】ql 一个字节横跨两行（lo nibble 给元素 l、hi nibble 给元素 l + 64），qh 一个
//       字节横跨四行（4 组 2 bit 分别给 l / l+32 / l+64 / l+96）——改一个字节会同时改多个元素。
//
// ④ demo + 多方样例：
//    🟢 合成块 ql = qh = 0 → 6bit 值 0，减 32 → -32；sc = 2、d = 0.25 → 全部元素 -16.0
//       （tests/test_quant.cpp:152-160）
//    🟡 再把 ql[0] 改成 0x21、qh[0] 改成 0x01：y[0] = -7.5，**同时** y[64] 变成 -15.0 ——
//       后者是 ql[0] 的 hi nibble 改了元素 64 的 q3，不是 bug（tests/test_quant.cpp:162-166）
//    🔴 元素 0 / 32 / 64 / 96 四个位置的取值来自同一批字节；只验第 0 个元素，hi nibble 与
//       qh 的 4 组 2 bit 里有 3 组从未被验过（docs/05-quantization.md §17.3）
// ============================================================================
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
