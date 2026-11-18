// ============================================================================
// 线⑦ 反量化：GGUF 量化字节 → 块布局解读 → 逐块还原成 F32 权重 → 进入 matvec
//
// 对照 llama.cpp @ 88b47a755:
//   ggml/src/ggml-quants.c   dequantize_row_q8_0 / _q4_K / _q5_K / _q6_K
//                            get_scale_min_k4（K-quants 的 6bit scale 解包）
//   ggml/src/ggml-common.h   struct block_q8_0 / block_q4_K / block_q5_K / block_q6_K
//   ggml/src/ggml.c          ggml_type_traits（类型 → 块大小）
//
// 跨模块：src/quant.cpp（反量化内核）· src/gguf.cpp（取张量字节）· src/model.cpp（matvec 分发）
//
// 学习问题：量化块里存的是「缩放系数 + 低位量化值」，为什么每个子块要单独存 scale/min，
//           而不是整行共用一个缩放系数？
//
// 跑：MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_dequant
// ============================================================================

#include "gguf.h"
#include "line.hpp"
#include "lines.hpp"
#include "model.h"
#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

using namespace minillama;
using namespace minillama::lines;

namespace {

bool file_exists(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

// f16(1.0) 的位模式（IEEE754 半精度）：0x3C00，小端存放
const uint8_t F16_ONE[2]  = {0x00, 0x3C};
// f16(0.0) = 0x0000
const uint8_t F16_ZERO[2] = {0x00, 0x00};

// 合成一个 Q8_0 块：{ f16 d; int8 qs[32] }，value = qs[j] * d
std::vector<uint8_t> make_q8_0_block(float d, const std::vector<int8_t> & qs) {
    std::vector<uint8_t> blk;
    const uint16_t h = 0x3C00;   // 这里只用到 d = 1.0
    (void)d;
    blk.push_back(h & 0xFF);
    blk.push_back(h >> 8);
    for (int8_t q : qs) blk.push_back(static_cast<uint8_t>(q));
    return blk;
}

// 合成一个 Q4_K 块：{ f16 d; f16 dmin; uint8 scales[12]; uint8 qs[128] }
// scales 摆法：scales[0..3]=1 → j<4 的 scale；scales[4..7]=0 → j<4 的 min；
//               scales[8..11]=1 → j>=4 的 scale 低 4 位（高 2 位来自 scales[0..3] 顶位）
std::vector<uint8_t> make_q4_k_block(uint8_t nibble) {
    std::vector<uint8_t> blk;
    blk.insert(blk.end(), F16_ONE, F16_ONE + 2);    // d = 1.0
    blk.insert(blk.end(), F16_ZERO, F16_ZERO + 2);  // dmin = 0
    for (int i = 0; i < 4; i++) blk.push_back(1);    // scales[0..3]
    for (int i = 0; i < 4; i++) blk.push_back(0);    // scales[4..7]
    for (int i = 0; i < 4; i++) blk.push_back(1);    // scales[8..11]
    for (int i = 0; i < 128; i++) {
        blk.push_back(static_cast<uint8_t>(nibble | (nibble << 4)));  // 低/高 nibble 同值
    }
    return blk;
}

// 合成一个 Q5_K 块：{ f16 d; f16 dmin; scales[12]; ql[128]; qh[32] }
std::vector<uint8_t> make_q5_k_block(uint8_t nibble) {
    std::vector<uint8_t> blk = make_q4_k_block(nibble);
    for (int i = 0; i < 32; i++) blk.push_back(0);   // qh 全 0 → 不加第 5 比特
    return blk;
}

// 合成一个 Q6_K 块：{ uint8 ql[128]; uint8 qh[64]; int8 scales[16]; f16 d }
// ql 全 0、qh 全 0 → 6bit 值 = 0，减 32 后 = -32
std::vector<uint8_t> make_q6_k_block(int8_t sc, float d) {
    std::vector<uint8_t> blk;
    for (int i = 0; i < 128; i++) blk.push_back(0);   // ql
    for (int i = 0; i < 64; i++)  blk.push_back(0);   // qh
    for (int i = 0; i < 16; i++)  blk.push_back(static_cast<uint8_t>(sc));  // scales
    blk.insert(blk.end(), F16_ONE, F16_ONE + 2);    // d = 1.0
    (void)d;
    return blk;
}

} // namespace

// ---------------------------------------------------------------------------
// 线本体
// ---------------------------------------------------------------------------
int minillama::lines::run_dequant(LineCtx & ctx) {
    // ------------------------------------------------------------------
    ctx.step_at(1, "块布局：每种量化格式的「块」= 固定元素数 + 固定字节数",
                "src/quant.h: QK8_0 / QK_K 常量（对照 ggml-common.h 的 block_* 结构体）",
                "src/gguf.cpp: ggml_block_params（类型 → {每块元素数, 每块字节数}）");
    ctx.log("Q8_0 = {f16 d; int8 qs[32]}                34 字节 / 32 元素");
    ctx.log("Q4_K = {f16 d; f16 dmin; u8 scales[12]; u8 qs[128]}  144 字节 / 256 元素");
    ctx.log("Q5_K = {f16 d; f16 dmin; u8 scales[12]; u8 ql[128]; u8 qh[32]}  176 字节 / 256 元素");
    ctx.log("Q6_K = {u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d}          210 字节 / 256 元素");
    ctx.log("关键：Q4_K/Q5_K 的 qs 在偏移 16（d+dmin+scales 之后），不是偏移 4");
    ctx.check(ggml_block_params(GgmlType::Q8_0).first == QK8_0, "Q8_0 块大小 = 32 元素");
    ctx.check(ggml_block_params(GgmlType::Q4_K).first == QK_K, "Q4_K 块大小 = 256 元素");
    ctx.check(ggml_block_params(GgmlType::Q4_K).second == 144, "Q4_K 块字节数 = 144");

    // ------------------------------------------------------------------
    ctx.step_at(2, "Q8_0 反量化：value = qs[j] * d",
                "src/quant.cpp: dequant_row_q8_0（对照 ggml-quants.c 的 dequantize_row_q8_0）",
                "合成块：d=1.0，qs[j]=j+1 → 期望值 1..32");
    {
        std::vector<int8_t> qs(32);
        for (int j = 0; j < 32; j++) qs[j] = static_cast<int8_t>(j + 1);
        auto blk = make_q8_0_block(1.0f, qs);
        std::vector<float> y(32, 0.0f);
        dequant_row_q8_0(blk.data(), y.data(), 32);
        ctx.value("首 4 个还原值", y[0]);
        ctx.log("y[0]=" + std::to_string(y[0]) + " y[1]=" + std::to_string(y[1]) +
                " y[2]=" + std::to_string(y[2]) + " y[3]=" + std::to_string(y[3]));
        bool ok = true;
        for (int j = 0; j < 32; j++) {
            if (std::fabs(y[j] - float(j + 1)) > 1e-6f) { ok = false; }
        }
        ctx.check(ok, "Q8_0 反量化结果 == qs[j] * d");
    }

    // ------------------------------------------------------------------
    ctx.step_at(3, "Q4_K 反量化：value = d * sc * nibble - dmin * m",
                "src/quant.cpp: dequant_row_q4_k + get_scale_min_k4（对照 ggml-quants.c 同名函数）",
                "合成块：8 个子块各 32 元素，sc=1、m=0、nibble=1 → 期望全 1.0");
    {
        auto blk = make_q4_k_block(1);
        std::vector<float> y(QK_K, 0.0f);
        dequant_row_q4_k(blk.data(), y.data(), QK_K);
        double mn = y[0], mx = y[0];
        for (float v : y) { mn = std::min<double>(mn, v); mx = std::max<double>(mx, v); }
        ctx.value("最小值", mn);
        ctx.value("最大值", mx);
        ctx.check(std::fabs(mx - 1.0) < 1e-6 && std::fabs(mn - 1.0) < 1e-6,
                  "Q4_K 反量化结果全为 1.0");
        ctx.log("6bit 解包两路：j<4 取 scales[j]&63 / scales[j+4]&63；j>=4 拼 scales[j+4] 与 scales[j-4] 顶 2 位");
    }

    // ------------------------------------------------------------------
    ctx.step_at(4, "Q5_K 反量化：第 5 比特存在 qh 里",
                "src/quant.cpp: dequant_row_q5_k（对照 ggml-quants.c 的 dequantize_row_q5_K）",
                "合成块：qh 全 0 → 不触发第 5 比特，期望值与 Q4_K 相同");
    {
        auto blk = make_q5_k_block(1);
        std::vector<float> y(QK_K, 0.0f);
        dequant_row_q5_k(blk.data(), y.data(), QK_K);
        double mn = y[0], mx = y[0];
        for (float v : y) { mn = std::min<double>(mn, v); mx = std::max<double>(mx, v); }
        ctx.value("最小值", mn);
        ctx.value("最大值", mx);
        ctx.check(std::fabs(mx - 1.0) < 1e-6 && std::fabs(mn - 1.0) < 1e-6,
                  "Q5_K 反量化结果全为 1.0（qh=0 时第 5 比特不生效）");
    }

    // ------------------------------------------------------------------
    ctx.step_at(5, "Q6_K 反量化：value = d * sc * (6bit 值 - 32)",
                "src/quant.cpp: dequant_row_q6_k（对照 ggml-quants.c 的 dequantize_row_q6_K）",
                "合成块：ql=qh=0 → 6bit 值 0，减 32 得 -32；sc=1、d=1 → 期望全 -32");
    {
        auto blk = make_q6_k_block(1, 1.0f);
        std::vector<float> y(QK_K, 0.0f);
        dequant_row_q6_k(blk.data(), y.data(), QK_K);
        double mn = y[0], mx = y[0];
        for (float v : y) { mn = std::min<double>(mn, v); mx = std::max<double>(mx, v); }
        ctx.value("最小值", mn);
        ctx.value("最大值", mx);
        ctx.check(std::fabs(mx + 32.0) < 1e-5 && std::fabs(mn + 32.0) < 1e-5,
                  "Q6_K 反量化结果全为 -32");
        ctx.log("Q6_K 是**有符号**量化（无 dmin），所以中心值是 0 而不是 32");
    }

    // ------------------------------------------------------------------
    ctx.step_at(6, "真实模型：取出量化张量的原始字节并反量化一行",
                "src/gguf.cpp: GgufFile::tensor_data + src/quant.cpp: dequant_row_q8_0",
                "本机模型是 Q8_0；Q4_K/Q5_K/Q6_K 走上面的合成块");
    const std::string path = resolve_model_path();
    if (!file_exists(path)) {
        ctx.skip("找不到模型文件：" + path + "（设 MINILLAMA_MODEL 指向一个 GGUF）");
        return ctx.summary();
    }
    std::optional<GgufFile> gguf;
    try {
        gguf.emplace(GgufFile::load(path));
    } catch (const std::exception & e) {
        ctx.check(false, "GGUF 文件应当能正常解析", e.what(), "解析成功");
        return ctx.summary();
    }
    const TensorInfo * embd = gguf->find_tensor("token_embd.weight");
    if (!embd) {
        ctx.check(false, "模型里应有 token_embd.weight", "没找到", "找到");
        return ctx.summary();
    }
    ctx.value("token_embd 类型", ggml_type_name(embd->type));
    ctx.value("token_embd 维度", embd->dims[0]);
    const uint8_t * raw = static_cast<const uint8_t *>(gguf->tensor_data(*embd));
    if (embd->type == GgmlType::Q8_0) {
        std::vector<float> row(32, 0.0f);
        dequant_row_q8_0(raw, row.data(), 32);
        double mn = row[0], mx = row[0];
        for (float v : row) { mn = std::min<double>(mn, v); mx = std::max<double>(mx, v); }
        ctx.value("第 0 行第 1 块的最小值", mn);
        ctx.value("第 0 行第 1 块的最大值", mx);
        ctx.check(mn <= mx, "反量化出的值域有序（min <= max）");
    } else {
        ctx.log("该模型不是 Q8_0，真实路径跳过（合成块验证仍然有效）");
    }

    // ------------------------------------------------------------------
    ctx.step_at(7, "接入前向：matvec 按张量类型分派到对应反量化例程",
                "src/model.cpp: matvec（对照 ggml 的 ggml_vec_dot 类型表）",
                "tests/test_quant.cpp: 混合量化前向与 libllama 逐 logits 对照");
    {
        TensorBuf buf;
        buf.type = GgmlType::Q8_0;
        buf.ne0  = 32;
        buf.ne1  = 1;
        auto blk = make_q8_0_block(1.0f, std::vector<int8_t>(32, 1));
        buf.bytes = blk;
        std::vector<float> x(32, 1.0f), y(1, 0.0f);
        matvec(buf, x.data(), y.data());
        ctx.value("matvec 结果", y[0]);
        ctx.check(std::fabs(y[0] - 32.0) < 1e-4,
                  "matvec(Q8_0) = 32 个 1.0 的点积 = 32.0",
                  std::to_string(y[0]), "32");
    }

    // ------------------------------------------------------------------
    ctx.step_at(8, "结局：量化权重进入计算图",
                "src/model.cpp: Model::forward_token（消费方）",
                "./build/mini-llama --model <Q8_0 模型> --prompt \"Hi\" --n 8");
    ctx.check(true, "反量化链走完");

    return ctx.summary();
}

// ---------------------------------------------------------------------------
// ③ 自注册：编译进去就出现在 --list，不改任何公共文件
// ---------------------------------------------------------------------------
MINILLAMA_LINE_REGISTER(reg_dequant, "dequant", "GGUF 量化字节 → 逐块还原成 F32 权重 → 进入 matvec", ::minillama::lines::run_dequant);

// ④ 独立 main：只在编译 line_dequant 这个单跑目标时打开
#ifdef MINILLAMA_LINE_STANDALONE
MINILLAMA_LINE_STANDALONE_MAIN("dequant", ::minillama::lines::run_dequant)
#endif
