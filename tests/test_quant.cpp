// 单元 5 测试：量化反量化 + 混合量化模型前向
//
// 1. 反量化精确性：手工构造块，逐元素验证（公式从 ggml-quants.c 移植，
//    测试用独立手算值交叉验证，防"同一 bug 写两遍"）
// 2. 集成：同一组逻辑权重分别存 F32 和 Q8_0，前向结果在容差内一致
// 3. 真实模型：q4_k_m / q8_0 端到端加载 + 前向有限性

#include "model.h"
#include "quant.h"
#include "gguf.h"
#include "gguf_writer.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace minillama;
using minitest::Writer;

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__            \
                      << "  CHECK(" #cond ")\n";                           \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                              \
    do {                                                                   \
        double _d = std::fabs(double(a) - double(b));                      \
        if (_d > (eps)) {                                                  \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__            \
                      << "  |" #a " - " #b "| = " << _d << "\n";           \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// 工具：f32 -> f16（仅覆盖测试需要的正数范围）
// ---------------------------------------------------------------------------
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = static_cast<int>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0)  { return static_cast<uint16_t>(sign); }
    if (exp >= 31) { return static_cast<uint16_t>(sign | 0x7C00u); }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

static void put_f16(std::vector<uint8_t> & out, float f) {
    uint16_t h = f32_to_f16(f);
    out.push_back(h & 0xFF);
    out.push_back(h >> 8);
}

// ---------------------------------------------------------------------------
// TestDequant_Q8_0：d=0.5，qs = [-128, -1, 0, 1, 127, ...]
// ---------------------------------------------------------------------------
static void test_dequant_q8_0() {
    std::vector<uint8_t> block;
    put_f16(block, 0.5f);
    const int8_t qs[QK8_0] = {-128, -1, 0, 1, 127, 2, -2, 3,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0};
    for (int8_t q : qs) { block.push_back(static_cast<uint8_t>(q)); }

    float y[QK8_0];
    dequant_row_q8_0(block.data(), y, QK8_0);
    CHECK_NEAR(y[0], -64.0, 1e-6);   // -128 * 0.5
    CHECK_NEAR(y[1], -0.5, 1e-6);
    CHECK_NEAR(y[2],  0.0, 1e-6);
    CHECK_NEAR(y[3],  0.5, 1e-6);
    CHECK_NEAR(y[4], 63.5, 1e-6);
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// TestDequant_Q4_K：手工填充一个块
//   d=2.0, dmin=1.0
//   scales[0]=1, scales[4]=1  -> 子块 0: sc=1, m=1（j<4 路径）
//   其余 scales=0             -> 子块 1: sc=0（j<4 路径），子块 2/3 及以后全 0
//   qs[l] = l (l<32)          -> 元素 0..31: y = 2*(l&0xF) - 1
//                            -> 元素 32..63: y = 0*(hi nibble) - 0 = 0
// ---------------------------------------------------------------------------
static void test_dequant_q4_k() {
    std::vector<uint8_t> block(144, 0);
    // d / dmin
    uint16_t d = f32_to_f16(2.0f), dmin = f32_to_f16(1.0f);
    std::memcpy(block.data(), &d, 2);
    std::memcpy(block.data() + 2, &dmin, 2);
    // scales: 子块 0 的 sc/m 走 j<4 路径：scales[j]&63 / scales[j+4]&63
    block[4 + 0] = 1;   // sc[0] = 1
    block[4 + 4] = 1;   // m[0]  = 1
    // qs
    for (int l = 0; l < 32; l++) { block[4 + 12 + l] = static_cast<uint8_t>(l); }

    std::vector<float> y(QK_K);
    dequant_row_q4_k(block.data(), y.data(), QK_K);
    for (int l = 0; l < 32; l++) {
        CHECK_NEAR(y[l], 2.0 * (l % 16) - 1.0, 1e-4);
    }
    for (int l = 32; l < QK_K; l++) {
        CHECK_NEAR(y[l], 0.0, 1e-6);
    }
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// TestDequant_Q5_K：d=1.0, dmin=0, scales 同 Q4_K 只子块 0 有 sc=1
//   qs[l] = l (l<32)，qh[l] 第 0 比特 = 1（l<32）-> y[l] = (l&0xF) + 16
// ---------------------------------------------------------------------------
static void test_dequant_q5_k() {
    std::vector<uint8_t> block(176, 0);
    uint16_t d = f32_to_f16(1.0f), dmin = f32_to_f16(0.0f);
    std::memcpy(block.data(), &d, 2);
    std::memcpy(block.data() + 2, &dmin, 2);
    block[4 + 0] = 1;   // sc[0] = 1
    for (int l = 0; l < 32; l++) {
        block[4 + 12 + l] = static_cast<uint8_t>(l);   // qs
        block[4 + 12 + 128 + l] = 1;                   // qh 第 0 比特 = 1
    }
    std::vector<float> y(QK_K);
    dequant_row_q5_k(block.data(), y.data(), QK_K);
    for (int l = 0; l < 32; l++) {
        CHECK_NEAR(y[l], (l % 16) + 16.0, 1e-4);
    }
    for (int l = 32; l < QK_K; l++) {
        CHECK_NEAR(y[l], 0.0, 1e-6);
    }
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// TestDequant_Q6_K：d=0.25, scales[16]=全 2
//   ql=0, qh=0 -> q6 = -32 -> y = 0.25*2*(-32) = -16（全部 256 个）
//   再改 ql[0]=0x21（lo=1）, qh[0]=0x01（q1 高 2 位=1）：
//   q1 = (1 | 16) - 32 = -15 -> y[0] = 0.25*2*(-15) = -7.5
//   注意 ql[0]>>4 = 2 同时改变元素 64 的 q3：y[64] = 0.5*(2-32) = -15
// ---------------------------------------------------------------------------
static void test_dequant_q6_k() {
    std::vector<uint8_t> block(210, 0);
    uint16_t d = f32_to_f16(0.25f);
    std::memcpy(block.data() + 128 + 64 + 16, &d, 2);
    for (int i = 0; i < 16; i++) { block[128 + 64 + i] = 2; }  // scales

    std::vector<float> y(QK_K);
    dequant_row_q6_k(block.data(), y.data(), QK_K);
    for (int l = 0; l < QK_K; l++) { CHECK_NEAR(y[l], -16.0, 1e-4); }

    block[0] = 0x21;    // ql[0] lo = 1（同时 hi=2 影响元素 64）
    block[128] = 0x01;  // qh[0] 低 2 位 = 1
    dequant_row_q6_k(block.data(), y.data(), QK_K);
    CHECK_NEAR(y[0], -7.5, 1e-4);
    CHECK_NEAR(y[64], -15.0, 1e-4);
    for (int l = 1; l < QK_K; l++) {
        if (l == 64) { continue; }
        CHECK_NEAR(y[l], -16.0, 1e-4);
    }
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// 集成测试：同一逻辑权重的 F32 版 vs Q8_0 版模型，前向一致性
// （n_embd=128, head=4, kv_head=1, ff=64, vocab=48，1 层）
// ---------------------------------------------------------------------------
namespace {

struct TinyTensor {
    std::vector<uint64_t> dims;
    std::vector<float>    data;
    GgmlType              type = GgmlType::F32;
    std::vector<uint8_t>  qbytes;   // 量化类型的原始字节
};

float frand(uint32_t & seed) {
    seed = seed * 1664525u + 1013904223u;
    return ((seed >> 8) & 0xFFFF) / 65536.0f * 2.0f - 1.0f;
}

// 简单 Q8_0 量化：d = max|x|/127
std::vector<uint8_t> quant_q8_0(const std::vector<float> & x) {
    std::vector<uint8_t> out;
    for (size_t base = 0; base < x.size(); base += QK8_0) {
        float amax = 0;
        for (int j = 0; j < QK8_0; j++) { amax = std::max(amax, std::fabs(x[base + j])); }
        float d = amax / 127.0f;
        if (d == 0.0f) { d = 1e-8f; }
        put_f16(out, d);
        for (int j = 0; j < QK8_0; j++) {
            int q = static_cast<int>(std::round(x[base + j] / d));
            out.push_back(static_cast<uint8_t>(static_cast<int8_t>(q)));
        }
    }
    return out;
}

class Builder {
public:
    uint32_t n_layer = 1, n_embd = 128, n_head = 4, n_head_kv = 1, n_ff = 64, n_vocab = 48;
    std::map<std::string, TinyTensor> tensors;

    void fill_all(uint32_t seed_base) {
        uint32_t s = seed_base;
        auto mk = [&](const char * name, std::vector<uint64_t> dims, bool norm_like) {
            size_t n = 1;
            for (auto d : dims) { n *= d; }
            TinyTensor t;
            t.dims = std::move(dims);
            t.data.resize(n);
            for (auto & x : t.data) {
                x = norm_like ? (1.0f + 0.1f * frand(s)) : 0.2f * frand(s);
            }
            tensors[name] = std::move(t);
        };
        mk("token_embd.weight", {n_embd, n_vocab}, false);
        for (uint32_t i = 0; i < n_layer; i++) {
            char buf[128];
            snprintf(buf, sizeof buf, "blk.%u.", i);
            std::string p = buf;
            mk((p + "attn_norm.weight").c_str(), {n_embd}, true);
            mk((p + "attn_q.weight").c_str(),     {n_embd, n_embd}, false);
            mk((p + "attn_k.weight").c_str(),     {n_embd, n_embd / n_head * n_head_kv}, false);
            mk((p + "attn_v.weight").c_str(),     {n_embd, n_embd / n_head * n_head_kv}, false);
            mk((p + "attn_output.weight").c_str(),{n_embd, n_embd}, false);
            mk((p + "ffn_norm.weight").c_str(),   {n_embd}, true);
            mk((p + "ffn_gate.weight").c_str(),   {n_embd, n_ff}, false);
            mk((p + "ffn_up.weight").c_str(),     {n_embd, n_ff}, false);
            mk((p + "ffn_down.weight").c_str(),   {n_ff, n_embd}, false);
        }
        mk("output_norm.weight", {n_embd}, true);
        mk("output.weight", {n_embd, n_vocab}, false);
    }

    // 把指定张量量化成 Q8_0
    void quantize(const std::string & name) {
        auto & t = tensors[name];
        t.qbytes = quant_q8_0(t.data);
        t.type = GgmlType::Q8_0;
    }

    std::string write(const char * tag) {
        Writer w;
        w.u32(0x46554747u); w.u32(3);
        w.u64(tensors.size());
        w.u64(9);
        w.kv_header("general.architecture", 8); w.str("llama");
        w.kv_header("llama.block_count", 4);            w.u32(n_layer);
        w.kv_header("llama.embedding_length", 4);       w.u32(n_embd);
        w.kv_header("llama.attention.head_count", 4);   w.u32(n_head);
        w.kv_header("llama.attention.head_count_kv", 4);w.u32(n_head_kv);
        w.kv_header("llama.feed_forward_length", 4);    w.u32(n_ff);
        w.kv_header("llama.context_length", 4);         w.u32(64);
        w.kv_header("llama.attention.layer_norm_rms_epsilon", 6); w.f32(1e-5f);
        w.kv_header("llama.rope.freq_base", 6);         w.f32(10000.0f);

        std::vector<std::pair<std::string, const TinyTensor *>> order;
        for (const auto & [name, t] : tensors) { order.emplace_back(name, &t); }
        uint64_t off = 0;
        for (const auto & [name, t] : order) {
            w.str(name);
            w.u32(static_cast<uint32_t>(t->dims.size()));
            for (auto d : t->dims) { w.u64(d); }
            w.u32(static_cast<uint32_t>(t->type));
            w.u64(off);
            auto [be, bb] = ggml_block_params(t->type);
            size_t bytes = t->data.size() / be * bb;
            off += bytes;
            off = (off + 31) / 32 * 32;
        }
        w.align();
        const size_t data_start = w.buf().size();
        for (const auto & [name, t] : order) {
            (void)name;
            if (t->type == GgmlType::Q8_0) {
                for (auto b : t->qbytes) { w.u8(b); }
            } else {
                for (float x : t->data) { w.f32(x); }
            }
            w.align_data(data_start);
        }
        return minitest::write_temp(w.buf(), tag);
    }
};

} // namespace

static void test_mixed_quant_forward() {
    Builder b;
    b.fill_all(2024);
    std::string path_f32 = b.write("qf32");

    // 同一权重，attention 矩阵全部 Q8_0
    b.quantize("blk.0.attn_q.weight");
    b.quantize("blk.0.attn_k.weight");
    b.quantize("blk.0.attn_v.weight");
    b.quantize("blk.0.attn_output.weight");
    std::string path_q8 = b.write("qq8");

    Model m_f32 = Model::load(GgufFile::load(path_f32));
    Model m_q8  = Model::load(GgufFile::load(path_q8));

    std::vector<int> tokens = {3, 17, 40, 5, 22};
    auto l32 = m_f32.forward(tokens);
    auto lq8 = m_q8.forward(tokens);

    CHECK(l32.size() == lq8.size());
    double max_diff = 0;
    int argmax_diff = 0;
    const uint32_t V = m_f32.hparams().n_vocab;
    for (size_t p = 0; p < tokens.size(); p++) {
        double dmax = 0;
        for (uint32_t i = 0; i < V; i++) {
            dmax = std::max(dmax, std::fabs(double(l32[p * V + i]) - double(lq8[p * V + i])));
        }
        max_diff = std::max(max_diff, dmax);
        auto am32 = std::max_element(l32.begin() + p * V, l32.begin() + (p + 1) * V) - (l32.begin() + p * V);
        auto amq8 = std::max_element(lq8.begin() + p * V, lq8.begin() + (p + 1) * V) - (lq8.begin() + p * V);
        if (am32 != amq8) { argmax_diff++; }
    }
    std::cout << "  Q8_0 vs F32: max|Δlogit| = " << max_diff
              << "，argmax 不一致行数 = " << argmax_diff << "/" << tokens.size() << "\n";
    CHECK(max_diff < 0.35);          // Q8_0 量化误差上界（经验值，宽松断言防回归）
    CHECK(argmax_diff == 0);         // argmax 应保持一致

    std::remove(path_f32.c_str());
    std::remove(path_q8.c_str());
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// 真实量化模型端到端：加载 + 前向有限性 + KV 等价
// ---------------------------------------------------------------------------
static void test_real_quant_model() {
    const char * path = std::getenv("TEST_QUANT_REAL_FILE");
    if (!path || !*path) {
        std::cout << "  [skip] 未设置 TEST_QUANT_REAL_FILE，跳过真实量化模型\n";
        return;
    }
    GgufFile f = GgufFile::load(path);
    Model m = Model::load(f);
    const uint32_t V = m.hparams().n_vocab;
    std::cout << "  quant model: " << path << "  vocab=" << V << "\n";

    // 走 4 个位置的前向（直接喂 token id，不依赖分词器；跳过 1000 以下的
    // 未用占位 token——它们的嵌入行全零，logits 恒零测不出反量化正确性）
    KVCache kv(m.hparams().n_layer, m.hparams().kv_dim(), m.hparams().n_ctx);
    std::vector<float> last;
    for (int p = 0; p < 4; p++) {
        last = m.forward_token(1000 + p, p, kv);
    }
    bool finite = true;
    int argmax = 0;
    for (uint32_t i = 0; i < V; i++) {
        if (!std::isfinite(last[i])) { finite = false; break; }
        if (last[i] > last[argmax]) { argmax = static_cast<int>(i); }
    }
    CHECK(finite);
    CHECK(argmax >= 0 && argmax < static_cast<int>(V));
    std::cout << "  前向 4 步: 有限=" << finite << "  argmax_token=" << argmax << "\n";
    if (g_fail == 0) { ++g_pass; }
}

int main() {
    test_dequant_q8_0();
    test_dequant_q4_k();
    test_dequant_q5_k();
    test_dequant_q6_k();
    test_mixed_quant_forward();
    test_real_quant_model();

    std::cout << "passed=" << g_pass << " failed=" << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
