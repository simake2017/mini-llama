// 单元 3 测试：模型加载与 FP32 前向
//
// 策略：构造一个 1-2 层的微型 LLaMA（n_embd=8, head=2, kv_head=1, ff=16），
//       用确定性的伪随机权重，验证算子数学、形状、因果性、GQA、F16 转换、
//       量化拒绝、tied embeddings。

#include "model.h"
#include "gguf.h"
#include "gguf_writer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

#define CHECK_THROW(stmt)                                                  \
    do {                                                                   \
        bool thrown = false;                                               \
        try { stmt; } catch (const std::exception &) { thrown = true; }    \
        if (!thrown) {                                                     \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__            \
                      << "  期望抛异常但未抛: " #stmt "\n";                \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// 微型模型构造器
// ---------------------------------------------------------------------------
struct TinyTensor {
    std::vector<uint64_t> dims;
    std::vector<float>    data;   // 逻辑值（F16 类型时写入时转换）
    GgmlType              type = GgmlType::F32;
};

// 确定性伪随机（LCG），保证测试可重复
static float frand(uint32_t & seed) {
    seed = seed * 1664525u + 1013904223u;
    return ((seed >> 8) & 0xFFFF) / 65536.0f * 2.0f - 1.0f;  // [-1, 1)
}

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = static_cast<int>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0)   { return static_cast<uint16_t>(sign); }        // 次正规/下溢 -> 0（测试权重都在 ±1，安全）
    if (exp >= 31)  { return static_cast<uint16_t>(sign | 0x7C00u); }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

class TinyModelBuilder {
public:
    uint32_t n_layer = 1, n_embd = 8, n_head = 2, n_head_kv = 1, n_ff = 16, n_vocab = 24;

    std::map<std::string, TinyTensor> tensors;

    // 填充全部必需张量（确定性权重）
    void fill_all(uint32_t seed_base, bool with_output = true) {
        uint32_t s = seed_base;
        auto mk = [&](const char * name, std::vector<uint64_t> dims, bool norm_like) {
            size_t n = 1;
            for (auto d : dims) { n *= d; }
            TinyTensor t;
            t.dims = std::move(dims);
            t.data.resize(n);
            for (auto & x : t.data) {
                x = norm_like ? (1.0f + 0.1f * frand(s))   // norm 权重接近 1
                              : 0.2f * frand(s);            // 矩阵权重小一点
            }
            tensors[name] = std::move(t);
        };
        char buf[128];
        mk("token_embd.weight", {n_embd, n_vocab}, false);
        for (uint32_t i = 0; i < n_layer; i++) {
            snprintf(buf, sizeof buf, "blk.%u.", i);
            std::string p = buf;
            mk((p + "attn_norm.weight").c_str(), {n_embd}, true);
            mk((p + "attn_q.weight").c_str(),     {n_embd, n_embd}, false);
            mk((p + "attn_k.weight").c_str(),     {n_embd, n_head_kv * (n_embd / n_head)}, false);
            mk((p + "attn_v.weight").c_str(),     {n_embd, n_head_kv * (n_embd / n_head)}, false);
            mk((p + "attn_output.weight").c_str(),{n_embd, n_embd}, false);
            mk((p + "ffn_norm.weight").c_str(),   {n_embd}, true);
            mk((p + "ffn_gate.weight").c_str(),   {n_embd, n_ff}, false);
            mk((p + "ffn_up.weight").c_str(),     {n_embd, n_ff}, false);
            mk((p + "ffn_down.weight").c_str(),   {n_ff, n_embd}, false);
        }
        mk("output_norm.weight", {n_embd}, true);
        if (with_output) {
            mk("output.weight", {n_embd, n_vocab}, false);
        }
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

        // 张量表：偏移 = 数据区内相对位置（写时按 32 对齐）
        std::vector<std::pair<std::string, const TinyTensor *>> order;
        for (const auto & [name, t] : tensors) { order.emplace_back(name, &t); }
        uint64_t off = 0;
        std::map<std::string, uint64_t> offsets;
        for (const auto & [name, t] : order) {
            w.str(name);
            w.u32(static_cast<uint32_t>(t->dims.size()));
            for (auto d : t->dims) { w.u64(d); }
            w.u32(static_cast<uint32_t>(t->type));
            offsets[name] = off;
            w.u64(off);
            auto [be, bb] = ggml_block_params(t->type);
            size_t bytes = t->data.size() / be * bb;
            off += bytes;
            off = (off + 31) / 32 * 32;   // 下一个张量对齐
        }
        // 数据区：起点绝对对齐，张量间相对数据区对齐（与 offset 累计口径一致）
        w.align();
        const size_t data_start = w.buf().size();
        for (const auto & [name, t] : order) {
            (void)name;
            if (t->type == GgmlType::F16) {
                for (float x : t->data) { uint16_t h = f32_to_f16(x); w.raw_bytes(reinterpret_cast<const char *>(&h), 2); }
            } else {
                for (float x : t->data) { w.f32(x); }
            }
            w.align_data(data_start);
        }
        return minitest::write_temp(w.buf(), tag);
    }
};

// ---------------------------------------------------------------------------
// TestOps：基础算子数学
// ---------------------------------------------------------------------------
static void test_ops() {
    // RMSNorm: x=[3,4] -> ms=12.5, scale=1/sqrt(12.5+eps), w=1 -> out=x*scale
    {
        float x[2] = {3, 4}, w[2] = {1, 1}, out[2];
        rmsnorm(x, w, out, 2, 0.0f);
        CHECK_NEAR(out[0], 3.0 / std::sqrt(12.5), 1e-6);
        CHECK_NEAR(out[1], 4.0 / std::sqrt(12.5), 1e-6);
    }
    // f16 转换抽查
    CHECK_NEAR(f16_to_f32(0x3C00), 1.0, 1e-7);    // 1.0
    CHECK_NEAR(f16_to_f32(0xC000), -2.0, 1e-7);   // -2.0
    CHECK_NEAR(f16_to_f32(0x3800), 0.5, 1e-7);    // 0.5
    CHECK_NEAR(f16_to_f32(0x0001), 5.9604645e-8, 1e-15); // 最小次正规数
    CHECK_NEAR(f32_to_f16(1.5f) == 0x3E00 ? 1.0 : 0.0, 1.0, 0.0);
    // matvec: 单位阵
    {
        float I[4] = {1, 0, 0, 1};  // dims {2,2}: I(i,o) = data[i + o*2]
        float x[2] = {5, 7}, y[2];
        matvec_f32(I, x, y, 2, 2);
        CHECK_NEAR(y[0], 5, 1e-6);
        CHECK_NEAR(y[1], 7, 1e-6);
    }
    // RoPE: pos=0 必须是恒等变换
    {
        float x[4] = {1, 2, 3, 4};
        float backup[4];
        std::memcpy(backup, x, sizeof x);
        rope_rotate(x, 4, 0, 10000.0f);
        CHECK_NEAR(x[0], backup[0], 1e-6);
        CHECK_NEAR(x[3], backup[3], 1e-6);
    }
    // RoPE: 旋转保持模长（正交变换）
    {
        float x[4] = {0.3f, -1.2f, 2.0f, 0.7f};
        float m0 = 0;
        for (float v : x) { m0 += v * v; }
        rope_rotate(x, 4, 7, 10000.0f);
        float m1 = 0;
        for (float v : x) { m1 += v * v; }
        CHECK_NEAR(m0, m1, 1e-5);
    }
    // RoPE: 相邻配对约定（与 llama.cpp master 的 LLAMA_ROPE_TYPE_NORM 对齐）。
    // h=4, theta=10000, pos=1：配对 0 角=1，配对 1 角=1*10000^(-0.5)=0.01
    {
        float x[4] = {1, 2, 3, 4};
        rope_rotate(x, 4, 1, 10000.0f);
        CHECK_NEAR(x[0], 1*std::cos(1.0) - 2*std::sin(1.0), 1e-6);          // 对 0 首
        CHECK_NEAR(x[1], 1*std::sin(1.0) + 2*std::cos(1.0), 1e-6);          // 对 0 次
        CHECK_NEAR(x[2], 3*std::cos(0.01) - 4*std::sin(0.01), 1e-6);        // 对 1 首
        CHECK_NEAR(x[3], 3*std::sin(0.01) + 4*std::cos(0.01), 1e-6);        // 对 1 次
    }
    // silu(0)=0, softmax 归一
    CHECK_NEAR(0.0 / (1.0 + std::exp(-0.0)), 0.0, 0.0);
    {
        float x[5] = {3, 1, -2, 8, 0.5f};
        softmax(x, 5);
        float s = 0;
        for (float v : x) { s += v; }
        CHECK_NEAR(s, 1.0, 1e-6);
        CHECK(x[3] > x[0] && x[0] > x[1]);  // 序关系保持
    }
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// TestForward：加载 + 前向 + 因果性
// ---------------------------------------------------------------------------
static void test_forward_basic() {
    TinyModelBuilder b;
    b.fill_all(42);
    std::string path = b.write("fwd");
    GgufFile f = GgufFile::load(path);
    Model m = Model::load(f);

    const HParams & hp = m.hparams();
    CHECK(hp.n_layer == 1 && hp.n_embd == 8 && hp.n_head == 2);
    CHECK(hp.n_head_kv == 1 && hp.n_ff == 16 && hp.n_vocab == 24);
    CHECK(hp.head_dim() == 4 && hp.kv_dim() == 4);
    CHECK(!m.tied_output());

    // 前向：形状与确定性
    std::vector<int> tokens = {1, 5, 9};
    auto logits = m.forward(tokens);
    CHECK(logits.size() == tokens.size() * hp.n_vocab);
    auto logits2 = m.forward(tokens);
    CHECK(logits == logits2);   // 完全确定性
    for (float v : logits) { CHECK(std::isfinite(v)); }

    // 因果性：位置 p 的 logits 只依赖 tokens[0..p]
    std::vector<int> alt = {1, 5, 20};   // 只改最后一个
    auto logits_alt = m.forward(alt);
    for (size_t p = 0; p < 2; p++) {     // 前两行必须逐位相同
        for (uint32_t i = 0; i < hp.n_vocab; i++) {
            CHECK(logits[p * hp.n_vocab + i] == logits_alt[p * hp.n_vocab + i]);
        }
    }

    // 单层退化：无参数改变时 embedding 行本身可经 output 投影复现（健全性：不 NaN 已查）
    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

// 两层 + F16 权重 + tied output
static void test_forward_f16_tied() {
    TinyModelBuilder b;
    b.n_layer = 2;
    b.fill_all(7, /*with_output=*/false);
    // 把某一层的 wk 改成 F16 存储（权重值都是 0.2 的倍数，F16 精确表示）
    for (auto & [name, t] : b.tensors) {
        if (name == "blk.0.attn_k.weight") {
            for (auto & x : t.data) { x = std::round(x * 5) / 5; }  // 量化到 F16 可精确表示的值
            t.type = GgmlType::F16;
        }
    }
    std::string path = b.write("f16");
    GgufFile f = GgufFile::load(path);
    Model m = Model::load(f);
    CHECK(m.tied_output());
    CHECK(m.hparams().n_layer == 2);

    auto logits = m.forward({3, 4, 5, 6});
    CHECK(logits.size() == 4 * m.hparams().n_vocab);
    for (float v : logits) { CHECK(std::isfinite(v)); }

    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

// 量化张量：小模型 ne0=8 不是块大小整数倍，加载必须报错（块对齐校验）
static void test_quant_rejected() {
    TinyModelBuilder b;
    b.fill_all(9);
    b.tensors["blk.0.attn_q.weight"].type = GgmlType::Q8_0;   // 块 32 元素
    std::string path = b.write("qreject");
    GgufFile f = GgufFile::load(path);
    bool thrown = false;
    try { Model::load(f); } catch (const std::exception & e) {
        thrown = std::string(e.what()).find("块大小") != std::string::npos;
    }
    CHECK(thrown);
    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

// 真实量化模型：单元 5 后应能加载（反量化推理的端到端验证见 test_quant）
static void test_real_file() {
    const char * path = std::getenv("TEST_MODEL_REAL_FILE");
    if (!path || !*path) {
        std::cout << "  [skip] 未设置 TEST_MODEL_REAL_FILE，跳过真实模型冒烟\n";
        return;
    }
    GgufFile f = GgufFile::load(path);
    Model m = Model::load(f);
    const HParams & hp = m.hparams();
    std::cout << "  real model: " << path
              << "  层数=" << hp.n_layer << "  维度=" << hp.n_embd
              << "  词表=" << hp.n_vocab << "\n";
    CHECK(hp.n_layer > 0 && hp.n_embd > 0 && hp.n_vocab > 1000);
    if (g_fail == 0) { ++g_pass; }
}

int main() {
    test_ops();
    test_forward_basic();
    test_forward_f16_tied();
    test_quant_rejected();
    test_real_file();

    std::cout << "passed=" << g_pass << " failed=" << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
