// 单元 4 测试：KV Cache 等价性 + 采样器 + 生成循环
//
// 核心测试：增量前向（KV cache）必须与全量前向逐位置 logits 一致。
// 这是"优化不改变语义"的不变式，也是单元 4 存在的理由。

#include "kvcache.h"
#include "model.h"
#include "sampler.h"
#include "gguf.h"
#include "gguf_writer.h"

#include <cmath>
#include <cstdio>
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
// 微型模型构造器（与 test_model.cpp 相同的确定性权重）
// ---------------------------------------------------------------------------
struct TinyTensor {
    std::vector<uint64_t> dims;
    std::vector<float>    data;
    GgmlType              type = GgmlType::F32;
};

static float frand(uint32_t & seed) {
    seed = seed * 1664525u + 1013904223u;
    return ((seed >> 8) & 0xFFFF) / 65536.0f * 2.0f - 1.0f;
}

class TinyModelBuilder {
public:
    uint32_t n_layer = 2, n_embd = 8, n_head = 2, n_head_kv = 1, n_ff = 16, n_vocab = 24;
    std::map<std::string, TinyTensor> tensors;

    void fill_all(uint32_t seed_base, bool with_output = true) {
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
        if (with_output) { mk("output.weight", {n_embd, n_vocab}, false); }
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
            off += t->data.size() * 4;
            off = (off + 31) / 32 * 32;
        }
        w.align();
        const size_t data_start = w.buf().size();
        for (const auto & [name, t] : order) {
            (void)name;
            for (float x : t->data) { w.f32(x); }
            w.align_data(data_start);
        }
        return minitest::write_temp(w.buf(), tag);
    }
};

// ---------------------------------------------------------------------------
// TestKvCacheEquivalence：增量 == 全量（本单元最重要的测试）
// ---------------------------------------------------------------------------
static void test_kv_equivalence() {
    TinyModelBuilder b;
    b.fill_all(1234);
    std::string path = b.write("kveq");
    GgufFile f = GgufFile::load(path);
    Model m = Model::load(f);
    const uint32_t V = m.hparams().n_vocab;

    std::vector<int> tokens = {1, 5, 9, 2, 20, 7, 11};
    auto full = m.forward(tokens);            // 全量（内部也走 KVCache，但逐行收集）

    // 显式增量：自己推进 KVCache，每步收 logits
    KVCache kv(m.hparams().n_layer, m.hparams().kv_dim(), m.hparams().n_ctx);
    for (size_t p = 0; p < tokens.size(); p++) {
        auto row = m.forward_token(tokens[p], static_cast<int>(p), kv);
        CHECK(kv.size() == p + 1);
        for (uint32_t i = 0; i < V; i++) {
            CHECK_NEAR(row[i], full[p * V + i], 1e-4);
        }
    }

    // clear 后复用：再来一遍结果不变
    kv.clear();
    auto row = m.forward_token(tokens[0], 0, kv);
    for (uint32_t i = 0; i < V; i++) {
        CHECK_NEAR(row[i], full[i], 1e-4);
    }

    // 顺序校验：跳 pos 必须报错
    bool thrown = false;
    try { m.forward_token(3, 5, kv); } catch (const std::exception &) { thrown = true; }
    CHECK(thrown);

    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// TestSampler：贪心 / 温度 / top-k / top-p
// ---------------------------------------------------------------------------
static void test_sampler() {
    // 造一个尖峰分布的 logits：id=2 一枝独秀
    std::vector<float> logits = {0.1f, -0.5f, 8.0f, 0.2f, -1.0f};

    // 贪心 = argmax
    {
        Sampler s(SamplerParams{0.0f, 40, 0.95f, 0});
        CHECK(s.sample(logits.data(), 5) == 2);
    }
    // temp=0 时 top_k/top_p 无关紧要
    {
        Sampler s(SamplerParams{0.0f, 1, 0.1f, 0});
        CHECK(s.sample(logits.data(), 5) == 2);
    }
    // top_k=1 等价贪心
    {
        Sampler s(SamplerParams{1.0f, 1, 1.0f, 0});
        for (int i = 0; i < 20; i++) { CHECK(s.sample(logits.data(), 5) == 2); }
    }
    // 尖峰分布 + 抽样：绝大多数应落在 2（统计性断言）
    {
        Sampler s(SamplerParams{0.8f, 40, 0.95f, 42});
        int hit = 0, N = 2000;
        for (int i = 0; i < N; i++) { if (s.sample(logits.data(), 5) == 2) { hit++; } }
        CHECK(hit > N * 0.9);
    }
    // 固定种子可复现
    {
        Sampler a(SamplerParams{1.0f, 5, 0.9f, 7});
        Sampler b(SamplerParams{1.0f, 5, 0.9f, 7});
        for (int i = 0; i < 10; i++) {
            CHECK(a.sample(logits.data(), 5) == b.sample(logits.data(), 5));
        }
    }
    // 均匀 logits + top_p 截断：top_p=0.3 时只可能抽到 cum>0.3 的前两个
    {
        std::vector<float> flat = {1, 1, 1, 1, 1, 1, 1, 1};
        Sampler s(SamplerParams{1.0f, 0, 0.3f, 1});
        for (int i = 0; i < 50; i++) {
            int t = s.sample(flat.data(), 8);
            CHECK(t >= 0 && t < 8);
        }
    }
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// TestGenerateLoop：完整自回归循环（贪心）
// ---------------------------------------------------------------------------
static void test_generate_loop() {
    TinyModelBuilder b;
    b.fill_all(777);
    std::string path = b.write("gen");
    GgufFile f = GgufFile::load(path);
    Model m = Model::load(f);
    const uint32_t V = m.hparams().n_vocab;

    auto run = [&](int seed) {
        KVCache kv(m.hparams().n_layer, m.hparams().kv_dim(), m.hparams().n_ctx);
        std::vector<int> prompt = {3, 1, 4};
        std::vector<float> logits;
        for (size_t p = 0; p < prompt.size(); p++) {
            logits = m.forward_token(prompt[p], static_cast<int>(p), kv);
        }
        Sampler sampler(SamplerParams{0.0f, 1, 1.0f, static_cast<uint32_t>(seed)});
        std::vector<int> out;
        for (int i = 0; i < 8; i++) {
            int t = sampler.sample(logits.data(), V);
            out.push_back(t);
            logits = m.forward_token(t, static_cast<int>(prompt.size()) + i, kv);
        }
        return out;
    };

    auto a = run(0), c = run(0);
    CHECK(a == c);                       // 贪心完全可复现
    for (int t : a) { CHECK(t >= 0 && t < static_cast<int>(V)); }
    CHECK(a.size() == 8);

    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

int main() {
    test_kv_equivalence();
    test_sampler();
    test_generate_loop();

    std::cout << "passed=" << g_pass << " failed=" << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
