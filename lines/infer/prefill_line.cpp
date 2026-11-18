// ============================================================================
// 线⑤ 预填充：token 序列 → 逐位置前向 → 写满 KV Cache → 末位 logits
//
// 对照 llama.cpp @ 88b47a755:
//   src/llama-context.cpp  llama_context::decode（一次 batch 推进多个位置）
//   src/llama-graph.cpp    build_attn / build_ffn / build_rope（计算图节点）
//   src/llama-kv-cache.cpp llama_kv_cache::find_slot / set_input（缓存写入）
//   ggml/src/ggml-cpu/     ggml_mul_mat / ggml_rms_norm / ggml_rope 内核
//
// 跨模块：src/model.cpp（前向）· src/kvcache.cpp（缓存追加与读取）· src/quant.cpp（反量化内核）
//
// 学习问题：为什么「增量前向」和「全量前向」必须给出逐位一致的 logits？
//           如果两者允许有微小差异，KV Cache 的正确性靠什么保证？
//
// 跑：MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_prefill
// ============================================================================

#include "gguf.h"
#include "kvcache.h"
#include "line.hpp"
#include "lines.hpp"
#include "model.h"
#include "tokenizer.h"

#include <cmath>
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

} // namespace

// ---------------------------------------------------------------------------
// 线本体
// ---------------------------------------------------------------------------
int minillama::lines::run_prefill(LineCtx & ctx) {
    const std::string path = resolve_model_path();

    // ------------------------------------------------------------------
    ctx.step_at(1, "前置：GGUF → 词表 + 模型",
                "src/gguf.cpp + src/tokenizer.cpp + src/model.cpp: load 三件套",
                "./build/mini-llama --model <同一文件> --prompt \"Hi\" --n 4");
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
    Tokenizer tok;
    Model model;
    try {
        tok = Tokenizer::load(*gguf);
        model = Model::load(*gguf);
    } catch (const std::exception & e) {
        ctx.check(false, "模型应当能装载", e.what(), "装载成功");
        return ctx.summary();
    }
    const HParams & hp = model.hparams();
    ctx.value("层数", hp.n_layer);
    ctx.value("维度", hp.n_embd);
    ctx.value("词表", hp.n_vocab);

    // ------------------------------------------------------------------
    ctx.step_at(2, "编码 prompt，得到要推进的位置序列",
                "src/tokenizer.cpp: Tokenizer::encode（上游对应 tokenizer 阶段）",
                "tests/test_tokenizer.cpp: 往返用例");
    std::vector<int> ids = tok.encode("Hi", true);
    ctx.value("prompt 文本", "Hi");
    ctx.value("token 数", ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
        ctx.log("位置 " + std::to_string(i) + " token id=" + std::to_string(ids[i]) +
                " 文本=\"" + tok.token_text(ids[i]) + "\"");
    }

    // ------------------------------------------------------------------
    ctx.step_at(3, "建 KV Cache（按层 × 按位置 × 维度）",
                "src/kvcache.cpp: KVCache::KVCache（对照 llama_kv_cache::llama_kv_cache_init）",
                "布局 k_[((layer*cap) + pos) * dim_ + i]；同一层的 K 连续");
    KVCache kv(hp.n_layer, hp.kv_dim(), hp.n_ctx);
    ctx.value("缓存层数", hp.n_layer);
    ctx.value("每位置 KV 维度", hp.kv_dim());
    ctx.value("容量（位置数）", hp.n_ctx);
    ctx.value("初始 size", kv.size());

    // ------------------------------------------------------------------
    ctx.step_at(4, "逐位置推进：embed → 每层 block → 输出头 → logits",
                "src/model.cpp: Model::forward_token（对照 llama_context::decode 的单 token 路径）",
                "tests/test_model.cpp: 前向形状 / 因果性用例");
    std::vector<float> logits;
    std::vector<std::vector<float>> per_pos;   // 每个位置的 logits，因果性对比要用
    for (size_t p = 0; p < ids.size(); p++) {
        logits = model.forward_token(ids[p], static_cast<int>(p), kv);
        per_pos.push_back(logits);
        ctx.value("位置 " + std::to_string(p) + " 之后缓存 size", kv.size());
        ctx.log("logits 长度 = " + std::to_string(logits.size()) +
                "（= 词表大小 " + std::to_string(hp.n_vocab) + "）");
    }
    ctx.check(logits.size() == hp.n_vocab, "logits 长度 == 词表大小",
              std::to_string(logits.size()), std::to_string(hp.n_vocab));
    ctx.check(kv.size() == ids.size(), "缓存 size == 已推进的位置数",
              std::to_string(kv.size()), std::to_string(ids.size()));

    // ------------------------------------------------------------------
    ctx.step_at(5, "缓存里确实写进了 K/V（不是只算不用）",
                "src/kvcache.cpp: KVCache::append / KVCache::k（对照 llama_kv_cache::set_input）",
                "第 0 层的第 0 个位置的 K 向量应当非零");
    {
        const float * k0 = kv.k(0, 0);
        const float * v0 = kv.v(0, 0);
        double knorm = 0, vnorm = 0;
        for (uint32_t i = 0; i < kv.dim(); i++) {
            knorm += double(k0[i]) * k0[i];
            vnorm += double(v0[i]) * v0[i];
        }
        ctx.value("第 0 层位置 0 的 K 向量模", std::sqrt(knorm));
        ctx.value("第 0 层位置 0 的 V 向量模", std::sqrt(vnorm));
        ctx.check(knorm > 0.0, "K 向量非零（缓存确实写进去了）");
        ctx.check(vnorm > 0.0, "V 向量非零（缓存确实写进去了）");
    }

    // ------------------------------------------------------------------
    ctx.step_at(6, "核心不变式：增量前向 ≡ 全量前向（逐位置 logits 一致）",
                "src/model.cpp: Model::forward（内部用临时 KVCache 逐位置调 forward_token）",
                "tests/test_model.cpp / test_generate.cpp: 增量与全量一致性用例");
    {
        // 全量路径：Model::forward 自己开一个临时 KVCache 逐位置推进，收 n 行 logits
        const std::vector<float> full = model.forward(ids);
        const size_t V = hp.n_vocab;
        double max_diff = 0.0;
        for (size_t p = 0; p < ids.size(); p++) {
            for (size_t i = 0; i < V; i++) {
                max_diff = std::max(max_diff,
                                    std::fabs(double(per_pos[p][i]) - double(full[p * V + i])));
            }
        }
        ctx.value("最大逐元素差", max_diff);
        ctx.check(max_diff < 1e-4,
                  "增量前向与全量前向逐位置一致，最大逐元素差 < 1e-4（缓存只是避免重算）",
                  std::to_string(max_diff), "< 1e-4");

        // clear 后复用：同一个缓存对象清空再来一遍，结果不变
        kv.clear();
        std::vector<float> again = model.forward_token(ids[0], 0, kv);
        double reuse_diff = 0.0;
        for (size_t i = 0; i < V; i++) {
            reuse_diff = std::max(reuse_diff, std::fabs(double(again[i]) - double(per_pos[0][i])));
        }
        ctx.value("clear 后复用的最大差", reuse_diff);
        ctx.check(reuse_diff < 1e-4, "清空缓存后重跑第 0 个位置，结果不变");
    }

    // ------------------------------------------------------------------
    ctx.step_at(7, "因果性：改后面的 token 不影响前面的 logits",
                "src/model.cpp: Model::forward_token 的注意力只可见 0..pos",
                "tests/test_model.cpp: 因果性逐位验证用例");
    {
        std::vector<int> alt = ids;
        alt.back() = (alt.back() + 1) % static_cast<int>(hp.n_vocab);
        KVCache kv3(hp.n_layer, hp.kv_dim(), hp.n_ctx);
        std::vector<std::vector<float>> alt_per_pos;
        for (size_t p = 0; p < alt.size(); p++) {
            alt_per_pos.push_back(model.forward_token(alt[p], static_cast<int>(p), kv3));
        }
        if (alt.size() >= 2) {
            // 前 n-1 个位置必须与原序列**逐位一致**（未来不该影响过去）
            double early_diff = 0.0;
            for (size_t p = 0; p + 1 < ids.size(); p++) {
                for (size_t i = 0; i < per_pos[p].size(); i++) {
                    early_diff = std::max(early_diff,
                                          std::fabs(double(alt_per_pos[p][i]) - double(per_pos[p][i])));
                }
            }
            // 末位必然变了（否则说明改动没生效，对比无意义）
            double last_diff = 0.0;
            for (size_t i = 0; i < per_pos.back().size(); i++) {
                last_diff = std::max(last_diff,
                                     std::fabs(double(alt_per_pos.back()[i]) - double(per_pos.back()[i])));
            }
            ctx.value("前 n-1 个位置的最大差", early_diff);
            ctx.value("末位位置的最大差", last_diff);
            ctx.check(early_diff == 0.0, "改末位 token 不影响前面位置的 logits（因果性）",
                      std::to_string(early_diff), "0");
            ctx.check(last_diff > 0.0, "改末位 token 后末位 logits 变了（对比有效）");
        } else {
            ctx.log("只有 1 个位置，因果性对比跳过");
        }
    }

    // ------------------------------------------------------------------
    ctx.step_at(8, "结局：末位 logits 备好，交给 decode 线做采样",
                "src/sampler.cpp: Sampler::sample（消费方）",
                "./build/mini-llama --model <同一文件> --prompt \"Hi\" --n 8");
    {
        int best = 0;
        for (size_t i = 1; i < logits.size(); i++) {
            if (logits[i] > logits[best]) best = static_cast<int>(i);
        }
        ctx.value("argmax token id", best);
        ctx.value("argmax 文本", tok.token_text(best));
    }
    ctx.check(true, "预填充完成");

    return ctx.summary();
}

// ---------------------------------------------------------------------------
// ③ 自注册：编译进去就出现在 --list，不改任何公共文件
// ---------------------------------------------------------------------------
MINILLAMA_LINE_REGISTER(reg_prefill, "prefill", "token 序列 → 写满 KV Cache 的 logits（增量前向 ≡ 全量前向）", ::minillama::lines::run_prefill);

// ④ 独立 main：只在编译 line_prefill 这个单跑目标时打开
#ifdef MINILLAMA_LINE_STANDALONE
MINILLAMA_LINE_STANDALONE_MAIN("prefill", ::minillama::lines::run_prefill)
#endif
