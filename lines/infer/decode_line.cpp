// ============================================================================
// 线⑥ 自回归生成：末位 logits → 采样 → 新 token → 前向写回 KV Cache → 循环到 EOS
//
// 对照 llama.cpp @ 88b47a755:
//   src/llama-sampler.cpp   llama_sampler_chain（temp / top_k / top_p / dist 逐节）
//   src/llama-context.cpp   llama_context::decode（把新 token 推进上下文）
//   examples/simple-chat    生成循环骨架
//
// 跨模块：src/sampler.cpp（采样）· src/model.cpp（前向）· src/kvcache.cpp（缓存追加）
//         · src/main.cpp（CLI 参数 → SamplerParams）
//
// 学习问题：生成循环里，每一步的「上下文」到底是什么——是重新跑一遍 prompt，
//           还是只把新 token 追加进去？两者结果一样吗？
//
// 跑：MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_decode
// ============================================================================

#include "gguf.h"
#include "kvcache.h"
#include "line.hpp"
#include "lines.hpp"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

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
int minillama::lines::run_decode(LineCtx & ctx) {
    const std::string path = resolve_model_path();

    // ------------------------------------------------------------------
    ctx.step_at(1, "前置：GGUF → 词表 + 模型",
                "src/gguf.cpp + src/tokenizer.cpp + src/model.cpp: load 三件套",
                "./build/mini-llama --model <同一文件> --prompt \"Hi\" --n 8");
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

    // ------------------------------------------------------------------
    ctx.step_at(2, "CLI 参数 → SamplerParams（温度 / top-k / top-p / 种子）",
                "src/main.cpp: 参数解析段（对照 examples/cli 的同名 flag）",
                "src/sampler.h: SamplerParams 的 🟢/🟡/🔴 三档说明");
    SamplerParams sp;
    sp.temp  = 0.0f;   // 贪心：结果可复现，便于逐 token 对照
    sp.top_k = 40;
    sp.top_p = 0.95f;
    sp.seed  = 1234;
    ctx.value("temp", sp.temp);
    ctx.value("top_k", sp.top_k);
    ctx.value("top_p", sp.top_p);
    ctx.value("seed", sp.seed);
    ctx.log("temp<=0 走贪心分支：采样器直接取 argmax，温度/top-k/top-p 都不参与");

    // ------------------------------------------------------------------
    ctx.step_at(3, "预填充：把 prompt 推进 KV Cache，拿到末位 logits",
                "src/model.cpp: Model::forward_token（对照 llama_context::decode 的首次调用）",
                "详见 prefill 线");
    std::vector<int> ids = tok.encode("Hi", true);
    KVCache kv(hp.n_layer, hp.kv_dim(), hp.n_ctx);
    std::vector<float> logits;
    for (size_t p = 0; p < ids.size(); p++) {
        logits = model.forward_token(ids[p], static_cast<int>(p), kv);
    }
    ctx.value("prompt token 数", ids.size());
    ctx.value("缓存 size", kv.size());

    // ------------------------------------------------------------------
    ctx.step_at(4, "采样：从 logits 挑出下一个 token",
                "src/sampler.cpp: Sampler::sample（对照 llama_sampler_chain 的 dist 节点）",
                "tests/test_generate.cpp: 采样器用例（贪心 / top_k=1 / 种子复现）");
    Sampler sampler(sp);
    const int first = sampler.sample(logits.data(), hp.n_vocab);
    ctx.value("采样出的 token id", first);
    ctx.value("token 文本", tok.token_text(first));

    // ------------------------------------------------------------------
    ctx.step_at(5, "前向：把新 token 写回 KV Cache（位置 = 已推进的位置数）",
                "src/model.cpp: Model::forward_token + src/kvcache.cpp: KVCache::append",
                "缓存 size 每步 +1，历史位置不重算");
    const size_t pos = kv.size();
    logits = model.forward_token(first, static_cast<int>(pos), kv);
    ctx.value("推进前位置", pos);
    ctx.value("推进后缓存 size", kv.size());
    ctx.check(kv.size() == pos + 1, "缓存 size 每步 +1");

    // ------------------------------------------------------------------
    ctx.step_at(6, "循环：采样 → 前向 → 直到 EOS 或达到上限",
                "src/main.cpp: 生成循环（对照 examples/simple-chat 的循环骨架）",
                "./build/mini-llama --model <同一文件> --prompt \"Hi\" --n 8 --temp 0");
    std::vector<int> generated;
    const int max_new = 4;
    for (int i = 0; i < max_new; i++) {
        int t = sampler.sample(logits.data(), hp.n_vocab);
        if (t == tok.eos_id()) {
            ctx.log("命中 EOS，生成结束");
            break;
        }
        generated.push_back(t);
        logits = model.forward_token(t, static_cast<int>(kv.size()), kv);
    }
    ctx.value("生成 token 数", generated.size());
    for (size_t i = 0; i < generated.size(); i++) {
        ctx.log("生成[" + std::to_string(i) + "] id=" + std::to_string(generated[i]) +
                " 文本=\"" + tok.token_text(generated[i]) + "\"");
    }
    ctx.check(generated.size() > 0, "至少生成出 1 个 token");
    // 缓存里还多着第 5 跳那一次前向（prompt 之后先算了一个 token 用来演示）
    ctx.check(kv.size() == ids.size() + 1 + generated.size(),
              "缓存 size == prompt 位置数 + 第 5 跳的 1 次 + 生成 token 数",
              std::to_string(kv.size()),
              std::to_string(ids.size() + 1 + generated.size()));

    // ------------------------------------------------------------------
    ctx.step_at(7, "贪心可复现：同样的种子 + 同样的输入 → 同样的输出",
                "src/sampler.cpp: Sampler::sample 的贪心分支（不消耗随机数）",
                "tests/test_generate.cpp: 贪心可复现用例");
    {
        Sampler s2(sp);
        int a = s2.sample(logits.data(), hp.n_vocab);
        Sampler s3(sp);
        int b = s3.sample(logits.data(), hp.n_vocab);
        ctx.value("两次采样结果", a);
        ctx.check(a == b, "同样输入两次采样结果相同（贪心是确定性的）");
    }

    // ------------------------------------------------------------------
    ctx.step_at(8, "结局：token 序列 → 文本",
                "src/tokenizer.cpp: Tokenizer::decode（消费方）",
                "./build/mini-llama --model <同一文件> --prompt \"Hi\" --n 8 --temp 0");
    const std::string text = tok.decode(generated);
    ctx.value("生成文本", text);
    ctx.check(true, "生成循环走完");

    return ctx.summary();
}

// ---------------------------------------------------------------------------
// ③ 自注册：编译进去就出现在 --list，不改任何公共文件
// ---------------------------------------------------------------------------
MINILLAMA_LINE_REGISTER(reg_decode, "decode", "末位 logits → 采样 → 新 token → 写回 KV Cache → 循环到 EOS", ::minillama::lines::run_decode);

// ④ 独立 main：只在编译 line_decode 这个单跑目标时打开
#ifdef MINILLAMA_LINE_STANDALONE
MINILLAMA_LINE_STANDALONE_MAIN("decode", ::minillama::lines::run_decode)
#endif
