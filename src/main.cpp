// ============================================================================
// ① 定位：命令行推理入口 —— 一个 main 走完「加载 → 编码 → prefill → 采样 → 生成」
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  examples/simple/simple.cpp:14:main
//                          —— 上游 examples 层最小的「加载 → decode → 采样」骨架
//    llama.cpp @ 88b47a755  common/sampling.cpp:334（默认采样链的装配处）
//
// ③ 动机：
//    【教学】本文件含教学便利代码：上游把这段拆成 examples/ 下几十个程序，本仓库
//      合成一个 main 覆盖全链路。登记见 docs/03-forward.md §16.4 的「CLI 入口」项。
//    【概念】CLI 只是「编排」，不含任何模型数学：真正的计算在 Model / Tokenizer /
//      Sampler 里——所以本文件能一眼看完推理链的全貌。
//    【简化】只支持单轮、不读 stdin、无对话历史：真实版 llama-cli 还有交互模式、
//      多会话、grammar 约束等；本项目只留一条最短可验证路径。
//    【对照】上游用 llama_batch 一次喂多个 token 做 prefill；本实现逐 token 调
//      forward_token，语义相同——prefill 就是把 prompt 逐位推进 KV cache。
//
// ④ demo + 多方样例：
//    🟢 最小：--model X.gguf --prompt "Hi" --n 4 → 生成 4 个 token 后退出。
//    🟡 典型：--model models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf --prompt "Hi" --n 4
//       → stderr 摘要「层数=22 维度=2048 头=32/4 词表=32000」，stdout 输出续写。
//    🔴 极限：--temp 1.2 --top-k 100 --top-p 0.98 → 开放生成；本项目是纯 CPU
//       单线程，--n 越大越慢，别拿它跑长上下文。
// ============================================================================
//
// 用法:
//   mini-llama --model model.gguf --prompt "你好" [--temp 0.8] [--top-k 40]
//              [--top-p 0.95] [--n 128] [--seed 1234]

#include "gguf.h"
#include "kvcache.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

using namespace minillama;

static void usage(const char * argv0) {
    std::cerr << "用法: " << argv0
              << " --model X.gguf --prompt \"文本\" [--temp 0.8] [--top-k 40]"
                 " [--top-p 0.95] [--n 128] [--seed 1234]\n";
}

int main(int argc, char ** argv) {
    std::string model_path, prompt;
    SamplerParams sp;
    int n_max = 256;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char * name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "参数 " << name << " 缺少值\n"; exit(2); }
            return argv[++i];
        };
        if (a == "--model")  { model_path = need("--model"); }
        else if (a == "--prompt") { prompt = need("--prompt"); }
        else if (a == "--temp")   { sp.temp = std::stof(need("--temp")); }
        else if (a == "--top-k")  { sp.top_k = std::stoi(need("--top-k")); }
        else if (a == "--top-p")  { sp.top_p = std::stof(need("--top-p")); }
        else if (a == "--n")      { n_max = std::stoi(need("--n")); }
        else if (a == "--seed")   { sp.seed = static_cast<uint32_t>(std::stoul(need("--seed"))); }
        else { usage(argv[0]); return 2; }
    }
    if (model_path.empty() || prompt.empty()) { usage(argv[0]); return 2; }

    try {
        // ① 加载三件套：同一个 GGUF 文件，分词器读词表、模型读权重
        GgufFile gguf = GgufFile::load(model_path);
        Tokenizer tok = Tokenizer::load(gguf);
        Model model = Model::load(gguf);
        const HParams & hp = model.hparams();

        std::cerr << "模型: " << model_path          // 日志级别 0：关键生命周期事件
                  << "  层数=" << hp.n_layer
                  << "  维度=" << hp.n_embd
                  << "  头=" << hp.n_head << "/" << hp.n_head_kv
                  << "  词表=" << hp.n_vocab << "\n";

        // ② 编码 + prefill：prefill = 把 prompt 逐位推进 KV cache，
        //    每一步的 pos 就是该 token 在 prompt 里的下标
        auto ids = tok.encode(prompt, true);
        KVCache kv(hp.n_layer, hp.kv_dim(), hp.n_ctx);
        std::vector<float> logits;
        for (size_t p = 0; p < ids.size(); p++) {
            logits = model.forward_token(ids[p], static_cast<int>(p), kv);
        }
        std::cerr << "prefill 完成: " << ids.size() << " tokens\n";   // 日志级别 1：摘要

        // ③ 自回归生成：采样 → 前向 → 再采样；KV cache 跨轮复用，历史不重算
        Sampler sampler(sp);
        auto t0 = std::chrono::steady_clock::now();
        int gen = 0;
        std::cout << tok.decode(ids);   // 回显 prompt
        for (; gen < n_max; gen++) {
            int t = sampler.sample(logits.data(), hp.n_vocab);
            if (t == tok.eos_id()) { break; }
            std::cout << tok.decode({t}) << std::flush;
            logits = model.forward_token(t, static_cast<int>(ids.size()) + gen, kv);
        }
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::cerr << "\n生成 " << gen << " tokens，耗时 " << sec << "s"   // 日志级别 1：摘要
                  << "（" << (sec > 0 ? gen / sec : 0) << " tok/s）\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
}
