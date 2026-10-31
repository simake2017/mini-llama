// mini-llama：命令行推理入口（单元 4 成型，单元 5 支持量化模型）
//
// 用法:
//   mini-llama --model model.gguf --prompt "你好" [--temp 0.8] [--top-k 40]
//              [--top-p 0.95] [--n 128] [--seed 1234]
//
// 【对照】examples/simple-chat / llama-cli 的最小骨架：
//   加载模型 -> 编码 prompt -> prefill 逐 token 推进 KV cache ->
//   采样下一个 token -> 前向 -> 重复直到 EOS 或达到上限。

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
        // ① 加载三件套
        GgufFile gguf = GgufFile::load(model_path);
        Tokenizer tok = Tokenizer::load(gguf);
        Model model = Model::load(gguf);
        const HParams & hp = model.hparams();

        std::cerr << "模型: " << model_path
                  << "  层数=" << hp.n_layer
                  << "  维度=" << hp.n_embd
                  << "  头=" << hp.n_head << "/" << hp.n_head_kv
                  << "  词表=" << hp.n_vocab << "\n";

        // ② 编码 + prefill
        auto ids = tok.encode(prompt, true);
        KVCache kv(hp.n_layer, hp.kv_dim(), hp.n_ctx);
        std::vector<float> logits;
        for (size_t p = 0; p < ids.size(); p++) {
            logits = model.forward_token(ids[p], static_cast<int>(p), kv);
        }
        std::cerr << "prefill 完成: " << ids.size() << " tokens\n";

        // ③ 自回归生成
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
        std::cerr << "\n生成 " << gen << " tokens，耗时 " << sec << "s"
                  << "（" << (sec > 0 ? gen / sec : 0) << " tok/s）\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
}
