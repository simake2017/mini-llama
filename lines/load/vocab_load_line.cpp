// ============================================================================
// 线② 词表载入：GGUF 元数据 → 词表 + 正反向索引 + BYTE 映射 → 可用的 Tokenizer
//
// 对照 llama.cpp @ 88b47a755:
//   src/llama-vocab.cpp  llama_vocab::impl::load（token_to_id / byte_to_token 构建）
//                      llama_vocab::impl::token_to_id / byte_to_token 的查找链
//   src/llama-arch.cpp  LLM_KV（tokenizer.ggml.* 的 key 命名）
//
// 跨模块：src/gguf.cpp（读元数据）· src/tokenizer.cpp（建词表）· src/unicode.cpp（BYTE 映射字符）
//
// 学习问题：llama.cpp 的词表里，BPE 与 SPM 两种类型共用一套「token 文本 ↔ id」结构，
//           只有编码路径不同。那「词表」这个概念到底是什么——是数据，还是算法？
//
// 跑：MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_vocab_load
// ============================================================================

#include "gguf.h"
#include "line.hpp"
#include "lines.hpp"
#include "tokenizer.h"

#include <fstream>
#include <optional>
#include <string>

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
int minillama::lines::run_vocab_load(LineCtx & ctx) {
    const std::string path = resolve_model_path();

    // ------------------------------------------------------------------
    ctx.step_at(1, "打开 GGUF（词表住在元数据里，不是独立的词表文件）",
                "src/gguf.cpp: GgufFile::load",
                "MINILLAMA_MODEL=<路径> ./build/line_vocab_load");
    ctx.value("模型路径", path);
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

    // ------------------------------------------------------------------
    ctx.step_at(2, "判定词表类型：tokenizer.ggml.model → SPM / BPE",
                "src/tokenizer.cpp: Tokenizer::load 的 ① 段（对照 llama_vocab::impl::load 的 type 分派）",
                "tests/test_tokenizer.cpp: llama-spm / llama-bpe / qwen2 三个真实词表冒烟");
    const std::string vmodel = gguf->get_string("tokenizer.ggml.model");
    ctx.value("tokenizer.ggml.model", vmodel);
    ctx.log(vmodel == "llama" ? "→ SPM 路径（句首补空格 + ▁ 转义 + 默认加 BOS）"
                              : "→ BPE 路径（预分词扫描器 + 词内 score 合并）");

    // ------------------------------------------------------------------
    ctx.step_at(3, "读三张平行数组：tokens / scores / token_type",
                "src/tokenizer.cpp: Tokenizer::load 的 ③ 段（对照 llama_vocab::impl::load 的 add_token 循环）",
                "tests/test_tokenizer.cpp: 缺 key / 类型错误的合成用例");
    const GgufArray & tokens = gguf->get_array("tokenizer.ggml.tokens");
    ctx.value("token 条数", tokens.size());
    const bool has_scores = gguf->find("tokenizer.ggml.scores") != nullptr;
    const bool has_types  = gguf->find("tokenizer.ggml.token_type") != nullptr;
    ctx.value("有 scores 数组", has_scores);
    ctx.value("有 token_type 数组", has_types);
    ctx.check(!tokens.empty(), "token 数组非空");
    if (!tokens.empty()) {
        ctx.log("token[0] 文本 = \"" + std::get<std::string>(tokens[0].v) + "\"");
    }

    // ------------------------------------------------------------------
    ctx.step_at(4, "建词表（正反向索引）",
                "src/tokenizer.cpp: Tokenizer::load 的 ④ 段（对照 llama_vocab::impl::token_to_id）",
                "反向索引在 encode 的合并阶段被反复查");
    Tokenizer tok;
    try {
        tok = Tokenizer::load(*gguf);
    } catch (const std::exception & e) {
        ctx.check(false, "词表应当能装载", e.what(), "装载成功");
        return ctx.summary();
    }
    ctx.check(true, "词表装载成功");
    ctx.value("词表大小", tok.n_tokens());
    ctx.check(tok.n_tokens() == tokens.size(),
              "词表大小 == GGUF 里 token 数组的长度",
              std::to_string(tok.n_tokens()), std::to_string(tokens.size()));

    // ------------------------------------------------------------------
    ctx.step_at(5, "建 BYTE token ↔ 原始字节 的双向映射（任意字节的兜底通道）",
                "src/tokenizer.cpp: Tokenizer::load 的 ⑤ 段 + byte_token_text_to_raw（对照 llama_vocab::byte_to_token）",
                "三种 BYTE 文本格式：<0xXX> / 单原始字节 / bytes_to_unicode 映射字符");
    ctx.log("查找链三级：显式 BYTE 表 → bytes_to_unicode 映射字符 → 单原始字节");
    ctx.check(tok.n_tokens() > 0, "词表非空");

    // ------------------------------------------------------------------
    ctx.step_at(6, "读特殊 token id 与开关（BOS / EOS / add_bos / add_eos）",
                "src/tokenizer.cpp: Tokenizer::load 的 ⑥ 段（对照 llama_vocab 的 special token 读取）",
                "tests/test_tokenizer.cpp: llama-bpe 加 BOS、qwen2 不加 BOS 的对照");
    ctx.value("bos_id", tok.bos_id());
    ctx.value("eos_id", tok.eos_id());
    ctx.value("add_bos", tok.add_bos());
    ctx.check(tok.bos_id() >= 0 && tok.bos_id() < static_cast<int>(tok.n_tokens()),
              "BOS id 落在词表范围内");
    ctx.check(tok.eos_id() >= 0 && tok.eos_id() < static_cast<int>(tok.n_tokens()),
              "EOS id 落在词表范围内");
    ctx.check(tok.is_eos(tok.eos_id()), "is_eos(eos_id) 为真");

    // ------------------------------------------------------------------
    ctx.step_at(7, "结局：Tokenizer 就绪，可以 encode / decode",
                "src/tokenizer.cpp: Tokenizer::encode / Tokenizer::decode",
                "./build/mini-llama --model <同一文件> --prompt \"hi\" --n 4");
    if (tok.bos_id() >= 0) {
        ctx.value("BOS 文本", tok.token_text(tok.bos_id()));
    }
    ctx.log("词表已就绪——后续 tokenize / detokenize 线都从这一步的产物出发");
    ctx.check(true, "词表载入完成");

    return ctx.summary();
}

// ---------------------------------------------------------------------------
// ③ 自注册：编译进去就出现在 --list，不改任何公共文件
// ---------------------------------------------------------------------------
MINILLAMA_LINE_REGISTER(reg_vocab_load, "vocab_load", "GGUF 元数据 → 可用的 Tokenizer（词表 + 索引 + BYTE 映射 + 特殊 token）", ::minillama::lines::run_vocab_load);

// ④ 独立 main：只在编译 line_vocab_load 这个单跑目标时打开
#ifdef MINILLAMA_LINE_STANDALONE
MINILLAMA_LINE_STANDALONE_MAIN("vocab_load", ::minillama::lines::run_vocab_load)
#endif
