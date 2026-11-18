// ============================================================================
// 线③ 文本编码：prompt 文本 → 归一化 → 切词 → 按 score 合并 → token id 序列
//
// 对照 llama.cpp @ 88b47a755:
//   src/llama-vocab.cpp  llm_tokenizer_spm_session::tokenize
//                      llm_tokenizer_bpe_session::tokenize
//                      llama_escape_whitespace / unicode_byte_to_utf8
//   src/unicode.cpp      unicode_regex_split（真实版用通用正则引擎切词）
//
// 跨模块：src/tokenizer.cpp（合并核心）· src/unicode.cpp（预分词扫描器 + 字节映射）
//         · src/gguf.cpp（词表来自元数据）
//
// 学习问题：为什么 SPM 和 BPE 两种分词器要共用同一个「按 score 合并」的核心，
//           而真实版把它们写成了两份几乎一样的代码？
//
// 跑：MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_tokenize
// ============================================================================

#include "gguf.h"
#include "line.hpp"
#include "lines.hpp"
#include "tokenizer.h"
#include "unicode.h"

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

// 这条线用的样本：英文 + 中文 + 数字 + 标点，四个预分词器分支都能碰到
const char * SAMPLE = "Hello world 2026, 你好！";

} // namespace

// ---------------------------------------------------------------------------
// 线本体
// ---------------------------------------------------------------------------
int minillama::lines::run_tokenize(LineCtx & ctx) {
    const std::string path = resolve_model_path();

    // ------------------------------------------------------------------
    ctx.step_at(1, "前置：GGUF → 词表",
                "src/gguf.cpp + src/tokenizer.cpp: GgufFile::load → Tokenizer::load",
                "词表住在 GGUF 元数据里，不是独立的词表文件");
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
    try {
        tok = Tokenizer::load(*gguf);
    } catch (const std::exception & e) {
        ctx.check(false, "词表应当能装载", e.what(), "装载成功");
        return ctx.summary();
    }
    ctx.value("词表类型", gguf->get_string("tokenizer.ggml.model"));
    ctx.value("词表大小", tok.n_tokens());
    ctx.value("样本", SAMPLE);

    // ------------------------------------------------------------------
    ctx.step_at(2, "加 BOS（add_special 且词表约定加）",
                "src/tokenizer.cpp: Tokenizer::encode 的开头段",
                "BOS 是词表约定，不是文本内容——llama-spm / llama3 加，qwen2 不加");
    std::vector<int> ids = tok.encode(SAMPLE, true);
    ctx.value("add_bos", tok.add_bos());
    ctx.value("首 token id", ids.empty() ? -1 : ids[0]);
    ctx.check(tok.add_bos() ? (ids.size() > 0 && ids[0] == tok.bos_id())
                            : (ids.empty() || ids[0] != tok.bos_id()),
              "首 token 是 BOS（当词表约定加 BOS 时）");

    // ------------------------------------------------------------------
    ctx.step_at(3, "归一化：SPM 句首补空格 + ▁ 转义；BPE 走 bytes_to_unicode 映射",
                "src/tokenizer.cpp: Tokenizer::encode 的 work 准备段（对照 llama_escape_whitespace / unicode_byte_to_utf8）",
                "SPM 句首补空格与 add_special 无关——这是 llama.cpp 的历史规范");
    ctx.log("SPM：' ' → U+2581 (▁)；BPE(qwen2/llama3)：原始字节 → 256 个可见码点");

    // ------------------------------------------------------------------
    ctx.step_at(4, "切词：BPE 走预分词扫描器，SPM 整段不切",
                "src/unicode.cpp: pre_tokenize（对照 llama.cpp 的 unicode_regex_split 通用正则引擎）",
                "固定 6 分支模式手写成确定性扫描器，行为等价、零依赖");
    {
        // 单独调扫描器看切分结果（真实管线里 BPE 词表会先做 bytes_to_unicode 映射）
        const auto pieces = pre_tokenize(SAMPLE, PreTokenizer::LLAMA3);
        ctx.value("预分词片段数", pieces.size());
        for (size_t i = 0; i < pieces.size() && i < 8; i++) {
            ctx.log("片段[" + std::to_string(i) + "] = \"" + pieces[i] + "\"");
        }
        // 切分不变式：片段拼起来必须等于原文（除空白被吞掉的场景）
        std::string joined;
        for (const auto & p : pieces) joined += p;
        ctx.log("片段拼接长度 " + std::to_string(joined.size()) +
                " / 原文本长度 " + std::to_string(std::string(SAMPLE).size()));
    }

    // ------------------------------------------------------------------
    ctx.step_at(5, "合并核心：symbol 链表 + 按 score 优先的相邻合并",
                "src/tokenizer.cpp: Tokenizer::merge_tokenize（对照 llm_tokenizer_spm_session / bpe_session）",
                "词表里每个多字符 token 的 score 就是它参与合并的优先级");
    ctx.value("编码后 token 数", ids.size());
    for (size_t i = 0; i < ids.size() && i < 12; i++) {
        ctx.log("token[" + std::to_string(i) + "] id=" + std::to_string(ids[i]) +
                " 文本=\"" + tok.token_text(ids[i]) + "\"");
    }

    // ------------------------------------------------------------------
    ctx.step_at(6, "字节兜底：词表里没有的残段逐字节落 BYTE token",
                "src/tokenizer.cpp: Tokenizer::byte_token 的三级查找链（对照 llama_vocab::byte_to_token）",
                "任意二进制输入都必须能编码——这是词表「完备性」的兜底保证");
    ctx.check(ids.size() > 0, "至少编码出 1 个 token");
    bool all_in_range = true;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(tok.n_tokens())) { all_in_range = false; }
    }
    ctx.check(all_in_range, "所有 token id 落在词表范围内");

    // ------------------------------------------------------------------
    ctx.step_at(7, "结局：token id 序列可交给 prefill 线",
                "src/model.cpp: Model::forward_token（消费方）",
                "./build/mini-llama --model <同一文件> --prompt \"" + std::string(SAMPLE) + "\"");
    const std::string back = tok.decode(ids);
    ctx.value("解码回来", back);
    ctx.log("SPM 词表解码回来会带句首空格——那是规范行为，不是 bug");

    return ctx.summary();
}

// ---------------------------------------------------------------------------
// ③ 自注册：编译进去就出现在 --list，不改任何公共文件
// ---------------------------------------------------------------------------
MINILLAMA_LINE_REGISTER(reg_tokenize, "tokenize", "prompt 文本 → token id 序列（归一化 → 切词 → 按 score 合并）", ::minillama::lines::run_tokenize);

// ④ 独立 main：只在编译 line_tokenize 这个单跑目标时打开
#ifdef MINILLAMA_LINE_STANDALONE
MINILLAMA_LINE_STANDALONE_MAIN("tokenize", ::minillama::lines::run_tokenize)
#endif
