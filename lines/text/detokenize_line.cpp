// ============================================================================
// 线④ 文本解码：token id 序列 → 跳过控制 token → BYTE 还原 → ▁ 逆变换 → 文本
//
// 对照 llama.cpp @ 88b47a755:
//   src/llama-vocab.cpp  llama_vocab::impl::detokenize
//                      llama_vocab::impl::token_to_piece（BYTE / ▁ 的还原）
//   src/unicode.cpp      unicode_utf8_to_byte（bytes_to_unicode 的逆映射）
//
// 跨模块：src/tokenizer.cpp（逐 token 还原）· src/unicode.cpp（映射字符 → 原始字节）
//
// 学习问题：解码时为什么要把 CONTROL / UNKNOWN / UNUSED 三类 token 全部跳过，
//           而不是输出一个占位符？
//
// 跑：MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_detokenize
// ============================================================================

#include "gguf.h"
#include "line.hpp"
#include "lines.hpp"
#include "tokenizer.h"

#include <fstream>
#include <map>
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
int minillama::lines::run_detokenize(LineCtx & ctx) {
    const std::string path = resolve_model_path();

    // ------------------------------------------------------------------
    ctx.step_at(1, "前置：GGUF → 词表",
                "src/gguf.cpp + src/tokenizer.cpp: GgufFile::load → Tokenizer::load",
                "解码需要词表里的 token 类型与文本");
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
    ctx.value("词表大小", tok.n_tokens());
    ctx.value("BOS id", tok.bos_id());
    ctx.value("EOS id", tok.eos_id());

    // ------------------------------------------------------------------
    ctx.step_at(2, "取一段 token 序列当输入（这里用 encode 的产物，保证是合法 id）",
                "src/tokenizer.cpp: Tokenizer::encode",
                "tests/test_tokenizer.cpp: 三个真实词表的往返用例");
    const std::string sample = "Hello world 2026, 你好！";
    std::vector<int> ids = tok.encode(sample, true);
    ctx.value("token 数", ids.size());
    for (size_t i = 0; i < ids.size() && i < 10; i++) {
        ctx.log("token[" + std::to_string(i) + "] id=" + std::to_string(ids[i]) +
                " 文本=\"" + tok.token_text(ids[i]) + "\"");
    }

    // ------------------------------------------------------------------
    ctx.step_at(3, "按 token 类型决定渲不渲染（CONTROL / UNKNOWN 跳过，USER_DEFINED 照原样输出）",
                "src/tokenizer.cpp: Tokenizer::decode 的 switch（对照 llama_vocab::impl::token_to_piece 的 attr_special 判定）",
                "tests/test_tokenizer.cpp: 跳过特殊/控制 token 的用例");
    {
        // 词表里的 token 类型分布（GGUF 的 tokenizer.ggml.token_type）
        std::map<int, int> hist;
        if (const GgufValue * v = gguf->find("tokenizer.ggml.token_type")) {
            for (const auto & e : gguf->get_array("tokenizer.ggml.token_type")) {
                hist[std::get<int32_t>(e.v)]++;
            }
        }
        for (const auto & h : hist) {
            ctx.log("类型 " + std::to_string(h.first) + " × " + std::to_string(h.second) +
                    "（1=NORMAL 3=CONTROL 4=USER_DEFINED 6=BYTE）");
        }
        ctx.log("本模型的 EOS 标成 **USER_DEFINED(4)** 而不是 CONTROL(3)——所以它会被原样渲染");
        ctx.log("llama.cpp 同一套规则：attr_special = UNKNOWN|CONTROL 跳过，USER_DEFINED 照原样输出");
        ctx.check(tok.decode({tok.eos_id()}) == "</s>",
                  "EOS 是 USER_DEFINED 类，解码时照原样输出（与 llama.cpp 一致）",
                  tok.decode({tok.eos_id()}), "</s>");
        // 越界 id 走的是另一条跳过路径
        ctx.check(tok.decode({-1}).empty() && tok.decode({999999999}).empty(),
                  "越界的 token id 被跳过（不会崩，也不会输出内容）");
    }

    // ------------------------------------------------------------------
    ctx.step_at(4, "BYTE token 还原成原始字节",
                "src/tokenizer.cpp: Tokenizer::decode 的 BYTE 分支 + byte_token_text_to_raw",
                "三种文本格式都认：<0xXX> / 单原始字节 / bytes_to_unicode 映射字符");
    ctx.log("BYTE token 是「任意字节」的兜底通道——没有它，未登录字符就编不出来");

    // ------------------------------------------------------------------
    ctx.step_at(5, "▁ 逆变换回空格；bytes_to_unicode 映射字符逆映射回原始字节",
                "src/tokenizer.cpp: Tokenizer::decode 尾部的 replace_all（对照 llama_escape_whitespace 的逆向）",
                "SPM 词表：token 文本里 U+2581 代表空格；qwen2/llama3：映射字符代表原始字节");
    const std::string out = tok.decode(ids);
    ctx.value("解码结果", out);

    // ------------------------------------------------------------------
    ctx.step_at(6, "结局：文本",
                "src/main.cpp: 生成循环里的 std::cout << tok.decode({t})（消费方）",
                "./build/mini-llama --model <同一文件> --prompt \"" + sample + "\" --n 8");
    ctx.check(!out.empty(), "解码结果非空");
    ctx.check(out.find("Hello") != std::string::npos,
              "解码结果里能看到原文的片段 Hello", out, "含 Hello");
    ctx.log("SPM 词表解码回来带句首空格；BPE 词表不带——两者的空格约定不同");

    return ctx.summary();
}

// ---------------------------------------------------------------------------
// ③ 自注册：编译进去就出现在 --list，不改任何公共文件
// ---------------------------------------------------------------------------
MINILLAMA_LINE_REGISTER(reg_detokenize, "detokenize", "token id 序列 → 文本（跳过控制 token → BYTE 还原 → ▁ 逆变换）", ::minillama::lines::run_detokenize);

// ④ 独立 main：只在编译 line_detokenize 这个单跑目标时打开
#ifdef MINILLAMA_LINE_STANDALONE
MINILLAMA_LINE_STANDALONE_MAIN("detokenize", ::minillama::lines::run_detokenize)
#endif
