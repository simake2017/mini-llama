// 单元 2 测试：分词器（预分词扫描器 + SPM/BPE 编码 + 解码）
//
// 策略:
//   1. 直接单测 pre_tokenize 扫描器（确定性最强的部分）
//   2. 用合成 GGUF 词表分别构造 SPM / BPE(qwen2) / BPE(llama-bpe) 三种词表，
//      验证编码结果、BOS 行为、byte 兜底、解码往返
//   3. 真实词表文件冒烟（环境变量 TEST_TOKENIZER_REAL_FILE）

#include "tokenizer.h"
#include "unicode.h"
#include "gguf.h"
#include "gguf_writer.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace minillama;
using minitest::Writer;
using minitest::write_temp;

static int g_fail = 0;
static int g_pass = 0;

// 容器打印辅助（CHECK_EQ 失败时输出实际值）
template <typename T>
static std::string to_str(const std::vector<T> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) { s += ", "; }
        s += "\"" + static_cast<std::string>(v[i]) + "\"";
    }
    return s + "]";
}

static std::string to_str(const std::vector<int> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) { s += ", "; }
        s += std::to_string(v[i]);
    }
    return s + "]";
}

static const std::string & to_str(const std::string & s) { return s; }

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__            \
                      << "  CHECK(" #cond ")\n";                           \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

#define CHECK_EQ(a, b)                                                     \
    do {                                                                   \
        auto _va = (a); auto _vb = (b);                                    \
        if (!(_va == _vb)) {                                               \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__            \
                      << "  CHECK_EQ(" #a ", " #b ")  实际: " << to_str(_va) \
                      << " vs " << to_str(_vb) << "\n";                    \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// 工具：写只含词表元数据的 GGUF
// ---------------------------------------------------------------------------
struct VocabEntry {
    std::string text;
    float       score;
    int32_t     type;
};

// 追加 256 个 BYTE token（原始单字节文本），返回起始 id（SPM 风格词表用）
static size_t add_byte_tokens(std::vector<VocabEntry> & v) {
    size_t base = v.size();
    for (int b = 0; b < 256; b++) {
        v.push_back({std::string(1, static_cast<char>(b)), -10.0f, 6});
    }
    return base;
}

// 追加 256 个 BYTE token（bytes_to_unicode 映射文本），返回起始 id（qwen2/gpt2 词表用）
static size_t add_byte_tokens_mapped(std::vector<VocabEntry> & v) {
    size_t base = v.size();
    for (int b = 0; b < 256; b++) {
        std::string s;
        utf8_encode(byte_to_unicode_cp(static_cast<uint8_t>(b)), s);
        v.push_back({s, -10.0f, 6});
    }
    return base;
}

static std::string write_vocab_gguf(
        const std::string & model,      // "llama" / "gpt2"
        const std::string & pre,        // "default" / "qwen2" / "llama-bpe"
        const std::vector<VocabEntry> & entries,
        int bos_id, int eos_id, bool add_bos) {
    Writer w;
    w.u32(0x46554747u); w.u32(3); w.u64(0); // magic/version/tensors=0
    w.u64(8);                               // kv count

    w.kv_header("general.architecture", 8); w.str("test");
    w.kv_header("tokenizer.ggml.model", 8); w.str(model);
    w.kv_header("tokenizer.ggml.pre", 8);   w.str(pre);
    w.kv_header("tokenizer.ggml.tokens", 9);
        w.u32(8); w.u64(entries.size());
        for (const auto & e : entries) { w.str(e.text); }
    w.kv_header("tokenizer.ggml.scores", 9);
        w.u32(6); w.u64(entries.size());
        for (const auto & e : entries) { w.f32(e.score); }
    w.kv_header("tokenizer.ggml.token_type", 9);
        w.u32(5); w.u64(entries.size());   // array of i32
        for (const auto & e : entries) { w.i32(e.type); }
    w.kv_header("tokenizer.ggml.bos_token_id", 4); w.u32(static_cast<uint32_t>(bos_id));
    w.kv_header("tokenizer.ggml.add_bos", 7);      w.u8(add_bos ? 1 : 0);

    (void)eos_id;
    return write_temp(w.buf(), model.c_str());
}

// ---------------------------------------------------------------------------
// TestPretokenize：预分词扫描器
// ---------------------------------------------------------------------------
static void test_pretokenize() {
    using VS = std::vector<std::string>;
    // 基本：字母/前导空格
    CHECK_EQ(pre_tokenize("Hello world", PreTokenizer::LLAMA3),
             (VS{"Hello", " world"}));
    // 多空格：段空白只留最后一个空格给后续词
    CHECK_EQ(pre_tokenize("  spaced   out  ", PreTokenizer::LLAMA3),
             (VS{" ", " spaced", "  ", " out", "  "}));
    // 缩略词大小写不敏感
    CHECK_EQ(pre_tokenize("I'M", PreTokenizer::LLAMA3), (VS{"I", "'M"}));
    CHECK_EQ(pre_tokenize("she'll", PreTokenizer::LLAMA3), (VS{"she", "'ll"}));
    // llama3: 数字 1~3 位分组
    CHECK_EQ(pre_tokenize("12345", PreTokenizer::LLAMA3), (VS{"123", "45"}));
    // qwen2: 数字单个分组
    CHECK_EQ(pre_tokenize("12345", PreTokenizer::QWEN2),
             (VS{"1", "2", "3", "4", "5"}));
    // 标点串带尾随换行
    CHECK_EQ(pre_tokenize("ok!\nnext", PreTokenizer::LLAMA3),
             (VS{"ok", "!\n", "next"}));
    // 切分结果拼起来必须等于原文（不变式，对所有样本成立）
    for (const char * s : {"Hello world", "  spaced   out  ", "a1b", "，。中文！test123",
                          "don't panic", "line1\nline2\r\n", "tab\there"}) {
        std::string joined;
        for (const auto & p : pre_tokenize(s, PreTokenizer::LLAMA3)) { joined += p; }
        CHECK_EQ(joined, std::string(s));
    }
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// TestTokenizer_SPM：sentencepiece 风格（TinyLlama/Llama-2）
// ---------------------------------------------------------------------------
static void test_tokenizer_spm() {
    std::vector<VocabEntry> v = {
        {"<unk>", 0.0f, 2},
        {"<s>",   0.0f, 3},   // id=1 = bos
        {"</s>",  0.0f, 3},
        {"\xE2\x96\x81", -1.0f, 1},  // id=3 "▁"
        {"a",     0.0f, 1},   // id=4
        {"b",     0.0f, 1},   // id=5
        {"ab",    5.0f, 1},   // id=6
        {"\xE2\x96\x81" "a", 6.0f, 1},   // id=7 "▁a"
        {"\xE2\x96\x81" "ab", 10.0f, 1}, // id=8 "▁ab"
    };
    add_byte_tokens(v);
    std::string path = write_vocab_gguf("llama", "default", v, 1, 2, true);

    GgufFile f = GgufFile::load(path);
    Tokenizer tok = Tokenizer::load(f);

    // "ab" -> [bos, "▁ab"]：句首补空格 + 最高优先级合并
    CHECK_EQ(tok.encode("ab"), (std::vector<int>{1, 8}));
    // "aab" -> [bos, "▁a", "ab"]：▁a(6分) 先于 ab(5分) 合并
    CHECK_EQ(tok.encode("aab"), (std::vector<int>{1, 7, 6}));
    // 未知字符 x 走 BYTE 兜底（byte token 基址 9，'x'=120 -> 129）
    CHECK_EQ(tok.encode("x"), (std::vector<int>{1, 3, 9 + 120}));
    // add_special=false 不加 bos，但 SPM 的句首空格前缀不受影响
    CHECK_EQ(tok.encode("ab", false), (std::vector<int>{8}));
    // 解码：跳过控制 token，▁ 还原为空格
    CHECK_EQ(tok.decode({1, 8}), std::string(" ab"));

    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// TestTokenizer_BPE_Qwen2：gpt2 词表 + qwen2 预分词（bytes_to_unicode 映射空格）
// ---------------------------------------------------------------------------
static void test_tokenizer_bpe_qwen2() {
    std::vector<VocabEntry> v = {
        {"<|endoftext|>", 0.0f, 3},  // id=0
        {"a",   0.0f, 1},   // id=1
        {"b",   0.0f, 1},   // id=2
        {"ab",  5.0f, 1},   // id=3
        {"abc", 10.0f, 1},  // id=4
        {"c",   0.0f, 1},   // id=5
    };
    // id=6: "Ġ"（bytes_to_unicode 映射后的空格）
    std::string gs;
    utf8_encode(byte_to_unicode_cp(' '), gs);
    v.push_back({gs, -1.0f, 1});
    add_byte_tokens_mapped(v);

    std::string path = write_vocab_gguf("gpt2", "qwen2", v, 0, 0, false);
    GgufFile f = GgufFile::load(path);
    Tokenizer tok = Tokenizer::load(f);

    // 词内合并到底
    CHECK_EQ(tok.encode("abc"), (std::vector<int>{4}));
    // qwen2 数字单切：a 1 b 三个词，词间绝不合并
    // （'1' 是可打印字节，映射后还是 "1"，其 BYTE token 基址 7，id = 7+49=56）
    CHECK_EQ(tok.encode("a1b"), (std::vector<int>{1, 7 + '1', 2}));
    // "ab abc"：映射后是 "abĠabc"，全部落在字母分支成一词，
    // 合并过程: ab, Ġ, abc -> [ab, Ġ, abc]
    CHECK_EQ(tok.encode("ab abc"), (std::vector<int>{3, 6, 4}));
    // 解码：BYTE/NORMAL token 都从映射空间逆回原始字节
    CHECK_EQ(tok.decode({3, 6, 4}), std::string("ab abc"));

    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// TestTokenizer_BPE_Llama3：gpt2 词表 + llama-bpe 预分词（bytes_to_unicode，加 BOS）
// ---------------------------------------------------------------------------
static void test_tokenizer_bpe_llama3() {
    std::vector<VocabEntry> v = {
        {"<s>", 0.0f, 3},   // id=0 = bos
        {"a",   0.0f, 1},   // id=1
        {"ab",  2.0f, 1},   // id=2
    };
    // id=3: "Ġ"  id=4: "Ġa"（高分，优先合并）
    std::string gs, gsa;
    utf8_encode(byte_to_unicode_cp(' '), gs);
    utf8_encode(byte_to_unicode_cp(' '), gsa);
    gsa += "a";
    v.push_back({gs,  -1.0f, 1});   // id=3
    v.push_back({gsa,  5.0f, 1});   // id=4
    add_byte_tokens_mapped(v);

    std::string path = write_vocab_gguf("gpt2", "llama-bpe", v, 0, 0, true);
    GgufFile f = GgufFile::load(path);
    Tokenizer tok = Tokenizer::load(f);

    // 句首无空格，BOS + "a"
    CHECK_EQ(tok.encode("a"), (std::vector<int>{0, 1}));
    // "a a"：映射为 "aĠa"，Ġ 在字母分支内与 a 合并成 "Ġa"
    CHECK_EQ(tok.encode("a a"), (std::vector<int>{0, 1, 4}));
    // 解码逆映射回原始字节
    CHECK_EQ(tok.decode({0, 1, 4}), std::string("a a"));

    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// TestTokenizer_Errors：缺 key / 不支持类型
// ---------------------------------------------------------------------------
static void test_tokenizer_errors() {
    {   // 缺 tokenizer.ggml.tokens
        Writer w;
        w.u32(0x46554747u); w.u32(3); w.u64(0); w.u64(1);
        w.kv_header("tokenizer.ggml.model", 8); w.str("llama");
        std::string path = write_temp(w.buf(), "err1");
        GgufFile f = GgufFile::load(path);
        bool thrown = false;
        try { Tokenizer::load(f); } catch (const std::exception &) { thrown = true; }
        CHECK(thrown);
        std::remove(path.c_str());
    }
    {   // 不支持的词表类型
        Writer w;
        w.u32(0x46554747u); w.u32(3); w.u64(0); w.u64(3);
        w.kv_header("tokenizer.ggml.model", 8); w.str("rwkv");
        w.kv_header("tokenizer.ggml.tokens", 9); w.u32(8); w.u64(1); w.str("x");
        w.kv_header("tokenizer.ggml.pre", 8); w.str("default");
        std::string path = write_temp(w.buf(), "err2");
        GgufFile f = GgufFile::load(path);
        bool thrown = false;
        try { Tokenizer::load(f); } catch (const std::exception &) { thrown = true; }
        CHECK(thrown);
        std::remove(path.c_str());
    }
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// TestTokenizer_RealFile：真实词表冒烟
// ---------------------------------------------------------------------------
static void test_real_file() {
    const char * path = std::getenv("TEST_TOKENIZER_REAL_FILE");
    if (!path || !*path) {
        std::cout << "  [skip] 未设置 TEST_TOKENIZER_REAL_FILE，跳过真实词表冒烟\n";
        return;
    }
    GgufFile f = GgufFile::load(path);
    Tokenizer tok = Tokenizer::load(f);
    std::cout << "  vocab: " << path << "  n_tokens=" << tok.n_tokens()
              << "  bos=" << tok.bos_id() << "  eos=" << tok.eos_id() << "\n";

    const std::string sample = "Hello world";
    auto ids = tok.encode(sample);
    std::string back = tok.decode(ids);
    std::cout << "  encode(\"" << sample << "\") -> " << ids.size()
              << " tokens, decode -> \"" << back << "\"\n";
    // 往返：SPM 词表会带句首空格前缀，BPE 词表应完全还原
    CHECK(back == sample || back == " " + sample);

    if (g_fail == 0) { ++g_pass; }
}

int main() {
    test_pretokenize();
    test_tokenizer_spm();
    test_tokenizer_bpe_qwen2();
    test_tokenizer_bpe_llama3();
    test_tokenizer_errors();
    test_real_file();

    std::cout << "passed=" << g_pass << " failed=" << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
