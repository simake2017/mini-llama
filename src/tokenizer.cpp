// ============================================================================
// ① 定位：分词器的实现 —— 词表加载、文本改写、按 score 合并、字节兜底、解码
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:1921:llama_vocab::impl::load（词表加载）
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:117:llm_tokenizer_spm_session::tokenize
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:202:try_add_bigram（按 score 合并相邻符号）
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:176:resegment（拆不成词时的字节兜底）
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:4092:llama_vocab::token_to_piece（解码）
//
// ③ 动机：
//    【对照】上游 SPM 与 BPE 各有一份 session（llama-vocab.cpp:117 / :603），两份实现
//       几乎相同 —— 上游自己在 llama-vocab.cpp:246 标了 TODO 说该合并复用。本仓库合成
//       一份 merge_tokenize，两条路径只在「要不要预分词」这一处分叉。
//    【简化】不做 special token 片段提取（tokenizer_st_partition）：Llama 主干 prompt
//       不含内嵌特殊串；承重结论见 docs/constraints/04-对齐约束.md §2.1 组 B。
//    【陷阱】上游读 add_bos_token / add_eos_token，本实现读 add_bos / add_eos —— GGUF 里
//       普遍写的是上游那个名字，所以词表级的覆盖实际读不到（待拍板，docs/02-tokenizer.md §15）。
//
// ④ demo + 多方样例：
//    🟢 合成 SPM 词表：encode("ab") → [1, 8]（BOS + "▁ab"）
//    🟡 真实 TinyLlama：encode("Hello world") → [1, 15043, 8879, 2096, 29881]
//    🔴 词表覆盖不到的多字节字符：SPM 路径按 UTF-8 字符逐个拆，单字节才落 BYTE token
//    调用方：src/tokenizer.h 的 Tokenizer::load / encode / decode 三个入口
// ============================================================================

#include "tokenizer.h"

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace minillama {

// ============================================================================
// ① 定位：SPM 词表里代表空格的转义符 ▁（U+2581，UTF-8 三字节 E2 96 81）
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:3285:llama_escape_whitespace
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:3357:llama_vocab::impl::tokenize 的转义调用点
//
// ③ 动机：
//    【概念】SPM 词表里不存原始空格 —— 空格是词边界标记，必须可见且不会与词内字符
//       混淆，所以用 U+2581 这个「方块」占位：编码时 ' ' → ▁、解码时 ▁ → ' '。
//    【陷阱】这条互转**只对 SPM 词表成立**：BPE 词表存的是原始空格，或经
//       bytes_to_unicode 映射后的 Ġ（U+0120）。两种转义不能混用。
//
// ④ demo + 多方样例：
//    🟢 合成 SPM 词表：encode("ab") → [1, 8]，而 token 8 的文本就是 "▁ab"
//    🟡 真实 TinyLlama：encode("Hello world", false) 的首个 token id=15043，文本 "▁Hello"
//    🔴 真实 TinyLlama 的 decode 会把 ▁ 还原成空格：decode({15043, 8879, 2096, 29881})
//       → " Hello world"（句首那个空格正是 ▁ 还原出来的）
//    调用方：src/tokenizer.cpp 的 encode / decode 两端的 replace_all
// ============================================================================
static const char * ESCAPED_SPACE = "\xE2\x96\x81";

// 同一句「代表一个字节」在真实词表里有三种写法：<0xXX> 十六进制（llama-spm / gemma）、
// 单原始字节（旧 BPE 词表）、bytes_to_unicode 映射字符（qwen2 / gpt2 / llama3）。按文本
// 形状识别而不是按词表类型分派，才能同时吃下三个家族（上游 byte_to_token 按类型分派，
// llama-vocab.cpp:3852；差异见 docs/02-tokenizer.md §17.2）。
static int byte_token_text_to_raw(const std::string & text) {
    if (text.size() == 7 && text[0] == '<' && text[1] == '0' && text[2] == 'x' &&
        text[6] == '>') {
        auto hexv = [](char c) -> int {
            if (c >= '0' && c <= '9') { return c - '0'; }
            if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
            if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
            return -1;
        };
        int hi = hexv(text[3]), lo = hexv(text[4]);
        if (hi >= 0 && lo >= 0) { return hi * 16 + lo; }
        return -1;
    }
    if (text.size() == 1) { return static_cast<uint8_t>(text[0]); }
    size_t len;
    uint32_t cp = utf8_decode(text.data(), text.size(), &len);
    if (len == text.size()) {
        return unicode_cp_to_byte(cp);
    }
    return -1;
}

// 就地替换全部出现处；转义与还原共用它，所以语义必须是「全部」而不是「第一处」。
static void replace_all(std::string & s, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// ============================================================================
// ① 定位：词表加载 —— 判类型、读三张平行数组、建正反向索引与字节映射、读特殊 token 开关
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:1921:llama_vocab::impl::load
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:2415:llama_vocab::impl::load 的 scores 长度校验
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:3089:llama_vocab::impl::token_to_byte
//
// ③ 动机：
//    【概念】判类型必须最先做：tokenizer.ggml.model 决定后面每个分支（补不补空格、
//       转不转义、做不做字节映射、要不要预分词），其余步骤都依赖它。
//    【陷阱】scores / token_type 数组上游会校验长度并抛错（llama-vocab.cpp:2415 /
//       :2425），本实现直接按下标读 —— 长度不一致时是越界读，属已知缺陷
//       （docs/02-tokenizer.md §16.2 的「结构档的偏离」表）。
//    【对照】上游按**词表类型**分派查 BYTE token，本实现按**文本形状**识别 —— 三种
//       BYTE 文本格式在真实词表里都出现过（docs/02-tokenizer.md §17.2）。
//
// ④ demo + 多方样例：
//    🟢 合成词表（tests/test_tokenizer.cpp）：model=llama → SPM、add_bos=1、256 个 BYTE token
//    🟡 真实 TinyLlama：32000 个 token、bos=1、eos=2、add_bos=1、scores 与 token_type 都在
//       （lines/load/vocab_load_line.cpp 第 3 / 6 跳）
//    🔴 词表类型不是 llama / gpt2 → 抛「不支持的词表类型 tokenizer.ggml.model: rwkv」
//    调用方：src/main.cpp；lines/load/vocab_load_line.cpp:91
// ============================================================================
Tokenizer Tokenizer::load(const GgufFile & gguf) {
    Tokenizer tok;

    // ① 词表类型（决定编码路径）
    const std::string model = gguf.get_string("tokenizer.ggml.model");
    if (model == "llama") {
        tok.type_ = VocabType::SPM;
        tok.add_space_prefix_ = true;   // llama.cpp: SPM 默认句首补空格
        tok.add_bos_ = true;            // llama.cpp: SPM 默认加 BOS
    } else if (model == "gpt2") {
        tok.type_ = VocabType::BPE;
        tok.escape_whitespaces_ = false; // qwen2/gpt2 词表是原始空格
    } else {
        throw std::runtime_error("不支持的词表类型 tokenizer.ggml.model: " + model +
                                 "（本单元只支持 llama/gpt2）");
    }

    // ② 预分词器类型（BPE 路径需要）
    if (tok.type_ == VocabType::BPE) {
        const std::string pre = gguf.get_string("tokenizer.ggml.pre");
        if (pre == "llama-bpe" || pre == "llama3") {
            tok.pre_ = PreTokenizer::LLAMA3;
            tok.byte_encode_ = true;         // llama3 词表同为 bytes_to_unicode 格式
            tok.add_bos_ = true;             // llama3 词表约定加 BOS
        } else if (pre == "qwen2") {
            tok.pre_ = PreTokenizer::QWEN2;
            tok.byte_encode_ = true;         // qwen2 词表是 bytes_to_unicode 格式
        } else {
            throw std::runtime_error("不支持的预分词器 tokenizer.ggml.pre: " + pre +
                                     "（本单元只支持 llama-bpe/qwen2）");
        }
    }

    // ③ token 列表 / 分数 / 类型
    const auto & token_arr = gguf.get_array("tokenizer.ggml.tokens");
    if (token_arr.empty()) {
        throw std::runtime_error("tokenizer.ggml.tokens 为空");
    }

    const GgufArray * scores  = gguf.find("tokenizer.ggml.scores")
        ? &gguf.get_array("tokenizer.ggml.scores") : nullptr;
    const GgufArray * types   = gguf.find("tokenizer.ggml.token_type")
        ? &gguf.get_array("tokenizer.ggml.token_type") : nullptr;

    tok.tokens_.reserve(token_arr.size());
    for (size_t i = 0; i < token_arr.size(); i++) {
        TokenData d;
        d.text  = std::get<std::string>(token_arr[i].v);
        d.score = scores ? std::get<float>((*scores)[i].v) : 0.0f;
        if (types) {
            d.type = static_cast<TokenType>(std::get<int32_t>((*types)[i].v));
        } else {
            // 【简化】旧版 GGUF 可能没有 token_type 数组：按长度猜（单字节 token 就是
            // BYTE，其余 NORMAL）。承重判据：砍掉它，「token 有类型」这个概念塌了吗？
            // 没塌 —— 类型数组本身在 GGUF 里存在，这里只是给老文件兜底。
            d.type = (d.text.size() == 1) ? TokenType::BYTE : TokenType::NORMAL;
        }
        tok.tokens_.push_back(std::move(d));
    }

    // ④ 反向索引
    for (size_t i = 0; i < tok.tokens_.size(); i++) {
        // 重复文本保留第一个 id（与 llama.cpp 一致：token_to_id 用 [] 覆盖为最后？
        //   ——实际 llama.cpp 也是覆盖，但词表本身无重复，这里保留首个更保守）
        tok.text_to_id_.emplace(tok.tokens_[i].text, static_cast<int>(i));
    }

    // ⑤ BYTE token <-> 原始字节 双向映射（任意字节的兜底编码通道）
    // 真实词表存在三种 BYTE token 文本格式，见 byte_token_text_to_raw
    tok.id_to_byte_.assign(tok.tokens_.size(), -1);
    for (size_t i = 0; i < tok.tokens_.size(); i++) {
        if (tok.tokens_[i].type != TokenType::BYTE) { continue; }
        int b = byte_token_text_to_raw(tok.tokens_[i].text);
        if (b >= 0) {
            tok.byte_to_id_[b] = static_cast<int>(i);
            tok.id_to_byte_[i] = b;
            tok.has_byte_map_ = true;
        }
    }

    // ⑥ 特殊 token id 与开关（缺失时容忍，语义为"无"）
    auto opt_u32 = [&](const char * key, int & dst) {
        if (const GgufValue * v = gguf.find(key)) {
            dst = static_cast<int>(std::get<uint32_t>(v->v));
        }
    };
    opt_u32("tokenizer.ggml.bos_token_id", tok.bos_id_);
    opt_u32("tokenizer.ggml.eos_token_id", tok.eos_id_);
    if (const GgufValue * v = gguf.find("tokenizer.ggml.add_bos")) {
        tok.add_bos_ = std::get<bool>(v->v);   // BOOL 解析为 bool（不是 u8）
    }
    if (const GgufValue * v = gguf.find("tokenizer.ggml.add_eos")) {
        tok.add_eos_ = std::get<bool>(v->v);
    }

    return tok;
}

// 反向索引是合并循环的热路径：每试一次相邻合并就查一次表。
int Tokenizer::find_by_text(const std::string & text) const {
    auto it = text_to_id_.find(text);
    return it == text_to_id_.end() ? -1 : it->second;
}

// ============================================================================
// ① 定位：字节兜底的查找链 —— 任意原始字节 → 一个能代表它的 token id
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:3852:llama_vocab::byte_to_token
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:176:resegment（调用方，逐字节兜底）
//
// ③ 动机：
//    【概念】词表覆盖不到的文本必须有去处，否则整段文本会丢字符。查找分三级：
//       显式 BYTE 映射 → bytes_to_unicode 映射字符本身 → 单字节原始文本。
//    【对照】上游按**词表类型**分派（SPM/UGM 查 <0xXX> 再退到单字节；WPM/BPE 查
//       unicode_byte_to_utf8(ch)；PLaMo2 查 <0x%02X>），本实现按**文本形状**识别 ——
//       llama3 词表根本没有显式 BYTE token，只能退到「拿映射字符查表」这一级。
//    【陷阱】三级都落空时返回 -1，调用方（merge_tokenize）会静默丢弃该字节 ——
//       不抛异常，因为「词表缺通道」不是调用方的错。
//
// ④ demo + 多方样例：
//    🟢 合成 SPM 词表：encode("x") → [1, 3, 129]，129 = BYTE token 基址 9 + 'x'(120)
//    🟡 真实 TinyLlama 没有显式 BYTE token，走第二级（映射字符查表）：
//       encode("\x01\x02") → [1, 29871, 30107, 31468]
//    🔴 多字节 UTF-8 字符走不到查找链：SPM 路径按 UTF-8 字符取字节（l == 1 才取），
//       3 字节字符的 3 个字节被整体丢弃 —— 合成 SPM 词表 encode("你") → [1, 3]；
//       上游 resegment 是逐字节兜底（llama-vocab.cpp:176）
//    调用方：src/tokenizer.cpp 的 merge_tokenize 收尾段
// ============================================================================
int Tokenizer::byte_token(int b) const {
    if (has_byte_map_) {
        int id = byte_to_id_[b];
        if (id >= 0 && id < static_cast<int>(id_to_byte_.size()) && id_to_byte_[id] == b) {
            return id;
        }
    }
    // llama3 式：没有显式 BYTE token，但 bytes_to_unicode 映射字符本身在词表里
    std::string mapped;
    utf8_encode(byte_to_unicode_cp(static_cast<uint8_t>(b)), mapped);
    int id = find_by_text(mapped);
    if (id >= 0) { return id; }
    // 最原始的形态：单字节原始文本
    return find_by_text(std::string(1, static_cast<char>(b)));
}

// SPM / BPE 共用的合并核心：symbol 链表 + 按 score 优先的相邻合并。
// 【对照】上游 SPM 与 BPE 各有一份几乎相同的实现（llama-vocab.cpp:117 / :603），
//         本仓库合成一份；两者的分歧点只有「BPE 要先切词」。
void Tokenizer::merge_tokenize(const std::string & text, std::vector<int> & out) const {
    // symbol = 文本中的一个 UTF-8 字符；用数组 + prev/next 下标模拟链表，
    // 合并时只改下标和长度，物理字符串永不移动（【性能】与 llama.cpp 相同手法）
    const size_t n = [&] {
        size_t cnt = 0, i = 0;
        while (i < text.size()) { i += utf8_len(text[i]); cnt++; }
        return cnt;
    }();
    if (n == 0) { return; }

    std::vector<size_t> start(n), len(n, 0);
    std::vector<int> prev(n), next(n);
    {
        size_t i = 0, cnt = 0;
        while (i < text.size()) {
            size_t l = utf8_len(text[i]);
            start[cnt] = i;
            len[cnt]   = l;
            prev[cnt]  = static_cast<int>(cnt) - 1;
            next[cnt]  = (i + l >= text.size()) ? -1 : static_cast<int>(cnt) + 1;
            i += l;
            cnt++;
        }
    }

    // 候选 bigram：合并后文本是词表中 token 的相邻 symbol 对
    struct Bigram {
        float  score;
        size_t size;   // 合并时的总字节数（陈旧检测用）
        int    left, right;
        uint64_t seq;  // 入队序号（同分时让先入队的先合并，保证确定性）
        bool operator<(const Bigram & o) const {  // priority_queue 是最大堆
            if (score != o.score) { return score < o.score; }
            if (size  != o.size)  { return size < o.size; }
            return seq > o.seq;
        }
    };

    std::priority_queue<Bigram> queue;
    uint64_t seq = 0;
    auto try_add = [&](int l, int r) {
        if (l < 0 || r < 0) { return; }
        const size_t total = len[l] + len[r];
        const std::string merged = text.substr(start[l], total);
        int id = find_by_text(merged);
        if (id < 0) { return; }
        queue.push(Bigram{tokens_[id].score, total, l, r, seq++});
    };

    for (size_t i = 1; i < n; i++) {
        try_add(static_cast<int>(i) - 1, static_cast<int>(i));
    }

    // 【概念】BPE 合并循环：每次取出全局最优 pair 合并，直到没有可合并的 pair。
    // 【陷阱】堆里的条目可能在弹出前就已失效（相邻 symbol 被别的合并吞掉），
    //         所以弹出时必须重新校验 size 是否还等于入队时的值。
    while (!queue.empty()) {
        Bigram bg = queue.top();
        queue.pop();
        if (len[bg.left] == 0 || len[bg.right] == 0 ||
            len[bg.left] + len[bg.right] != bg.size) {
            continue;  // 陈旧条目
        }
        // right 并入 left，right 从链表摘除
        len[bg.left] += len[bg.right];
        len[bg.right] = 0;
        next[bg.left] = next[bg.right];
        if (next[bg.right] >= 0) { prev[next[bg.right]] = bg.left; }
        try_add(prev[bg.left], bg.left);
        try_add(bg.left, next[bg.left]);
    }

    // 收尾：沿链表输出。symbol 是 token 就直接输出；否则逐字节落到 BYTE token
    for (int i = 0; i != -1; i = next[i]) {
        const std::string piece = text.substr(start[i], len[i]);
        int id = find_by_text(piece);
        if (id >= 0) {
            out.push_back(id);
            continue;
        }
        // 词表未覆盖：逐 UTF-8 字符还原成原始字节，走字节兜底
        size_t k = 0;
        while (k < len[i]) {
            size_t l;
            uint32_t cp = utf8_decode(text.data() + start[i] + k, len[i] - k, &l);
            int byte = byte_encode_ ? unicode_cp_to_byte(cp)
                                    : (l == 1 ? static_cast<uint8_t>(text[start[i] + k]) : -1);
            if (byte >= 0) {
                int bid = byte_token(byte);
                if (bid >= 0) { out.push_back(bid); }
            }
            k += l;
        }
    }
}

// ============================================================================
// ① 定位：文本 → token id 序列 —— 加 BOS、改写文本、切词、按 score 合并、加 EOS
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:117:llm_tokenizer_spm_session::tokenize
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:603:llm_tokenizer_bpe_session::tokenize
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:3348:llama_vocab::impl::tokenize 的 SPM 补空格判据
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:3285:llama_escape_whitespace
//
// ③ 动机：
//    【陷阱】**SPM 句首补空格与 add_special 无关**：add_special 只决定拼不拼 BOS/EOS，
//       补空格由词表开关 add_space_prefix_ 单独决定。上游判据是 add_space_prefix &&
//       is_prev_special（llama-vocab.cpp:3348），而 is_prev_special 初值 true（:3335）—— 与 add_special 无关。
//    【历史】SPM 用 ▁ 标词边界，句首不补空格，首词与后续词就分不开 —— 这条
//       sentencepiece 规范 llama.cpp 至今保留。
//    【对照】上游把文本切成 special token 片段与普通文本两类分别处理，本实现整段当一段（不切片段，已登记为可砍，docs/constraints/04-对齐约束.md §2.1 组 B）。
//
// ④ demo + 多方样例：
//    🟢 最小：合成 SPM 词表 encode("ab") → [1, 8]；encode("ab", false) → [8]
//       —— 仍带句首空格，因为补空格与 add_special 无关
//    🟡 典型：真实 TinyLlama encode("Hello world") → [1, 15043, 8879, 2096, 29881]
//    🔴 词表覆盖不到的多字节字符在 SPM 路径上被整体丢弃（实测 encode("😀") → [1, 29871]）；
//       BPE 路径因为先做了 bytes_to_unicode 映射，字符粒度就是 1 个原始字节（合成 qwen2 词表实测 encode("你") → 3 个 BYTE token）
//    调用方：src/main.cpp；lines/text/tokenize_line.cpp:80
// ============================================================================
std::vector<int> Tokenizer::encode(const std::string & text, bool add_special) const {
    std::vector<int> out;

    if (add_special && add_bos_ && bos_id_ >= 0) {
        out.push_back(bos_id_);
    }

    std::string work = text;
    if (add_space_prefix_ && !work.empty()) {
        work = " " + work;   // llama.cpp：SPM 句首补空格（与 add_special 无关）
    }
    if (escape_whitespaces_) {
        replace_all(work, " ", ESCAPED_SPACE);
    }
    if (byte_encode_) {
        // qwen2/gpt2：原始字节 -> bytes_to_unicode 可见字符
        std::string mapped;
        mapped.reserve(work.size());
        for (uint8_t b : work) {
            utf8_encode(byte_to_unicode_cp(b), mapped);
        }
        work = std::move(mapped);
    }

    if (type_ == VocabType::SPM) {
        // SPM：整段一次性合并（不切词）
        merge_tokenize(work, out);
    } else {
        // BPE：先正则切词，词内合并（词间不合并！）
        for (const auto & piece : pre_tokenize(work, pre_)) {
            merge_tokenize(piece, out);
        }
    }

    if (add_special && add_eos_ && eos_id_ >= 0) {
        out.push_back(eos_id_);
    }
    return out;
}

// ============================================================================
// ① 定位：token id 序列 → 文本 —— 按类型分派渲染，再把 ▁ 换回空格
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:4092:llama_vocab::token_to_piece
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:3519:llama_vocab::impl::token_to_piece
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:3584:llama_vocab::impl::token_to_piece 的 ▁ 还原
//
// ③ 动机：
//    【对照】上游跳过掩码只有 UNKNOWN | CONTROL（llama-vocab.cpp:3521 的 attr_special），
//       本实现多跳过 UNUSED / UNDEFINED 两种上游不跳的类型 —— 真实 GGUF 里 <s> 常被
//       标成 USER_DEFINED(4)，上游也照渲染，两边一致。
//    【陷阱】▁ ↔ 空格是「编码空间 ↔ 文本空间」的边界；BPE 词表走 bytes_to_unicode
//       逆映射（空格 ↔ U+0120 Ġ），SPM 词表走 ▁ ↔ ' '，两套转义不能混用。
//
// ④ demo + 多方样例：
//    🟢 最小：合成 SPM 词表 decode({1, 8}) → " ab"（BOS 是 CONTROL 被跳过，"▁ab" 还原）
//    🟡 典型：真实 TinyLlama decode(encode("Hello world")) → "<s> Hello world"
//    🔴 越界 id 直接跳过、不抛异常 → decode({999999999}) 返回空串
//       （lines/text/detokenize_line.cpp:106 断言了这一点）
//    调用方：src/main.cpp / :103；lines/text/detokenize_line.cpp:120
// ============================================================================
std::string Tokenizer::decode(const std::vector<int> & ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(tokens_.size())) { continue; }
        const TokenData & d = tokens_[id];
        switch (d.type) {
            case TokenType::NORMAL:
            case TokenType::USER_DEFINED:
                if (byte_encode_) {
                    // qwen2/gpt2 词表 token 文本在 bytes_to_unicode 空间，逆映射回原始字节
                    size_t k = 0;
                    while (k < d.text.size()) {
                        size_t l;
                        uint32_t cp = utf8_decode(d.text.data() + k, d.text.size() - k, &l);
                        int b = unicode_cp_to_byte(cp);
                        if (b >= 0) { out += static_cast<char>(b); }
                        else        { out.append(d.text, k, l); }
                        k += l;
                    }
                } else {
                    out += d.text;
                }
                break;
            case TokenType::BYTE: {
                // 还原原始字节（三种文本格式都试）
                int b = id_to_byte_[id];
                if (b < 0) { b = byte_token_text_to_raw(d.text); }
                if (b >= 0) { out += static_cast<char>(b); }
                break;
            }
            default:
                // CONTROL / UNKNOWN / UNUSED / UNDEFINED：跳过
                // （llama.cpp 默认行为：不渲染特殊 token）
                break;
        }
    }
    if (escape_whitespaces_) {
        replace_all(out, ESCAPED_SPACE, " ");
    }
    return out;
}

} // namespace minillama
