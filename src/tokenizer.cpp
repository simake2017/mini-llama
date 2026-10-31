// 分词器实现：词表加载 + SPM/BPE 编码 + 解码

#include "tokenizer.h"

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace minillama {

// U+2581 Lower One Eighth Block，SPM/Llama3 词表里表示空格的转义符
static const char * ESCAPED_SPACE = "\xE2\x96\x81";

// 把一个 BYTE token 的文本解析成原始字节，三种格式都认：
//   "<0x41>" 十六进制格式（llama-spm/gemma）
//   单原始字节（旧 BPE 词表）
//   bytes_to_unicode 映射字符（qwen2/gpt2/llama3 BPE）
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

static void replace_all(std::string & s, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// ---------------------------------------------------------------------------
// 加载
// ---------------------------------------------------------------------------

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
            // 【简化】旧版 GGUF 可能没有 token_type 数组：按长度猜
            // （单字节 token 就是 BYTE，其余 NORMAL）——llama.cpp 早期也这么兜底
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

int Tokenizer::find_by_text(const std::string & text) const {
    auto it = text_to_id_.find(text);
    return it == text_to_id_.end() ? -1 : it->second;
}

// 单字节 -> token id 的兜底查找链（对应 llama.cpp 的 llama_vocab::byte_to_token）
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

// ---------------------------------------------------------------------------
// 编码共用核心：symbol 链表 + 按 score 优先的相邻合并
// 【对照】llama.cpp 的 llm_tokenizer_spm_session / llm_tokenizer_bpe_session
//         各有一份几乎相同的实现（它俩没合并是历史原因），我们合成一份。
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 编码
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 解码
// ---------------------------------------------------------------------------

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
