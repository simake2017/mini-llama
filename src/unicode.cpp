// UTF-8 / Unicode / 预分词扫描器实现

#include "unicode.h"

#include "unicode-tables.h"

#include <algorithm>

namespace minillama {

// ---------------------------------------------------------------------------
// UTF-8 解码
// ---------------------------------------------------------------------------

size_t utf8_len(char first) {
    const uint8_t b = static_cast<uint8_t>(first);
    if (b < 0x80) { return 1; }
    if ((b >> 5) == 0x6) { return 2; }
    if ((b >> 4) == 0xE) { return 3; }
    if ((b >> 3) == 0x1E) { return 4; }
    return 1; // 非法前导字节，按单字节容错前进
}

uint32_t utf8_decode(const char * s, size_t avail, size_t * len) {
    const uint8_t b0 = static_cast<uint8_t>(s[0]);
    if (b0 < 0x80) { *len = 1; return b0; }

    const size_t need = utf8_len(s[0]);
    if (need > avail) { *len = 1; return 0xFFFD; } // 截断的序列

    // 多字节序列：后续字节必须是 10xxxxxx
    for (size_t i = 1; i < need; i++) {
        if ((static_cast<uint8_t>(s[i]) & 0xC0) != 0x80) { *len = 1; return 0xFFFD; }
    }

    uint32_t cp;
    switch (need) {
        case 2: cp = b0 & 0x1F; break;
        case 3: cp = b0 & 0x0F; break;
        default: cp = b0 & 0x07; break;
    }
    for (size_t i = 1; i < need; i++) {
        cp = (cp << 6) | (static_cast<uint8_t>(s[i]) & 0x3F);
    }
    // 超范围 / 过度编码（如用 3 字节编码 ASCII）视为非法
    if (cp > 0x10FFFF || (need == 2 && cp < 0x80) || (need == 3 && cp < 0x800)) {
        *len = 1; return 0xFFFD;
    }
    *len = need;
    return cp;
}

void utf8_encode(uint32_t cp, std::string & out) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// ---------------------------------------------------------------------------
// Unicode 类别：区间表二分查找
// 【性能】622+133 个区间，二分 10 次比较；每个字符只查一次表，
//         分词开销相对 BPE 合并循环可以忽略。
// ---------------------------------------------------------------------------

static bool in_ranges(const uint32_t ranges[][2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < ranges[mid][0]) { hi = mid; }
        else if (cp > ranges[mid][1]) { lo = mid + 1; }
        else { return true; }
    }
    return false;
}

bool is_unicode_letter(uint32_t cp) {
    return in_ranges(UNICODE_LETTER_RANGES, UNICODE_LETTER_RANGES_n, cp);
}

bool is_unicode_number(uint32_t cp) {
    return in_ranges(UNICODE_NUMBER_RANGES, UNICODE_NUMBER_RANGES_n, cp);
}

bool is_ascii_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// ---------------------------------------------------------------------------
// GPT-2 bytes_to_unicode
// 【对照】llama.cpp ggml/src/unicode.cpp unicode_byte_to_utf8
// ---------------------------------------------------------------------------

static bool is_identity_byte(uint8_t b) {
    return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
}

uint32_t byte_to_unicode_cp(uint8_t b) {
    if (is_identity_byte(b)) { return b; }
    uint32_t n = 0;
    for (uint8_t k = 0; k < b; k++) {
        if (!is_identity_byte(k)) { n++; }
    }
    return 256 + n;
}

int unicode_cp_to_byte(uint32_t cp) {
    if (cp < 256 && is_identity_byte(cp)) { return static_cast<int>(cp); }
    if (cp >= 256 && cp < 324) {
        uint32_t n = cp - 256;
        for (uint8_t k = 0; k < 256; k++) {
            if (!is_identity_byte(k)) {
                if (n == 0) { return k; }
                n--;
            }
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// 预分词扫描器
// ---------------------------------------------------------------------------

namespace {

struct CharInfo {
    uint32_t cp;      // 码点
    size_t   len;     // UTF-8 字节数
    bool     letter;  // \p{L}
    bool     number;  // \p{N}
    bool     space;   // \s（ASCII）
};

// 把文本解码成字符信息数组；ASCII 快路径避免每次查表
std::vector<CharInfo> decode(const std::string & text) {
    std::vector<CharInfo> out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        CharInfo ci {};
        if (static_cast<uint8_t>(text[i]) < 0x80) {
            ci.cp = static_cast<uint8_t>(text[i]);
            ci.len = 1;
            ci.letter = ('a' <= ci.cp && ci.cp <= 'z') || ('A' <= ci.cp && ci.cp <= 'Z');
            ci.number = '0' <= ci.cp && ci.cp <= '9';
            ci.space  = is_ascii_space(text[i]);
        } else {
            ci.cp = utf8_decode(text.data() + i, text.size() - i, &ci.len);
            ci.letter = is_unicode_letter(ci.cp);
            ci.number = is_unicode_number(ci.cp);
        }
        out.push_back(ci);
        i += ci.len;
    }
    return out;
}

// 收缩语: 's 't 're 've 'm 'll 'd（大小写不敏感）。返回匹配字符数，0 表示不匹配。
size_t match_contraction(const std::vector<CharInfo> & cs, size_t i) {
    static const char * words[] = {"'ll", "'re", "'ve", "'s", "'t", "'m", "'d"};
    if (cs[i].cp != '\'') { return 0; }
    for (const char * w : words) {
        size_t n = std::char_traits<char>::length(w);
        if (i + n > cs.size()) { continue; }
        bool ok = true;
        for (size_t k = 1; k < n; k++) {
            char want = w[k];
            char got  = static_cast<char>(cs[i + k].cp);
            if (cs[i + k].len != 1 || (got | 0x20) != want) { ok = false; break; }
        }
        if (ok) { return n; }
    }
    return 0;
}

} // namespace

std::vector<std::string> pre_tokenize(const std::string & text, PreTokenizer pre) {
    std::vector<std::string> out;
    if (text.empty()) { return out; }

    const auto cs = decode(text);
    const size_t n = cs.size();

    // 预计算每个字符的原文字节偏移，substr 用
    std::vector<size_t> byte_off(n + 1, 0);
    for (size_t k = 0; k < n; k++) { byte_off[k + 1] = byte_off[k] + cs[k].len; }

    auto emit = [&](size_t begin, size_t end) {
        if (end > begin) { out.emplace_back(text.substr(byte_off[begin], byte_off[end] - byte_off[begin])); }
    };

    // 【概念】正则交替的分支优先级：按声明顺序尝试，先命中先走。
    // 6 个分支（llama3 / qwen2 同款，仅分支 ③ 的数字分组不同）:
    //   ① 's|'t|'re|'ve|'m|'ll|'d        英文缩略
    //   ② [^\r\n\p{L}\p{N}]?\p{L}+       可选 1 个非字母数字前缀（含空格）+ 字母串
    //   ③ \p{N}{1,3} | \p{N}             数字（llama3 最多 3 位一组，qwen2 单个）
    //   ④ ?[^\s\p{L}\p{N}]+[\r\n]*       可选前导空格 + 标点符号串 + 尾随换行
    //   ⑤ \s*[\r\n]+                     换行（带前导空白）
    //   ⑥ \s+(?!\S) | \s+                空白：先尝试"非最后一段"的空白，兜底任意空白
    size_t i = 0;
    while (i < n) {
        size_t adv = 0; // 命中的字符数

        // ① 缩略词
        if (cs[i].cp == '\'') {
            adv = match_contraction(cs, i);
        }

        // ② 前缀 + 字母串（前缀不能是 \r \n，也不能是字母/数字）
        if (adv == 0) {
            size_t j = i;
            if (cs[j].cp != '\r' && cs[j].cp != '\n' &&
                !cs[j].letter && !cs[j].number) {
                j++; // 吃掉 1 个前缀字符（空格、标点、其他符号）
            }
            if (j < n && cs[j].letter) {
                while (j < n && cs[j].letter) { j++; }
                adv = j - i;
            }
        }

        // ③ 数字
        if (adv == 0 && cs[i].number) {
            size_t max_digits = (pre == PreTokenizer::LLAMA3) ? 3 : 1;
            size_t j = i;
            while (j < n && cs[j].number && j - i < max_digits) { j++; }
            adv = j - i;
        }

        // ④ 可选空格 + 标点串 + 尾随换行
        if (adv == 0) {
            size_t j = i;
            if (cs[j].cp == ' ') { j++; } // 分支里的 " ?"：至多一个空格
            size_t sym_start = j;
            while (j < n && !cs[j].space && !cs[j].letter && !cs[j].number) { j++; }
            if (j > sym_start) {
                while (j < n && (cs[j].cp == '\r' || cs[j].cp == '\n')) { j++; }
                adv = j - i;
            }
        }

        // ⑤ 前导空白 + 换行串
        if (adv == 0) {
            size_t j = i;
            while (j < n && cs[j].space && cs[j].cp != '\r' && cs[j].cp != '\n') { j++; }
            size_t nl = j;
            while (j < n && (cs[j].cp == '\r' || cs[j].cp == '\n')) { j++; }
            if (j > nl) { adv = j - i; }
        }

        // ⑥ 空白：优先"后面还是空白/结尾"的（\s+(?!\S)），兜底 \s+
        // 【陷阱】两个分支在这里都"吃掉整段空白"，差别只体现在贪心回溯上：
        //         "  word" 应切成 [" ", " word"] 而不是 ["  ", "word"]。
        //         实现窍门：若段空白后紧跟非空白，则只保留最后一个空格给分支 ② 当前缀。
        if (adv == 0 && cs[i].space) {
            size_t j = i;
            while (j < n && cs[j].space) { j++; }
            if (j < n && !cs[j].space && j - i >= 2) {
                // 段空白 + 非空白：最后 1 个空格留下，配后面的字母/标点
                adv = j - i - 1;
            } else {
                adv = j - i;
            }
        }

        if (adv == 0) {
            // 理论上不可达（每个字符至少落入分支 ②④⑥ 之一），防御性单字符前进
            adv = 1;
        }
        emit(i, i + adv);
        i += adv;
    }
    return out;
}

} // namespace minillama
