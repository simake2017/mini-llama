// UTF-8 编解码 + Unicode 类别判定 + 预分词正则扫描器（单元 2 的基础设施）
//
// 对照真实 llama.cpp:
//   src/unicode.cpp      —— UTF-8/码点转换、类别归一化
//   src/unicode-data.cpp —— 类别表（本仓库由 tools/gen_unicode_tables.py 生成）
//
// 本最小版简化点:
//   - 只实现分词器需要的 \p{L}（字母）/\p{N}（数字）/\s（ASCII 空白）
//   - 不做 Unicode 归一化（NFKC）、大小写折叠 —— llama.cpp 也用不到（正则里是显式 [sS]）

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace minillama {

// UTF-8 单字符解码。返回码点；非法序列返回 0xFFFD 并置 *len = 1（容错前进）。
uint32_t utf8_decode(const char * s, size_t avail, size_t * len);

// 码点编码为 UTF-8，追加到 out（byte_encode 的反向操作用）
void utf8_encode(uint32_t cp, std::string & out);

// 字符长度（字节数），等价 llama.cpp 的 unicode_len_utf8
size_t utf8_len(char first_byte);

// Unicode 类别判定（基于生成表二分查找）
bool is_unicode_letter(uint32_t cp);
bool is_unicode_number(uint32_t cp);

// ASCII 空白（正则 \s 在这些预分词器里等价于 ASCII 空白集）
bool is_ascii_space(char c);

// ---------------------------------------------------------------------------
// GPT-2 bytes_to_unicode 编解码（OpenAI BPE 的配套技巧）
// 【概念】0-255 原始字节 ↔ 256 个可见 Unicode 码点：可打印字节映射到自身，
//         其余 68 个（控制字符/空白）依次映射到 256..323。
//         BPE 词表因此可以只含"文本" token，任意二进制输入都能编码。
// ---------------------------------------------------------------------------
uint32_t byte_to_unicode_cp(uint8_t b);   // 字节 -> 映射码点
int      unicode_cp_to_byte(uint32_t cp); // 映射码点 -> 字节（不在表中返回 -1）

// ---------------------------------------------------------------------------
// 预分词：把原始文本切成"词"（word），词内再做 BPE 合并。
//
// 【对照】真实版用通用正则引擎（unicode_regex_split）跑 tokenizer.json 里的
//         模式；我们发现这些模式都是固定的 6 个分支，于是手写成确定性扫描器，
//         行为等价、零依赖、还能在注释里逐分支讲语义。
//
// 支持两种模式（llama.cpp 的 LLAMA_VOCAB_PRE_TYPE）:
//   Llama3: \p{N} 按 1~3 位数字分组
//   Qwen2 : \p{N} 按单个数字分组
// ---------------------------------------------------------------------------
enum class PreTokenizer { LLAMA3, QWEN2 };

// 按预分词规则切分文本。返回的每个 piece 都是原文的连续子串。
std::vector<std::string> pre_tokenize(const std::string & text, PreTokenizer pre);

} // namespace minillama
