// ============================================================================
// ① 定位：分词器的基础设施 —— UTF-8 编解码、Unicode 类别判定、字节映射、预分词扫描
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/unicode.cpp:16  unicode_len_utf8
//    llama.cpp @ 88b47a755  src/unicode.cpp:148  unicode_byte_to_utf8_map
//    llama.cpp @ 88b47a755  src/unicode.cpp:333  unicode_regex_split_custom_llama3
//    llama.cpp @ 88b47a755  src/unicode.cpp:474  unicode_regex_split_custom_qwen2
//
// ③ 动机：
//    【概念】这四件事是「文本 → token」这条链的地基：先按字节切开并分类字符，
//    再按预分词规则切词，词内才轮到 BPE 合并。顺序错了词边界就跟着错。
//    【简化】只实现 \p{L}（字母）与 \p{N}（数字）两类，外加 ASCII 空白：上游
//    llama3 / qwen2 两个扫描器（src/unicode.cpp:333 / :474）查的也正好是
//    is_letter / is_number / is_whitespace 三个标志，其余类别不在 Llama 主链上。
//    【对照】上游的 Unicode 数据是一张 7034 行的全类别表（unicode-data.cpp），
//    本仓库由 tools/gen_unicode_tables.py 生成两张区间表（字母 622 段、数字 133 段）。
//
// ④ demo + 多方样例：
//    🟢 utf8_decode("A") → 码点 65，长 1 字节
//    🟡 utf8_decode("中") → 码点 U+4E2D，长 3 字节；pre_tokenize("Hello world") → 2 片
//    🔴 非法字节序列 → 返回 0xFFFD（替换符）并前进 1 字节，不中断整段解析
//    调用方：src/tokenizer.cpp 的分词核心；lines/text/tokenize_line.cpp 的演示线
// ============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace minillama {

// ============================================================================
// ① 定位：UTF-8 单字符的解码 / 编码 / 长度 —— 三个互逆的小工具
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/unicode.cpp:16  unicode_len_utf8
//    llama.cpp @ 88b47a755  src/unicode.cpp:30  unicode_cpt_from_utf8
//    llama.cpp @ 88b47a755  src/unicode.cpp:1088  unicode_cpt_to_utf8
//
// ③ 动机：
//    【概念】分词全程按「字符」而不是「字节」思考（切词按字符、合并按字符），
//    所以必须有一层可靠的 UTF-8 ⇄ 码点转换。
//    【陷阱】非法序列不能抛异常：词表可能覆盖不到的字节必须还能往前走，
//    否则一段坏字节会让整段文本编不出来。这里统一回退成 U+FFFD（替换符）
//    并前进 1 字节，保证「一定能走完」。
//    【对照】上游 unicode_cpt_from_utf8 对非法序列抛 std::invalid_argument；
//    本实现选择容错前进，是为了让字节兜底路径（docs/02-tokenizer.md §7.4）有东西可兜。
//
// ④ demo + 多方样例：
//    🟢 utf8_len('A') → 1；utf8_decode("A") → 65
//    🟡 utf8_decode("中") → 0x4E2D，长 3；utf8_encode(0x4E2D, out) 写出 3 字节
//    🔴 utf8_decode("\xFF") → 0xFFFD，长 1（非法前导字节，容错前进）
//    调用方：src/tokenizer.cpp / :261 / :334；src/unicode.cpp 的 decode 快路径
// ============================================================================
uint32_t utf8_decode(const char * s, size_t avail, size_t * len);

// 码点编码为 UTF-8，追加到 out（byte_encode 的反向操作用）
void utf8_encode(uint32_t cp, std::string & out);

// 字符长度（字节数），等价 llama.cpp 的 unicode_len_utf8
size_t utf8_len(char first_byte);

// ============================================================================
// ① 定位：Unicode 类别判定 —— 扫描器只问三件事：是不是字母、是不是数字、是不是空白
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/unicode.h:8  struct unicode_cpt_flags
//    llama.cpp @ 88b47a755  src/unicode.cpp:116  unicode_cpt_flags_array
//
// ③ 动机：
//    【概念】上游把 Unicode 类别压成 16 位标志位（is_letter / is_number / ...），
//    扫描器按位查；本实现只有两张区间表 + 二分，能力不减（够这两类用）。
//    【性能】字母 622 段、数字 133 段，二分约 10 次比较；每个字符只查一次表，
//    相对 BPE 合并循环可以忽略。
//    【对照】上游的空白判定覆盖全部 Unicode 空白字符，本实现只认 ASCII 的 6 个
//    （空格 / 制表 / 换行 / 回车 / 换页 / 垂直制表）—— llama3 / qwen2 扫描器里
//    的 \s 用的就是同一套 ASCII 集合。
//
// ④ demo + 多方样例：
//    🟢 is_unicode_letter(0x4E2D) → true（U+4E2D 中，落在 {0x4E00, 0x9FFC} 段里）
//    🟡 is_unicode_number(0x35) → true（ASCII '5'，落在 {0x0030, 0x0039} 段里）
//    🔴 is_ascii_space('\xC2') → false（非 ASCII 一律不算空白）
//    调用方：src/unicode.cpp 的 decode() 与 pre_tokenize() 的分支 ②③④⑥
// ============================================================================
bool is_unicode_letter(uint32_t cp);
bool is_unicode_number(uint32_t cp);

// ASCII 空白（正则 \s 在这些预分词器里等价于 ASCII 空白集）
bool is_ascii_space(char c);

// ============================================================================
// ① 定位：GPT-2 的 bytes_to_unicode —— 0~255 原始字节 ↔ 256 个可见 Unicode 码点
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/unicode.cpp:148  unicode_byte_to_utf8_map
//    llama.cpp @ 88b47a755  src/unicode.cpp:1162  unicode_byte_to_utf8
//
// ③ 动机：
//    【概念】BPE 词表只含「文本 token」：任意二进制输入都得能变成一串可见字符，
//    否则词表装不下。所以 256 个字节被一一映射到 256 个**可打印**码点。
//    【陷阱】映射不是「所有字节 +256」：可打印字节（0x21~0x7E、0xA1~0xAC、
//    0xAE~0xFF，共 188 个）映射到自身，其余 68 个才顺次映射到 256..323。
//    【对照】上游用 unordered_map 预先算好，本实现按公式现算：字节 b 的码点是
//    256 +「小于 b 的非身份字节个数」；反查时倒着数同一个序列就是。
//
// ④ demo + 多方样例：
//    🟢 byte_to_unicode_cp(0x41) → 0x41（'A' 是可打印字节，映射到自身）
//    🟡 byte_to_unicode_cp(0x20) → 288 = U+0120 'Ġ'（空格）；unicode_cp_to_byte(288) → 32
//    🔴 unicode_cp_to_byte(0x4E2D) → -1（不是映射表里的码点）
//    调用方：src/tokenizer.cpp / :296 编码时映射、:335 解码时反查
// ============================================================================
uint32_t byte_to_unicode_cp(uint8_t b);   // 字节 -> 映射码点
int      unicode_cp_to_byte(uint32_t cp); // 映射码点 -> 字节（不在表中返回 -1）

// ============================================================================
// ① 定位：预分词模式 —— 同一套 6 分支扫描器，只有「数字怎么分组」这一处不同
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-vocab.h:10  enum llama_vocab_pre_type
//    llama.cpp @ 88b47a755  src/unicode.cpp:1050  unicode_regex_split_custom
//
// ③ 动机：
//    【概念】模式只决定一件事：连着的一串数字切成几片。llama3 按 1~3 位一组，
//    qwen2 按单个数字一组；其余 5 个分支两条路完全一样。
//    【对照】上游是「一串正则表达式 → 文本比对 → 分派到对应扫描器」
//    （src/unicode.cpp:1050），模式信息藏在表达式文本里；本仓库把它显式成一个
//    枚举，调用方直接选。
//
// ④ demo + 多方样例：
//    🟢 "1" → 两种模式都是 ["1"]
//    🟡 "12345" → LLAMA3 得 ["123", "45"]；QWEN2 得 ["1","2","3","4","5"]
//    🔴 模式选错 → 词边界错 → 合并出的 token 序列与上游分叉，而且不报错
//    调用方：src/tokenizer.cpp 把 pre_ 传进 pre_tokenize
// ============================================================================
enum class PreTokenizer { LLAMA3, QWEN2 };

// ============================================================================
// ① 定位：把整段文本切成「词」—— BPE 合并只在词内发生，词边界由这里定
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/unicode.cpp:1050  unicode_regex_split_custom
//    llama.cpp @ 88b47a755  src/unicode.cpp:333  unicode_regex_split_custom_llama3
//    llama.cpp @ 88b47a755  src/unicode.cpp:474  unicode_regex_split_custom_qwen2
//
// ③ 动机：
//    【对照】上游用通用正则引擎跑 tokenizer.json 里的模式串；但上游对 llama3 /
//    qwen2 这两条模式**自己也不走**通用路径，而是手写扫描器
//    （unicode_regex_split_custom 先特判再分派）。本实现对齐的是上游的手写
//    扫描器路线：6 个分支按声明顺序试，先命中先走 —— 分支顺序就是正则交替的优先级。
//    【简化】只支持 llama3 与 qwen2 两种模式：上游另有 gpt2 / qwen35 / kimi-k2 /
//    afmoe 四套扫描器，都不在 Llama 主链上（承重结论见 docs/02-tokenizer.md §16.3）。
//    【概念】SPM 词表不经过本函数 —— 整段文本直接进合并核心（src/tokenizer.cpp）。
//
// ④ demo + 多方样例：
//    🟢 "Hello world" → ["Hello", " world"]（空格归后一词）
//    🟡 "  word" → [" ", " word"]；"12345" 在 LLAMA3 下 → ["123", "45"]
//    🔴 "I'll" → ["I", "'ll"]（收缩语整块成一片，大小写不敏感）
//    调用方：src/tokenizer.cpp 在 BPE 路径上逐词送进 merge_tokenize
// ============================================================================
std::vector<std::string> pre_tokenize(const std::string & text, PreTokenizer pre);

} // namespace minillama
