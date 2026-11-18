// ============================================================================
// ① 定位：分词器 —— 从 GGUF 元数据里的词表，把文本编成 token 序列、再解回文本
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:1921:llama_vocab::impl::load（词表加载）
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:117:llm_tokenizer_spm_session::tokenize
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:603:llm_tokenizer_bpe_session::tokenize
//    llama.cpp @ 88b47a755  include/llama.h:72:enum llama_vocab_type（SPM / BPE 是其子集）
//
// ③ 动机：
//    【概念】词表是「文本 ↔ token id」的对照表，GGUF 把它放在元数据里而不是独立文件；
//       编解码是同一张表的两面，所以本类同时持有表与算法。
//    【简化】上游 6 种词表类型（include/llama.h:72）只取 SPM 与 BPE 两支，另 4 支
//       （UGM / WPM / RWKV / PLAMO2）不在 Llama 主链上；承重结论见
//       docs/constraints/04-对齐约束.md §2.1 组 B。
//    【对照】上游把「词表」与「编码会话」拆两层（llama_vocab::impl +
//       llm_tokenizer_*_session），本实现合成一层：会话的全部状态就是「这张表 +
//       一个 score」，合层后对照时少一层间接。
//
// ④ demo + 多方样例：
//    🟢 最小：GGUF → Tokenizer::load → encode("Hello world") → 5 个 token
//    🟡 典型：中英混合 + 标点 → encode("Hello, world! 你好 123") 得 14 个 token
//    🔴 极限：词表覆盖不到的字符走 BYTE 兜底，兜不住的字节被丢弃（SPM 的多字节
//       字符实测被整体丢弃，详见 encode 的四段式）
//    调用方：src/main.cpp 构造；编解码的实现在 src/tokenizer.cpp
// ============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"
#include "unicode.h"

namespace minillama {

// ============================================================================
// ① 定位：token 的类型标签 —— 决定解码时这个 token 渲染成文本、还原成字节，还是跳过
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  include/llama.h:91:enum llama_token_type（7 个取值逐个一致）
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:2449:llama_vocab::impl::load 的 type→attr 映射
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:2968:llama_vocab::impl::load 的 per-token attr 补全
//
// ③ 动机：
//    【概念】GGUF 的 token_type 数组按 token id 同序给出每个 token 的类型；解码时
//       BYTE 要还原成原始字节、CONTROL 要跳过、NORMAL 要原样渲染 —— 类型是
//       「渲染策略」的唯一依据，也是词表文件与解码器之间的契约。
//    【对照】上游另有 llama_token_attr 位标志集合（UNKNOWN / CONTROL / UNUSED 可叠加），
//       本实现取 7 值枚举 —— GGUF 里存的就是单值枚举（上游 :2449 逐值映射成单个位），
//       上游多出的位是运行时按模型名补的（:2968 起，GGUF 文件里没有这项信息）。
//
// ④ demo + 多方样例：
//    🟢 NORMAL = 1：解码时原样渲染文本
//    🟡 BYTE = 6：解码时还原成 1 个原始字节（src/tokenizer.cpp 的 BYTE 分支）
//    🔴 CONTROL = 3：解码时整条跳过；而 TinyLlama 的 <s> 标的是 4（USER_DEFINED），
//       于是它会被渲染出来 —— 与上游行为一致（上游跳过掩码只有 UNKNOWN | CONTROL）
//    调用方：src/tokenizer.cpp 的 decode 按 type 分派
// ============================================================================
enum class TokenType : int32_t {
    UNDEFINED    = 0,
    NORMAL       = 1,
    UNKNOWN      = 2,
    CONTROL      = 3,
    USER_DEFINED = 4,
    UNUSED       = 5,
    BYTE         = 6,
};

// 词表类型（对照 include/llama.h:72 的 enum llama_vocab_type，本实现取 SPM / BPE 两支）
enum class VocabType { SPM, BPE };

// ============================================================================
// ① 定位：Tokenizer —— 一份 GGUF 词表在运行期的视图，既是「表」也是「编解码器」
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:1779:llama_vocab::impl（词表本体）
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:114:llm_tokenizer_spm_session（编码会话）
//    llama.cpp @ 88b47a755  src/llama-vocab.cpp:561:llm_tokenizer_bpe_session
//
// ③ 动机：
//    【对照】上游拆两层：llama_vocab::impl 持有表与索引，llm_tokenizer_*_session 持有
//       一次编码的临时状态。本实现合成一层 —— 会话的全部状态就是「这张表 + 一个
//       score」，合层后对照时少一层间接（上游自己在 llama-vocab.cpp:246 标了 TODO
//       说两份 session 该合并复用）。
//    【简化】词表类型只取 SPM / BPE 两支（理由见文件头【简化】段）；special token
//       片段提取（tokenizer_st_partition）不做，承重结论见
//       docs/constraints/04-对齐约束.md §2.1 组 B。
//
// ④ demo + 多方样例：
//    🟢 类型级用法：GgufFile g = GgufFile::load(path); Tokenizer tok = Tokenizer::load(g);
//    🟡 典型：tok.encode("Hello world") 拿 token 序列，tok.decode(ids) 还原文本
//    🔴 词表类型不是 llama / gpt2 → Tokenizer::load 抛「不支持的词表类型 ...」
//    调用方：src/main.cpp 是全仓唯一的构造点；lines/ 下 5 条线各自构造一份
// ============================================================================
class Tokenizer {
public:
    // =========================================================================
    // ① 定位：从 GGUF 元数据构建词表 —— 读三张平行数组、建正反向索引、判出词表开关
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  src/llama-vocab.cpp:1921:llama_vocab::impl::load
    //    llama.cpp @ 88b47a755  src/llama-vocab.cpp:3882:llama_vocab::text_to_token
    //
    // ③ 动机：
    //    【概念】词表住在 GGUF 的元数据里而不是独立文件：tokens / scores / token_type
    //       三张数组按 token id 同序，第 i 条分别给出文本、合并分数、类型。
    //    【陷阱】scores 与 token_type 数组上游会校验长度并抛错（llama-vocab.cpp:2415 /
    //       :2425），本实现直接按下标读 —— 长度不一致时是越界读，属已知缺陷
    //       （docs/02-tokenizer.md §16.2）。
    //    【对照】上游读 add_bos_token / add_eos_token，本实现读 add_bos / add_eos ——
    //       GGUF 里普遍写的是上游那个名字，所以词表级的覆盖实际读不到，靠「SPM 与
    //       llama-bpe 默认加 BOS」的族默认值兜住（待拍板，docs/02-tokenizer.md §15）。
    //
    // ④ demo + 多方样例：
    //    🟢 合成词表（tests/test_tokenizer.cpp）：model=llama → SPM、add_bos=1
    //    🟡 真实 TinyLlama：model=llama、32000 个 token、bos=1、eos=2、add_bos=1
    //       （lines/load/vocab_load_line.cpp 第 3 / 6 跳）
    //    🔴 model=rwkv → 抛「不支持的词表类型 tokenizer.ggml.model: rwkv」
    //    调用方：src/main.cpp；lines/load/vocab_load_line.cpp:91
    // =========================================================================
    static Tokenizer load(const GgufFile & gguf);

    // =========================================================================
    // ① 定位：文本 → token id 序列 —— 改写文本、切分、按 score 合并、拼 BOS / EOS
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  src/llama-vocab.cpp:117:llm_tokenizer_spm_session::tokenize
    //    llama.cpp @ 88b47a755  src/llama-vocab.cpp:603:llm_tokenizer_bpe_session::tokenize
    //    llama.cpp @ 88b47a755  src/llama-vocab.cpp:570:llm_tokenizer_bpe_session::append_bos
    //    llama.cpp @ 88b47a755  src/llama-vocab.cpp:579:llm_tokenizer_bpe_session::append_eos
    //
    // ③ 动机：
    //    【陷阱】**SPM 句首补空格与 add_special 无关**：add_special 只决定拼不拼
    //       BOS / EOS，补空格由词表开关 add_space_prefix_ 单独决定。上游判据是
    //       add_space_prefix && is_prev_special（llama-vocab.cpp:3348），而
    //       is_prev_special 的初值是 true（:3335），与 add_special 无关 ——
    //       两个都带 "add" 的开关语义正交，把一个当另一个的子条件就会多出空格。
    //    【历史】SPM 用 ▁(U+2581) 标词边界，句首不补空格，首词与后续词就分不开 ——
    //       这条 sentencepiece 规范 llama.cpp 至今保留。
    //
    // ④ demo + 多方样例：
    //    🟢 英文短词：真实 TinyLlama 的 encode("Hello world") → 5 个 token
    //       [1, 15043, 8879, 2096, 29881]
    //    🟡 中英混合 + 标点：encode("Hello, world! 你好 123") → 14 个 token
    //    🔴 大量未登录字符：SPM 路径按 UTF-8 字符逐个拆、只有单字节字符才落 BYTE token；
    //       实测 encode("😀") 只有 2 个 token（[1, 29871]），emoji 本身没进序列（上游 resegment 是逐字节兜底，llama-vocab.cpp:176）
    //    调用方：src/main.cpp 的生成循环；lines/text/tokenize_line.cpp:80
    // =========================================================================
    std::vector<int> encode(const std::string & text, bool add_special = true) const;

    // =========================================================================
    // ① 定位：token id 序列 → 文本 —— 按类型分派渲染，再把 ▁ 换回空格
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  src/llama-vocab.cpp:4092:llama_vocab::token_to_piece
    //    llama.cpp @ 88b47a755  src/llama-vocab.cpp:3519:llama_vocab::impl::token_to_piece
    //
    // ③ 动机：
    //    【对照】上游跳过掩码只有 UNKNOWN | CONTROL（llama-vocab.cpp:3521 的
    //       attr_special），本实现多跳过 UNUSED / UNDEFINED 两种上游不跳的类型 ——
    //       真实 GGUF 里 <s> 常被标成 USER_DEFINED(4)，上游也照渲染，两边一致。
    //    【陷阱】▁ ↔ 空格的互转是「编码空间 ↔ 文本空间」的边界：BPE 词表走
    //       bytes_to_unicode 逆映射（空格 ↔ U+0120 Ġ），SPM 词表走 ▁ ↔ ' '，
    //       两边不能混用同一套转义。
    //
    // ④ demo + 多方样例：
    //    🟢 最小：decode({8}) → " ab"（合成 SPM 词表，"▁ab" 还原成空格 + ab）
    //    🟡 典型：真实 TinyLlama 的 decode(encode("Hello world")) → "<s> Hello world"
    //    🔴 越界 id 直接跳过、不抛异常 → decode({999999999}) 返回空串
    //       （lines/text/detokenize_line.cpp:106 断言了这一点）
    //    调用方：src/main.cpp / :103 的回显；lines/text/detokenize_line.cpp:120
    // =========================================================================
    std::string decode(const std::vector<int> & ids) const;

    // 词表规模与特殊 token 的只读视图（O(1) 取值，不做任何计算）
    int  bos_id() const { return bos_id_; }
    int  eos_id() const { return eos_id_; }
    bool add_bos() const { return add_bos_; }
    size_t n_tokens() const { return tokens_.size(); }

    const std::string & token_text(int id) const { return tokens_[id].text; }
    bool is_eos(int id) const { return id == eos_id_; }

private:
    // token 三元组：文本 / 合并分数 / 类型。分数是 BPE 合并优先级的唯一来源。
    struct TokenData {
        std::string text;
        float       score;
        TokenType   type;
    };

    // 反向索引是合并循环的热路径：每试一次相邻合并就查一次表。
    int find_by_text(const std::string & text) const;  // 找不到返回 -1
    // 字节兜底的三级查找链：显式 BYTE 表 → bytes_to_unicode 映射字符 → 单字节原文。
    int byte_token(int b) const;

    // SPM / BPE 共用的合并核心：对一段文本做「按 score 优先的相邻合并」。
    // 【概念】这就是 BPE 训练时学到的合并规则在推理侧的应用：词表里每个多字符
    //         token 的 score 就是它参与合并的优先级，所以「查表」和「定优先级」
    //         是同一张表的两个用途。
    void merge_tokenize(const std::string & text, std::vector<int> & out) const;

    std::vector<TokenData>               tokens_;
    std::unordered_map<std::string, int> text_to_id_;
    int                                  byte_to_id_[256] {};   // 原始字节 -> BYTE token id
    std::vector<int>                     id_to_byte_;           // BYTE token id -> 原始字节
    bool                                 has_byte_map_ = false; // 词表里有没有显式 BYTE token

    VocabType    type_               = VocabType::SPM;
    PreTokenizer pre_                = PreTokenizer::LLAMA3;
    int          bos_id_             = -1;
    int          eos_id_             = -1;
    bool         add_bos_            = false;
    bool         add_eos_            = false;
    bool         escape_whitespaces_ = true;  // ' ' ↔ ▁(U+2581) 互转
    bool         add_space_prefix_   = false; // SPM 传统：句首补空格
    bool         byte_encode_        = false; // GPT-2 bytes_to_unicode（qwen2/gpt2）
};

} // namespace minillama
