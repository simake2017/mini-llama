// 分词器（单元 2）：从 GGUF 元数据加载词表，实现 SPM / BPE 两种编码路径
//
// 对照真实 llama.cpp:
//   src/llama-vocab.cpp —— llama_vocab（加载）+ llm_tokenizer_spm/bpe_session（编码）
//
// 本最小版支持:
//   - SPM  (tokenizer.ggml.model = "llama"，TinyLlama/Llama-2 系):
//       整段文本按 UTF-8 字符切 symbol，按 score 贪心地两两合并，
//       无法识别的残段逐字节落到 BYTE token（llama.cpp 的 llm_tokenizer_spm_session）
//   - BPE  (model = "gpt2"，Llama-3/Qwen2 系):
//       先用预分词正则切词，词内做同样的 score 合并
//       （llm_tokenizer_bpe_session；我们不用 GPT-2 bytes_to_unicode 编码，
//        因为 llama3/qwen2 的词表都是原始 UTF-8 字节型，byte_encode=false）
//
// 明确不做:
//   - UGM / WPM / RWKV 等其他词表类型（llama-vocab.cpp 还有 4 套实现）
//   - special token 片段提取（llama.cpp 的 tokenizer_st_partition，
//     处理 "<|end_of_text|>" 这类文本内嵌特殊串的场景）
//   - 通用正则引擎（固定模式手写成扫描器，见 unicode.cpp）

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"
#include "unicode.h"

namespace minillama {

// token 类型（对照 llama.h enum llama_token_type）
enum class TokenType : int32_t {
    UNDEFINED    = 0,
    NORMAL       = 1,
    UNKNOWN      = 2,
    CONTROL      = 3,
    USER_DEFINED = 4,
    UNUSED       = 5,
    BYTE         = 6,
};

enum class VocabType { SPM, BPE };

class Tokenizer {
public:
    // 从 GGUF 元数据构建词表。缺少必要 key 抛异常。
    static Tokenizer load(const GgufFile & gguf);

    // 文本 -> token id 序列。add_special=true 时按词表约定加 BOS。
    std::vector<int> encode(const std::string & text, bool add_special = true) const;

    // token id 序列 -> 文本。跳过特殊/控制 token；BYTE token 还原为原始字节。
    std::string decode(const std::vector<int> & ids) const;

    int  bos_id() const { return bos_id_; }
    int  eos_id() const { return eos_id_; }
    bool add_bos() const { return add_bos_; }
    size_t n_tokens() const { return tokens_.size(); }

    const std::string & token_text(int id) const { return tokens_[id].text; }
    bool is_eos(int id) const { return id == eos_id_; }

private:
    struct TokenData {
        std::string text;
        float       score;
        TokenType   type;
    };

    int find_by_text(const std::string & text) const;  // 找不到返回 -1
    int byte_token(int b) const;                       // 字节兜底查找链

    // SPM/BPE 共用的合并核心：对一段文本做"按 score 优先的相邻合并"
    // 【概念】这就是 BPE 训练时学到的合并规则的推理侧应用：
    //         词表里每个多字符 token 的 score 就是它参与合并的优先级。
    void merge_tokenize(const std::string & text, std::vector<int> & out) const;

    std::vector<TokenData>               tokens_;
    std::unordered_map<std::string, int> text_to_id_;
    int                                  byte_to_id_[256] {};   // 原始字节 -> BYTE token id
    std::vector<int>                     id_to_byte_;           // BYTE token id -> 原始字节
    bool                                 has_byte_map_ = false;

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
