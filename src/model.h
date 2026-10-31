// LLaMA 模型结构与 FP32 前向计算（单元 3）
//
// 对照真实 llama.cpp:
//   src/llama-model.cpp + src/llama-graph.cpp —— 模型定义与计算图构建
//   ggml 的 vec_dot/softmax/rope 内核
//
// 本最小版的关键简化（每个都是重要的教学决策）:
//   - 不做 GGML 计算图：直接写前向循环，读代码 = 读数学
//   - 单元 3 不用 KV cache：每层物化整个序列 X(n×d)，注意力 O(n²)，
//     正确性最易验证；单元 4 再把它优化成增量式 KV cache
//   - 权重加载时统一转成 FP32（F16 存储的模型在加载期转换）；
//     量化类型（Q4_K 等）在单元 5 支持，这里直接报错
//   - 只支持 LLaMA 架构主干（对应 llama.cpp 的 llama arch，也是 qwen3 的命名系）:
//     RMSNorm -> RoPE + GQA 注意力 -> SwiGLU FFN，残差连接
//
// 明确不做:
//   - 其他架构（Gemma 的 q/k-norm、Qwen2MoE、注意力 sink 等变体）
//   - 批量/图捕获/多线程（ggml 的计算后端体系）

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf.h"
#include "kvcache.h"

namespace minillama {

// 模型超参数（对照 llama.cpp llama_hparams，从 GGUF "llama.*" key 读取）
struct HParams {
    uint32_t n_layer    = 0;    // llama.block_count
    uint32_t n_embd     = 0;    // llama.embedding_length
    uint32_t n_head     = 0;    // llama.attention.head_count
    uint32_t n_head_kv  = 0;    // llama.attention.head_count_kv（GQA 的 KV 头数）
    uint32_t n_ff       = 0;    // llama.feed_forward_length
    uint32_t n_ctx      = 0;    // llama.context_length
    uint32_t n_vocab    = 0;    // 由 token_embd.weight 维度推出
    float    rms_eps    = 1e-5f;// llama.attention.layer_norm_rms_epsilon
    float    rope_theta = 1e4f; // llama.rope.freq_base（默认 10000）

    uint32_t head_dim() const { return n_embd / n_head; }
    uint32_t kv_dim()   const { return n_head_kv * head_dim(); }
};

// ---------------------------------------------------------------------------
// 基础算子（独立成函数，便于单测；对照 ggml 内核的 naive 版）
// ---------------------------------------------------------------------------

// RMSNorm: out = x / sqrt(mean(x²) + eps) * weight
// 【概念】LLaMA 系没有 bias 也没有减去均值的 LayerNorm，只有"均方根缩放"，
//        再乘一个可学习的逐元素 weight。比 LayerNorm 少一次均值统计。
void rmsnorm(const float * x, const float * w, float * out, int n, float eps);

// F16 -> F32（权重加载用）
float f16_to_f32(uint16_t h);

// 矩阵乘向量：y = W x（纯 F32 版，测试与小尺寸 norm 用）
void matvec_f32(const float * w, const float * x, float * y,
                int in_dim, int out_dim);

// ---------------------------------------------------------------------------
// 类型化张量缓冲（单元 5）：量化权重不展开存储，计算时按块反量化
// 【概念】GGML 张量 "最快维在前"：ne0 = 输入维（行长度），ne1 = 输出行数，
//         第 o 个输出向量在 bytes[o*ne0*elem_size ... ]，连续存放。
// ---------------------------------------------------------------------------
struct TensorBuf {
    GgmlType type = GgmlType::F32;
    uint32_t ne0 = 0, ne1 = 0;
    std::vector<uint8_t> bytes;
};

// 分发矩阵乘向量：F32 / F16 / Q8_0 / Q6_K / Q5_K / Q4_K，其余类型抛异常
void matvec(const TensorBuf & w, const float * x, float * y);

// 取向量张量的第 row 个输出向量（FP32），用于 embedding 查行等
void tensor_row_to_f32(const TensorBuf & w, uint64_t row, float * out);

// RoPE（旋转位置编码），half-split（两半对旋）风格：
//   把 x[0:h/2] 与 x[h/2:h] 成对旋转，频率 inv_freq[i] = 1/theta^(2i/h)
// 【对照】llama.cpp llama-graph.cpp build_rope()，现代 ggml 默认 neox 风格
// 【历史】早期 llama.cpp 用交错对 (2i, 2i+1)，后与 HF 对齐改为半切风格
void rope_rotate(float * x, int h, int pos, float theta);

// SiLU（SwiGLU 的激活函数）：silu(x) = x * sigmoid(x)
void silu(float * x, int n);

// softmax（数值稳定版：先减最大值）
void softmax(float * x, int n);

// ---------------------------------------------------------------------------
// 模型
// ---------------------------------------------------------------------------

struct LayerWeights {
    std::vector<float> attn_norm, ffn_norm;  // norm 恒为 F32/F16，加载期转 F32
    TensorBuf wq, wk, wv, wo;                // 注意力投影（可能是量化类型）
    TensorBuf w_gate, w_up, w_down;          // FFN（可能是量化类型）
};

class Model {
public:
    // 加载模型权重。量化张量抛异常（单元 5 支持）。
    static Model load(const GgufFile & gguf);

    const HParams & hparams() const { return hp_; }

    // 全量前向：输入 token 序列，输出每个位置的 logits（n × n_vocab，行主序）。
    // 内部用临时 KVCache 逐位置调用 forward_token，与增量路径共享同一份代码。
    std::vector<float> forward(const std::vector<int> & tokens) const;

    // 增量前向（单元 4 核心）：只计算 position=pos 这一个新 token，
    // 历史位置的 K/V 从 kv 缓存读取；算出的本层 K/V 追加进缓存。
    // 调用前 kv.size() 必须等于 pos（顺序生成）。
    // 返回该位置的 logits（长度 n_vocab）。
    std::vector<float> forward_token(int token, int pos, KVCache & kv) const;

    const TensorBuf & embedding() const { return token_embd_; }
    bool tied_output() const { return tied_output_; }

private:
    HParams hp_;
    TensorBuf token_embd_;             // {n_embd, n_vocab}（可能是量化类型）
    std::vector<LayerWeights> layers_;
    std::vector<float> output_norm_;   // {n_embd}
    TensorBuf output_;                 // {n_embd, n_vocab}（缺省时与 embedding 共享）
    bool tied_output_ = false;
};

} // namespace minillama
