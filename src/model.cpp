// 模型加载与前向实现（单元 3 全量 / 单元 4 KV cache / 单元 5 量化）

#include "model.h"

#include "quant.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace minillama {

// ---------------------------------------------------------------------------
// 基础算子
// ---------------------------------------------------------------------------

void rmsnorm(const float * x, const float * w, float * out, int n, float eps) {
    float ms = 0.0f;
    for (int i = 0; i < n; i++) { ms += x[i] * x[i]; }
    ms /= n;
    const float scale = 1.0f / std::sqrt(ms + eps);
    for (int i = 0; i < n; i++) { out[i] = x[i] * scale * w[i]; }
}

// IEEE 754 half -> float（位操作版，对照 ggml 的 ggml_compute_fp16_to_fp32）
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        // 次正规数：value = mant * 2^-24（作为 float 算出再拼符号位）
        float val = static_cast<float>(mant) * (1.0f / (1u << 24));
        uint32_t bits;
        std::memcpy(&bits, &val, 4);
        f = sign | bits;
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

// GGUF 布局"最快维在前"：W(i,o) 存放在 data[i + o*in_dim]，y[o] = Σ x[i]·W(i,o)。
// 【对照】ggml 的 ggml_mul_mat 的 CPU naive 实现
void matvec_f32(const float * w, const float * x, float * y, int in_dim, int out_dim) {
    // 【性能】naive i-j 循环对 CPU 缓存不友好（按 out_dim 跨步访问 W）。
    // ggml 实际用分块 + SIMD + 多线程；教学版保持直译数学定义。
    for (int o = 0; o < out_dim; o++) {
        const float * row = w + static_cast<size_t>(o) * in_dim;
        float s = 0.0f;
        for (int i = 0; i < in_dim; i++) {
            s += row[i] * x[i];
        }
        y[o] = s;
    }
}

// ---------------------------------------------------------------------------
// 量化矩阵乘向量（单元 5）
// 策略：对每个输出列，把该列的权重块反量化到栈缓冲，再与 x 做点积。
// 【性能】真实版 ggml 融合"反量化+点积"（边反量化边乘累加，省一次写读）；
//         我们分开做，代码意图更清晰，速度对 1B 级模型仍可接受。
// ---------------------------------------------------------------------------

// 反量化一列（k 个元素）到 buf；k 必须是块大小的整数倍（加载期已校验）
static void dequant_column(const TensorBuf & w, uint64_t row, float * buf) {
    auto [blck_elems, blck_bytes] = ggml_block_params(w.type);
    // 第 row 列的字节偏移：每列 = (ne0 / 块元素数) × 块字节数
    const size_t col_bytes = static_cast<size_t>(w.ne0) / blck_elems * blck_bytes;
    const uint8_t * col = w.bytes.data() + row * col_bytes;
    switch (w.type) {
        case GgmlType::F32:
            std::memcpy(buf, col, w.ne0 * 4);
            break;
        case GgmlType::F16:
            for (uint32_t i = 0; i < w.ne0; i++) {
                buf[i] = f16_to_f32(reinterpret_cast<const uint16_t *>(col)[i]);
            }
            break;
        case GgmlType::Q8_0:
            dequant_row_q8_0(col, buf, w.ne0);
            break;
        case GgmlType::Q6_K:
            dequant_row_q6_k(col, buf, w.ne0);
            break;
        case GgmlType::Q5_K:
            dequant_row_q5_k(col, buf, w.ne0);
            break;
        case GgmlType::Q4_K:
            dequant_row_q4_k(col, buf, w.ne0);
            break;
        default:
            throw std::runtime_error(
                std::string("matvec: 未支持的量化类型 ") + ggml_type_name(w.type));
    }
}

void tensor_row_to_f32(const TensorBuf & w, uint64_t row, float * out) {
    if (row >= w.ne1) { throw std::runtime_error("tensor_row_to_f32: 行越界"); }
    dequant_column(w, row, out);
}

void matvec(const TensorBuf & w, const float * x, float * y) {
    // 栈缓冲覆盖常见 ne0；超过则退堆分配（如 6144 维的 ffn_down）
    float sbuf[4096];
    std::vector<float> hbuf;
    float * buf = sbuf;
    if (w.ne0 > sizeof(sbuf) / sizeof(sbuf[0])) {
        hbuf.resize(w.ne0);
        buf = hbuf.data();
    }
    for (uint32_t o = 0; o < w.ne1; o++) {
        dequant_column(w, o, buf);
        float s = 0.0f;
        for (uint32_t i = 0; i < w.ne0; i++) { s += buf[i] * x[i]; }
        y[o] = s;
    }
}

void rope_rotate(float * x, int h, int pos, float theta) {
    // llama.cpp master（88b47a7 及之后）对 LLaMA 系架构采用"相邻配对"旋转
    // （llama-model.cpp: LLM_ARCH_LLAMA → LLAMA_ROPE_TYPE_NORM，注释："operating
    //   on pairs of consecutive head values"，即 GPT-J/GPT-2 风格），而非老教程里
    //   常见的"前半后半配对"（GGML_ROPE_TYPE_NEOX，Falcon/GPT-NeoX 风格）。
    // 两种约定不等价（除非加载期对 wq/wk 做行置换），与 llama.cpp 逐数值对照必须用这一种。
    // 频率：inv_freq[i] = theta^(-2i/h)，i 是"配对序号"（每对占相邻 2 维）。
    const int half = h / 2;
    for (int i = 0; i < half; i++) {
        const float freq = 1.0f / std::pow(theta, 2.0f * i / h);
        const float angle = pos * freq;
        const float c = std::cos(angle), s = std::sin(angle);
        const float x1 = x[2 * i], x2 = x[2 * i + 1];
        x[2 * i]     = x1 * c - x2 * s;
        x[2 * i + 1] = x1 * s + x2 * c;
    }
}

void silu(float * x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void softmax(float * x, int n) {
    float m = -1e30f;
    for (int i = 0; i < n; i++) { m = std::max(m, x[i]); }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - m); sum += x[i]; }
    for (int i = 0; i < n; i++) { x[i] /= sum; }
}

// ---------------------------------------------------------------------------
// 加载
// ---------------------------------------------------------------------------

// 读张量为 FP32：F32 原样拷贝，F16 逐元素转换；其余类型抛异常
static std::vector<float> load_tensor_f32(const GgufFile & gguf, const std::string & name) {
    const TensorInfo * t = gguf.find_tensor(name);
    if (!t) {
        throw std::runtime_error("模型缺少张量: " + name);
    }
    size_t n = gguf.tensor_nbytes(*t);
    std::vector<float> out;
    const void * data = gguf.tensor_data(*t);
    switch (t->type) {
        case GgmlType::F32: {
            out.resize(n / 4);
            std::memcpy(out.data(), data, n);
            break;
        }
        case GgmlType::F16: {
            out.resize(n / 2);
            const uint16_t * src = static_cast<const uint16_t *>(data);
            for (size_t i = 0; i < out.size(); i++) {
                out[i] = f16_to_f32(src[i]);
            }
            break;
        }
        default:
            throw std::runtime_error(
                "张量 " + name + " 是量化类型 " + ggml_type_name(t->type) +
                "，FP32 前向不支持（单元 5 实现量化推理）");
    }
    return out;
}

// 读张量为 TensorBuf：量化类型原样保存字节，计算期按块反量化（单元 5）。
// 校验：ne0（输入维）必须是量化块大小的整数倍，否则反量化会越界。
static TensorBuf load_tensor_buf(const GgufFile & gguf, const std::string & name) {
    const TensorInfo * t = gguf.find_tensor(name);
    if (!t) {
        throw std::runtime_error("模型缺少张量: " + name);
    }
    if (t->dims.size() != 2) {
        throw std::runtime_error("张量 " + name + " 必须是 2 维矩阵");
    }
    TensorBuf buf;
    buf.type = t->type;
    buf.ne0  = static_cast<uint32_t>(t->dims[0]);
    buf.ne1  = static_cast<uint32_t>(t->dims[1]);
    auto [blck_elems, blck_bytes] = ggml_block_params(t->type);
    if (buf.ne0 % static_cast<uint32_t>(blck_elems) != 0) {
        throw std::runtime_error("张量 " + name + " 的输入维 " + std::to_string(buf.ne0) +
                                 " 不是块大小 " + std::to_string(blck_elems) + " 的整数倍");
    }
    size_t nbytes = static_cast<size_t>(buf.ne0) / blck_elems * blck_bytes * buf.ne1;
    buf.bytes.resize(nbytes);
    std::memcpy(buf.bytes.data(), gguf.tensor_data(*t), nbytes);
    return buf;
}

static uint32_t get_u32_def(const GgufFile & gguf, const char * key, uint32_t def) {
    if (const GgufValue * v = gguf.find(key)) {
        return std::get<uint32_t>(v->v);
    }
    return def;
}

Model Model::load(const GgufFile & gguf) {
    Model m;

    // ① 超参数
    // 【对照】llama.cpp 的 LLM_KV_NAMES：超参 key 带架构前缀（llama.*/qwen3.*/...），
    //         不同架构的 key 名相同只是前缀不同。张量命名则统一用 blk.N.*。
    //         Qwen2/3 架构派生自 LLaMA，张量布局完全一致，所以一个实现通吃。
    std::string prefix = "llama.";
    if (const GgufValue * v = gguf.find("general.architecture")) {
        prefix = std::get<std::string>(v->v) + ".";
    }
    auto key = [&](const char * suffix) { return prefix + suffix; };

    m.hp_.n_layer   = get_u32_def(gguf, key("block_count").c_str(), 0);
    m.hp_.n_embd    = get_u32_def(gguf, key("embedding_length").c_str(), 0);
    m.hp_.n_head    = get_u32_def(gguf, key("attention.head_count").c_str(), 0);
    m.hp_.n_head_kv = get_u32_def(gguf, key("attention.head_count_kv").c_str(), m.hp_.n_head);
    m.hp_.n_ff      = get_u32_def(gguf, key("feed_forward_length").c_str(), 0);
    m.hp_.n_ctx     = get_u32_def(gguf, key("context_length").c_str(), 4096);
    if (const GgufValue * v = gguf.find(key("attention.layer_norm_rms_epsilon"))) {
        m.hp_.rms_eps = std::get<float>(v->v);
    }
    if (const GgufValue * v = gguf.find(key("rope.freq_base"))) {
        m.hp_.rope_theta = std::get<float>(v->v);
    }
    if (m.hp_.n_layer == 0 || m.hp_.n_embd == 0 || m.hp_.n_head == 0 || m.hp_.n_ff == 0) {
        throw std::runtime_error("模型超参数不完整（需要 " + prefix + "block_count/embedding_length/"
                                 "attention.head_count/feed_forward_length）");
    }
    if (m.hp_.n_embd % m.hp_.n_head != 0) {
        throw std::runtime_error("embedding_length 必须能被 head_count 整除");
    }

    // ② 词表嵌入表（顺带推出 n_vocab）
    m.token_embd_ = load_tensor_buf(gguf, "token_embd.weight");
    m.hp_.n_vocab = m.token_embd_.ne1;

    // ③ 各层权重（norm 是 FP32 小向量直接展开，矩阵保留原始类型）
    m.layers_.resize(m.hp_.n_layer);
    const uint32_t d = m.hp_.n_embd, kvd = m.hp_.kv_dim(), ff = m.hp_.n_ff;
    for (uint32_t i = 0; i < m.hp_.n_layer; i++) {
        const std::string p = "blk." + std::to_string(i) + ".";
        LayerWeights & L = m.layers_[i];
        L.attn_norm = load_tensor_f32(gguf, p + "attn_norm.weight");
        L.wq = load_tensor_buf(gguf, p + "attn_q.weight");
        L.wk = load_tensor_buf(gguf, p + "attn_k.weight");
        L.wv = load_tensor_buf(gguf, p + "attn_v.weight");
        L.wo = load_tensor_buf(gguf, p + "attn_output.weight");
        L.ffn_norm = load_tensor_f32(gguf, p + "ffn_norm.weight");
        L.w_gate   = load_tensor_buf(gguf, p + "ffn_gate.weight");
        L.w_up     = load_tensor_buf(gguf, p + "ffn_up.weight");
        L.w_down   = load_tensor_buf(gguf, p + "ffn_down.weight");

        // 尺寸校验：宁可加载时报错，不要前向时越界
        auto expect = [&](const TensorBuf & w, uint32_t want0, uint32_t want1, const char * what) {
            if (w.ne0 != want0 || w.ne1 != want1) {
                throw std::runtime_error("张量尺寸不符: " + p + what +
                                         " 期望 {" + std::to_string(want0) + "," +
                                         std::to_string(want1) + "} 实际 {" +
                                         std::to_string(w.ne0) + "," +
                                         std::to_string(w.ne1) + "}");
            }
        };
        expect(L.wq, d, d, "attn_q");
        expect(L.wk, d, kvd, "attn_k");
        expect(L.wv, d, kvd, "attn_v");
        expect(L.wo, d, d, "attn_output");
        expect(L.w_gate, d, ff, "ffn_gate");
        expect(L.w_up,   d, ff, "ffn_up");
        expect(L.w_down, ff, d, "ffn_down");
        if (L.attn_norm.size() != d || L.ffn_norm.size() != d) {
            throw std::runtime_error("norm 张量尺寸不符: " + p);
        }
    }

    // ④ 输出头（缺省则与 embedding 绑定 —— TinyLlama/Llama-3 的 tied embeddings）
    m.output_norm_ = load_tensor_f32(gguf, "output_norm.weight");
    if (gguf.find_tensor("output.weight")) {
        m.output_ = load_tensor_buf(gguf, "output.weight");
        m.tied_output_ = false;
    } else {
        m.output_ = m.token_embd_;  // 拷贝共享（简化：物理上复制一份，语义清晰）
        m.tied_output_ = true;
    }
    if (m.output_.ne0 != m.hp_.n_embd || m.output_.ne1 != m.hp_.n_vocab) {
        throw std::runtime_error("output.weight 尺寸与词表不符");
    }
    return m;
}

// ---------------------------------------------------------------------------
// 前向
//
// 两条入口共享同一份单位置计算（forward_token）:
//   - forward(tokens)：教学/测试用，内部临时 KVCache 逐位置推进
//   - forward_token(token, pos, kv)：生成循环用，缓存跨调用复用
// 等价性由 test_generate 的"增量 vs 全量逐位一致"测试钉死。
// ---------------------------------------------------------------------------

std::vector<float> Model::forward(const std::vector<int> & tokens) const {
    KVCache kv(hp_.n_layer, hp_.kv_dim(), hp_.n_ctx);
    std::vector<float> all;
    for (size_t p = 0; p < tokens.size(); p++) {
        auto row = forward_token(tokens[p], static_cast<int>(p), kv);
        all.insert(all.end(), row.begin(), row.end());
    }
    return all;
}

std::vector<float> Model::forward_token(int token, int pos, KVCache & kv) const {
    const uint32_t d = hp_.n_embd, hd = hp_.head_dim();
    const uint32_t nh = hp_.n_head, nkv = hp_.n_head_kv;
    const uint32_t kvd = hp_.kv_dim();

    if (token < 0 || token >= static_cast<int>(hp_.n_vocab)) {
        throw std::runtime_error("token id 越界");
    }
    if (kv.size() != static_cast<size_t>(pos)) {
        throw std::runtime_error("forward_token: 要求 kv.size() == pos（顺序生成）");
    }

    // 残差流当前值（只有一个位置，直接用一维向量）
    std::vector<float> x(d);
    tensor_row_to_f32(token_embd_, static_cast<uint64_t>(token), x.data());

    std::vector<float> h(d), q(d), k(kvd), v(kvd), o(d);
    std::vector<float> gate(hp_.n_ff), up(hp_.n_ff);
    std::vector<float> scores(pos + 1);
    std::vector<float> att_out(d);

    for (uint32_t l = 0; l < hp_.n_layer; l++) {
        const LayerWeights & L = layers_[l];

        // ---- 注意力子层（前置 RMSNorm + 残差） ----
        rmsnorm(x.data(), L.attn_norm.data(), h.data(), d, hp_.rms_eps);
        matvec(L.wq, h.data(), q.data());
        matvec(L.wk, h.data(), k.data());
        matvec(L.wv, h.data(), v.data());
        // RoPE 按头逐段旋转：每次旋转一个 head_dim 的切片（相邻元素配对），
        // 绝不能跨头配对 —— q 有 nh 段、k 有 nkv 段
        for (uint32_t hq = 0; hq < nh; hq++) {
            rope_rotate(&q[static_cast<size_t>(hq) * hd], hd, pos, hp_.rope_theta);
        }
        for (uint32_t hkv = 0; hkv < nkv; hkv++) {
            rope_rotate(&k[static_cast<size_t>(hkv) * hd], hd, pos, hp_.rope_theta);
        }
        kv.append(l, k.data(), v.data());   // 本层 K/V 入缓存，注意力历史从此读

        // 缩放点积注意力：query 头逐个对缓存里 0..pos 的全部 KV 打分
        // 【概念】GQA：query 头数 > KV 头数，每个 KV 头被 q_per_kv 个 query 头
        //         共享（按需索引，不物理复制；真实版 llama.cpp 用 repeat/expand）。
        const uint32_t q_per_kv = nh / nkv;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        for (uint32_t hq = 0; hq < nh; hq++) {
            const uint32_t hkv = hq / q_per_kv;
            const float * qh = &q[static_cast<size_t>(hq) * hd];
            for (int j = 0; j <= pos; j++) {
                const float * kh = kv.k(l, j) + static_cast<size_t>(hkv) * hd;
                float s = 0.0f;
                for (uint32_t i = 0; i < hd; i++) { s += qh[i] * kh[i]; }
                scores[j] = s * scale;
            }
            softmax(scores.data(), pos + 1);
            float * oh = &att_out[static_cast<size_t>(hq) * hd];
            for (uint32_t i = 0; i < hd; i++) { oh[i] = 0.0f; }
            for (int j = 0; j <= pos; j++) {
                const float * vh = kv.v(l, j) + static_cast<size_t>(hkv) * hd;
                const float w = scores[j];
                for (uint32_t i = 0; i < hd; i++) { oh[i] += w * vh[i]; }
            }
        }
        matvec(L.wo, att_out.data(), o.data());
        for (uint32_t i = 0; i < d; i++) { x[i] += o[i]; }

        // ---- FFN 子层（SwiGLU + 残差） ----
        // 【概念】SwiGLU: down( silu(gate(x)) * up(x) )。
        //   两个矩阵 gate/up 是"双通道"，逐元素相乘后投影回去——
        //   这是 LLaMA 相对原始 Transformer ReLU FFN 的主要结构差异之一。
        rmsnorm(x.data(), L.ffn_norm.data(), h.data(), d, hp_.rms_eps);
        matvec(L.w_gate, h.data(), gate.data());
        matvec(L.w_up,   h.data(), up.data());
        silu(gate.data(), hp_.n_ff);
        for (uint32_t i = 0; i < hp_.n_ff; i++) { gate[i] *= up[i]; }
        matvec(L.w_down, gate.data(), o.data());
        for (uint32_t i = 0; i < d; i++) { x[i] += o[i]; }
    }

    // ---- 输出头：最终 RMSNorm + 词表投影 ----
    std::vector<float> logits(hp_.n_vocab);
    rmsnorm(x.data(), output_norm_.data(), h.data(), d, hp_.rms_eps);
    matvec(output_, h.data(), logits.data());
    return logits;
}

} // namespace minillama
