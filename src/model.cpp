// ============================================================================
// ① 定位：模型加载与前向计算的实现 —— 单元 3 全量 / 单元 4 增量 / 单元 5 量化
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/models/llama.cpp:99:llama_model_llama::graph<embed>::graph
//                          —— 上游的计算图；本文件把同一段数学直接写成循环
//    llama.cpp @ 88b47a755  src/llama-model.cpp:1060:llama_model_base::load_hparams
//                          src/llama-model.cpp:1250:llama_model_base::load_tensors
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:553:dequantize_row_q8_0
//                          —— 量化张量的反量化公式，逐行移植同一套
//
// ③ 动机：
//    【简化】用直接循环替代 GGML 计算图：承重判据见
//      docs/constraints/04-对齐约束.md §2.2 第 1 条——图只是「先描述、后执行」的
//      调度层，数学与阶段划分一一对应，砍掉它概念不塌。
//    【对照】量化权重按块反量化而不是加载期展开：展开会让 TinyLlama-Q8_0 的权重
//      大 3.8 倍，也丢掉「量化块」这个上游语义。
//    【陷阱】加载期就把形状校验做完：错误发生在离错误源最近的地方，才读得懂。
//
// ④ demo + 多方样例：
//    🟢 最小：tests/test_model.cpp 的 1 层微型模型 → 加载 + 前向全绿。
//    🟡 典型：models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf → 实测摘要
//       「层数=22 维度=2048 头=32/4 词表=32000」，--n 4 能生成 token。
//    🔴 极限：形状不符的 GGUF → 加载期抛「张量尺寸不符: blk.0.attn_k 期望 {...}」。
// ============================================================================

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

// 【对照】ggml 的 rms_norm 同样是「先扫一遍求 ms，再扫一遍归一化」：ms 是整行的
//        标量依赖，一趟算不完；本实现照此结构，便于逐行对公式。
void rmsnorm(const float * x, const float * w, float * out, int n, float eps) {
    float ms = 0.0f;
    for (int i = 0; i < n; i++) { ms += x[i] * x[i]; }
    ms /= n;
    const float scale = 1.0f / std::sqrt(ms + eps);
    for (int i = 0; i < n; i++) { out[i] = x[i] * scale * w[i]; }
}

// ============================================================================
// ① 定位：F16 位模式 → F32 位模式的手写转换（正规数 / 次正规数 / inf-nan 三分支）
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml.c:448:ggml_fp16_to_fp32（单元素）
//                          ggml/src/ggml.c:468:ggml_fp16_to_fp32_row（整行）
//
// ③ 动机：
//    【陷阱】指数域为 0 是次正规数：它的值是 mant * 2^-24（量级 6e-8），不能用
//      正规数的位拼接公式。若把「浮点算出来的值」的位模式当整数拼进去，尾数会被
//      截断，整条分支退化成恒 0 —— 正规数样例全绿，只有最小的一批权重悄悄变 0
//      （docs/03-forward.md §17.1）。
//    【陷阱】指数域为 31 是 inf / nan：套用正规数公式会把 exp=31 算成 exp=46，
//      得到一个巨大的有限数而不是 inf。
//    【对照】上游走硬件 F16C 指令或 GGML_FP16_TO_FP32 宏；本实现手写位运算，
//      是不依赖特定指令集的刻意简化（能力不减：三个分支全覆盖）。
//
// ④ demo + 多方样例：
//    🟢 正规数：0x3C00 → 1.0；0xC000 → -2.0；0x3800 → 0.5（tests/test_model.cpp）。
//    🟡 次正规数：0x0001 → 5.9604645e-8（= 2^-24），单测单独盯这一条分支。
//    🔴 边界：0x7C00 → +inf（指数域全 1、尾数全 0）；0x7E00 → NaN —— 权重里
//       混进 NaN 会一路污染到 logits。
//    调用方：本文件的 load_tensor_f32 把 F16 张量逐元素转过来。
// ============================================================================
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        // 次正规数：value = mant * 2^-24 —— 必须先算出真实数值再拼符号位；
        // 直接把位模式当整数拼会被截断成 0（docs/03-forward.md §17.1）
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

// GGML 布局"最快维在前"：W(i,o) 存放在 data[i + o*in_dim]，y[o] = Σ x[i]·W(i,o)。
// 【对照】对应 ggml 的 ggml_mul_mat（ggml/src/ggml-cpu/ggml-cpu.c:1254）的 CPU 版
// 【性能】naive i-j 循环对 CPU 缓存不友好（按 out_dim 跨步访问 W）。
//         ggml 实际用分块 + SIMD + 多线程；教学版保持直译数学定义。
void matvec_f32(const float * w, const float * x, float * y, int in_dim, int out_dim) {
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
// 【对照】上游按 src0 的类型选 vec_dot 内核（ggml/src/ggml-cpu/ggml-cpu.c:1254），
//         量化类型自带反量化（ggml/src/ggml-quants.c 的 dequantize_row_q*_K 系列）。
// 【性能】真实版 ggml 融合"反量化+点积"（边反量化边乘累加，省一次写读）；
//         我们分开做，代码意图更清晰，速度对 1B 级模型仍可接受。
// ---------------------------------------------------------------------------

// 反量化一列（ne0 个元素）到 buf；ne0 必须是块大小的整数倍（加载期已校验）。
// 【简化】F16 也走这里逐元素转换，与 f16_to_f32 共用同一套位运算口径，
//         免得出现两条 F16→F32 的实现慢慢分叉。
static void dequant_column(const TensorBuf & w, uint64_t row, float * buf) {
    auto [blck_elems, blck_bytes] = ggml_block_params(w.type);
    // 一列 = ne0 个元素 = (ne0 / 块元素数) 个块；量化块沿 ne0 方向连续，
    // 所以第 row 列的字节偏移就是 row 个列长
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

// 【陷阱】越界检查必须在取字节之前：越界的行号读到的是别的张量的字节，
//         不报错但会污染整条残差流。
void tensor_row_to_f32(const TensorBuf & w, uint64_t row, float * out) {
    if (row >= w.ne1) { throw std::runtime_error("tensor_row_to_f32: 行越界"); }
    dequant_column(w, row, out);
}

// 【性能】栈缓冲覆盖常见 ne0；超过则退堆分配（如 5632 维的 ffn_down）。
//         反量化与点积分成两步，比 ggml 融合内核多一次写读，但数值完全一致。
void matvec(const TensorBuf & w, const float * x, float * y) {
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

// ============================================================================
// ① 定位：把一个 head_dim 切片按「相邻配对 (2i, 2i+1)」旋转 —— 就地改写 x
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-model.cpp:2430:llama_model_rope_type
//                          → 2456 行注释「operating on pairs of consecutive head
//                            values」→ 2494 行 return LLAMA_ROPE_TYPE_NORM
//    llama.cpp @ 88b47a755  ggml/src/ggml-cpu/ops.cpp:6070:rotate_pairs(n_dims, 1, ...)
//                          —— NORMAL 分支 offset = 1，即相邻配对
//
// ③ 动机：
//    【陷阱】相邻配对，不是老教程里的 half-split（前半后半对旋）：两种约定给出不同
//      的旋转矩阵，除非加载期对 wq/wk 做行置换才能互换（docs/03-forward.md §17.3）。
//    【概念】i 是「配对序号」不是元素下标：频率 inv_freq[i] = theta^(-2i/h)，
//      每对占相邻 2 维——h=4 时只有 i=0,1 两对，频率是 theta^0 与 theta^-0.5。
//    【对照】上游 ggml_rope_ext（ggml/src/ggml.c:4289）一次处理整个张量、内部按
//      head 逐段循环；本项目由调用方切段，语义等价。
//
// ④ demo + 多方样例：
//    🟢 pos=0：任意配对约定都是恒等变换（所以 pos=0 的单 token 测试测不出错）。
//    🟡 pos=1, h=4, theta=10000：配对 0 的角 = 1、配对 1 的角 = 10000^-0.5 = 0.01
//       —— tests/test_model.cpp 用 x={1,2,3,4} 逐项对拍这两对。
//    🔴 跨头配对：调用方必须按 head_dim 切段传入；整个 q（nh 段）一次传进来，
//       会把位置信息串到别的头上（docs/03-forward.md §17.2）。
// ============================================================================
void rope_rotate(float * x, int h, int pos, float theta) {
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

// 【简化】写 x / (1 + exp(-x)) 而不是 x * sigmoid(x)：两者数学相同，但少一次
//         函数调用，且 x 很大时 exp(-x) 下溢到非规格化数 → 结果就是 x，不溢出。
void silu(float * x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// 【陷阱】必须先减最大值：float32 的 exp(x) 在 x ≈ 89 就溢出成 inf，inf/inf = NaN。
//         分母是「减完最大值之后」的 exp 之和，分子分母同乘 e^-max，比值不变。
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

// 读张量为 FP32：F32 原样拷贝，F16 逐元素转换；其余类型抛异常。
// 【简化】只在这里做 F16→F32 转换：norm 类小向量必须全展开成 float，而投影
//         矩阵走 load_tensor_buf 保留原始量化形态。
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
// 【陷阱】ne0（输入维）必须是量化块大小的整数倍：量化块的编码以整块为单位，
//         行尾的零头没有对应的块，留到计算期就是越界读——所以在加载期就拦。
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

// 读 u32 超参，key 不存在时用默认值。
// 【陷阱】静默缺省值是错误信息的最大伪装：n_ctx / n_head_kv 的缺省是合法缺省，
//         但 block_count 这类必需 key 若名字写错会安静取 0，靠 load 的完整性
//         检查兜住（docs/03-forward.md §17.4）。
static uint32_t get_u32_def(const GgufFile & gguf, const char * key, uint32_t def) {
    if (const GgufValue * v = gguf.find(key)) {
        return std::get<uint32_t>(v->v);
    }
    return def;
}

Model Model::load(const GgufFile & gguf) {
    Model m;

    // ① 超参数
    // 【对照】llama.cpp 的 LLM_KV_NAMES（src/llama-arch.cpp:149）：超参 key 带架构
    //         前缀（llama.*/qwen3.*/...），不同架构的 key 名相同只是前缀不同。
    //         张量命名则统一用 blk.N.*。
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

        // ========================================================================
        // ① 定位：加载期的形状校验 —— 逐张量比对 ne0/ne1，不符即抛
        //
        // ② 源码锚点：
        //    llama.cpp @ 88b47a755  src/llama-model-loader.cpp:858
        //                          :llama_model_loader::check_tensor_dims
        //                          —— 上游读张量时比对期望维度，不符即报「wrong shape」
        //
        // ③ 动机：
        //    【陷阱】错误发生的层次要尽量靠近错误产生的地方：超参与张量形状同时
        //      在手的时刻只有加载期，推迟到前向等于让错误漂到运行期——那时已经写坏
        //      了一堆中间量，输出只是一片 NaN，没有任何可读的错误信息
        //      （docs/03-forward.md §17.5）。
        //    【概念】错误信息必须带张量名 + 期望 + 实际：只说「尺寸不符」等于没说。
        //    【对照】上游 check_tensor_dims 是加载器的通用设施，本实现用 lambda 内联
        //      在层循环里——两者的时机与信息量一致。
        //
        // ④ demo + 多方样例：
        //    🟢 最小：wq 必须是 {d, d}，wk / wv 必须是 {d, kvd}。
        //    🟡 典型：TinyLlama 的 blk.0.attn_k 期望 {2048, 256}（kvd = 4*64 = 256）。
        //    🔴 极限：架构前缀取错 → block_count 取 0，在超参完整性检查处就报
        //       「模型超参数不完整」，根本走不到这一步（docs/03-forward.md §17.4）。
        // ========================================================================
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

    // ④ 输出头：优先用独立的 output.weight；缺省才与 embedding 绑定（tied embeddings）
    // 【对照】上游同样把 output 标成 TENSOR_NOT_REQUIRED，找不到才回落到 tok_embd
    //         （src/models/llama.cpp:41 与 :45）。
    //         本仓库自带的 TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf 里**有**独立的
    //         output.weight {2048, 32000}，所以走的是不绑定分支。
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

// ============================================================================
// ① 定位：全量前向的外壳 —— 临时 KVCache + 逐位置收集，不是第二条计算路径
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-context.cpp:4072:llama_decode
//                          —— 上游一次喂一批 token，用计算图并行算完
//    llama.cpp @ 88b47a755  src/models/llama.cpp:99:llama_model_llama::graph<embed>::graph
//
// ③ 动机：
//    【概念】增量路径是唯一的计算路径：这里没有任何自己的注意力 / FFN 代码，
//      只是「建一个临时缓存 → 逐位置调 forward_token → 把各行 logits 拼起来」。
//      两条路径共用一份计算，所以「增量 ≡ 全量」在结构上就成立；tests/
//      test_generate.cpp 仍把它钉成一条回归用例。
//    【对照】上游用 llama_batch 一次塞多个 token、靠计算图并行；本实现逐位置
//      串行算，数值路径完全相同。
//    【简化】临时缓存用完即弃：教学 / 测试场景不需要跨调用复用；生产路径用
//      forward_token 配自己的长寿命缓存。
//
// ④ demo + 多方样例：
//    🟢 最小：forward({1}) → 长度 n_vocab 的 logits。
//    🟡 典型：forward({1,5,9}) → 长度 3*n_vocab 的行主序结果，每行只看得见
//       自己及之前的位置（因果性）。
//    🔴 极限：tokens.size() 超过 n_ctx → 抛「KVCache 已满（超过 context length）」。
//    调用方：tests/test_model.cpp 与 tests/test_generate.cpp 用它做对照基准。
// ============================================================================
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
        // ========================================================================
        // ① 定位：RoPE 按头逐段旋转 —— 每次只转一个 head_dim 的切片
        //
        // ② 源码锚点：
        //    llama.cpp @ 88b47a755  src/models/llama.cpp:146:ggml_rope_ext（Q）
        //                          src/models/llama.cpp:152:ggml_rope_ext（K）
        //    llama.cpp @ 88b47a755  ggml/src/ggml-cpu/ops.cpp:5953
        //                          :ggml_compute_forward_rope_flt —— 内核按 attn-heads 维逐头循环
        //
        // ③ 动机：
        //    【陷阱】绝不能跨头配对：RoPE 的作用范围是一个头内部（head_dim 维），
        //      把整个 q（nh 段）当成一整块旋转，会让位置信息串到别的头上。更隐蔽的是
        //      pos=0 时 RoPE 是恒等变换，所以全部单 token 测试都漏检它
        //      （docs/03-forward.md §17.2）。
        //    【概念】q 有 nh 段、k 有 nkv 段，两者的段数不同但段长都是 head_dim——
        //      GQA 下 KV 头少，K 的总维数是 kvd 而不是 d。
        //    【对照】上游 ggml_rope_ext 一次处理整个张量、内部按头循环；本项目由
        //      调用方切段，语义等价。
        //
        // ④ demo + 多方样例：
        //    🟢 最小：pos=0 → 每段都不变（恒等变换）。
        //    🟡 典型：TinyLlama 的 q 切成 32 段、k 切成 4 段，每段 64 维。
        //    🔴 极限：段长写成 d（2048）而不是 head_dim（64）→ 一次跨过 32 个头，
        //       单 token 测试全绿但输出明显不对（docs/03-forward.md §17.2）。
        // ========================================================================
        for (uint32_t hq = 0; hq < nh; hq++) {
            rope_rotate(&q[static_cast<size_t>(hq) * hd], hd, pos, hp_.rope_theta);
        }
        for (uint32_t hkv = 0; hkv < nkv; hkv++) {
            rope_rotate(&k[static_cast<size_t>(hkv) * hd], hd, pos, hp_.rope_theta);
        }
        kv.append(l, k.data(), v.data());   // 本层 K/V 入缓存，注意力历史从此读

        // 缩放点积注意力：query 头逐个对缓存里 0..pos 的全部 KV 打分
        // ========================================================================
        // ① 定位：GQA 的 KV 头按需索引共享 —— hkv = hq / q_per_kv，不物理复制
        //
        // ② 源码锚点：
        //    llama.cpp @ 88b47a755  ggml/src/ggml-cpu/ggml-cpu.c:1185:r2 = ne12 / ne02
        //                          —— 上游 mul_mat 用「ne12/ne02」做索引广播，同样
        //                          不物化复制（:1220 的 i02 = i12 / r2）
        //    llama.cpp @ 88b47a755  src/llama-hparams.cpp:74:llama_hparams::n_gqa
        //
        // ③ 动机：
        //    【概念】GQA：query 头数 > KV 头数，每个 KV 头被 q_per_kv 个 query 头
        //      共享。共享是「按需索引」，K/V 缓存里只存 nkv 份——物化复制会让缓存
        //      按 nh/nkv 倍膨胀，而这部分带宽正是 KV cache 的命门。
        //    【对照】上游用 ggml 的 view 广播实现同一件事，数值上与本实现的整除索引
        //      等价（docs/03-forward.md §16.2）。
        //    【陷阱】整除方向不能反：hq / q_per_kv 才是「query 头 → KV 头」的映射；
        //      写成 q_per_kv / hq 会让大部分 query 头指向 KV 头 0，静默出错。
        //
        // ④ demo + 多方样例：
        //    🟢 最小：nh=2、nkv=1 → q_per_kv=2 → hq=0,1 都映射到 hkv=0。
        //    🟡 典型：TinyLlama 的 nh=32、nkv=4 → q_per_kv=8 → hq 0..7→hkv 0、
        //       8..15→1、16..23→2、24..31→3。
        //    🔴 极限：nkv 不整除 nh（上游要求整除）→ q_per_kv 截断，后面那些
        //       query 头会共享错误的 KV 头。
        // ========================================================================
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
        // 【对照】上游 build_ffn（src/llama-graph.cpp:1564）的 LLM_FFN_SILU 分支
        //   走 ggml_swiglu_split（把 silu(gate)*up 融成一个算子），数学等价。
        rmsnorm(x.data(), L.ffn_norm.data(), h.data(), d, hp_.rms_eps);
        matvec(L.w_gate, h.data(), gate.data());
        matvec(L.w_up,   h.data(), up.data());
        silu(gate.data(), hp_.n_ff);
        for (uint32_t i = 0; i < hp_.n_ff; i++) { gate[i] *= up[i]; }
        matvec(L.w_down, gate.data(), o.data());
        for (uint32_t i = 0; i < d; i++) { x[i] += o[i]; }
    }

    // ---- 输出头：最终 RMSNorm + 词表投影 ----
    // 【对照】上游同一位置是 build_norm(..., LLM_NORM_RMS, -1) 紧跟
    //         build_lora_mm(model.output, ...)（src/models/llama.cpp:231 与 :240）。
    std::vector<float> logits(hp_.n_vocab);
    rmsnorm(x.data(), output_norm_.data(), h.data(), d, hp_.rms_eps);
    matvec(output_, h.data(), logits.data());
    return logits;
}

} // namespace minillama
