// ============================================================================
// ① 定位：LLaMA 主干的结构定义与前向计算接口 —— 单元 3 定形状、4 加增量、5 加量化
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/models/llama.cpp:llama_model_llama::graph<embed>::graph
//                          —— 上游同一条链：norm → QKV+rope → attn → SwiGLU → norm → lm_head
//    llama.cpp @ 88b47a755  src/llama-hparams.cpp:llama_hparams::n_head / n_head_kv / n_gqa
//                          —— HParams 字段职责与上游一一对应
//
// ③ 动机：
//    【简化】不做 GGML 计算图：直接写前向循环，读代码 = 读数学。承重判据见
//      docs/constraints/04-对齐约束.md §2.2 第 1 条——砍掉图不改变「一次前向 = 一个
//      token 进、一份 logits 出」的语义，阶段划分与公式一一对应，三测全过。
//    【对照】量化张量不展开成 FP32：TensorBuf 存原始字节、计算期按块反量化，
//      否则 1B 级模型的权重会膨胀数倍。
//    【简化】单元 3 不用 KV cache：注意力物化整段序列（O(n²)）最易验证；单元 4 后两条路径共用 forward_token。
//    明确不做：其他架构（Gemma 的 q/k-norm、Qwen2MoE、注意力 sink）、批量 / 多线程。
//
// ④ demo + 多方样例：
//    🟢 最小：tests/test_model.cpp 的 TinyModelBuilder 合成 1 层微型模型
//       （n_embd=8、n_head=2、n_head_kv=1、n_ff=16、n_vocab=24）。
//    🟡 典型：真实 GGUF（TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf）→ Model::load
//       → forward_token 逐位推进，实测层数=22、维度=2048、头=32/4、词表=32000。
//    🔴 极限：Q4_K_M 量化模型 → TensorBuf 走反量化路径；形状不对在加载期就报错。
// ============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf.h"
#include "kvcache.h"

namespace minillama {

// ============================================================================
// ① 定位：模型超参数 —— 从 GGUF 元数据读出的「这一族模型长什么样」
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-hparams.cpp:llama_hparams::n_head / n_head_kv / n_gqa
//                          —— 上游同名访问器按层取同一组量，语义与本结构一致
//    llama.cpp @ 88b47a755  src/llama-arch.cpp:149:LLM_KV_NAMES（"%s." 前缀由架构名替换）
//
// ③ 动机：
//    【对照】key 名带架构前缀：llama.block_count 与 qwen3.block_count 后缀相同、
//      前缀不同（docs/03-forward.md §17.4），读取方必须先拼前缀再查。
//    【简化】上游的 n_head / n_head_kv 是 per-layer 数组，本结构只有单层值：承重判据
//      （04-对齐约束.md §2.2 第 2 条）——GQA 组数在所有层相同，砍掉 per-layer 不改变语义。
//    【概念】head_dim 与 kv_dim 是推导量，不单独存：存了就多一个真相来源，
//      可能与 n_embd / n_head 不一致。
//
// ④ demo + 多方样例：
//    🟢 微型模型（tests/test_model.cpp）：n_embd=8、n_head=2 → head_dim=4；
//       n_head_kv=1 → kv_dim=4（两个 query 头共享一个 KV 头）。
//    🟡 真实 TinyLlama-1.1B-Q8_0：n_embd=2048、n_head=32 → head_dim=64；
//       n_head_kv=4 → kv_dim=256；rope_theta=10000、rms_eps=1e-5。
//    🔴 加载期校验：n_embd % n_head != 0 直接抛「embedding_length 必须能被 head_count
//       整除」——头宽算不出来，后面每个张量的形状都对不上。
//    调用方：src/model.cpp 的 Model::load 填这些字段；src/main.cpp 只读不改。
// ============================================================================
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

// ============================================================================
// ① 定位：RMSNorm —— 只做「均方根缩放 × 可学习权重」，不减均值、不加 bias
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-cpu/ops.cpp:3860:ggml_compute_forward_rms_norm
//                          —— 上游 CPU 内核，公式与本实现一致
//    llama.cpp @ 88b47a755  ggml/src/ggml.c:3170:ggml_rms_norm（建图算子）
//
// ③ 动机：
//    【概念】LLaMA 系没有 bias、也不减均值，比 LayerNorm 少一次均值统计——省下的
//      不是那点算力，而是「均值必须在整行上先扫一遍」的串行依赖。
//    【陷阱】eps 加在均值上（sqrt(ms + eps)），不是加在输出上：ms=0 时 1/sqrt(0)
//      会是 inf，先加 eps 才能把全零行压成 0 而不是 NaN。
//    【对照】上游 LLM_NORM_RMS（src/models/llama.cpp 的 build_norm 调用）同义。
//
// ④ demo + 多方样例：
//    🟢 最小（tests/test_model.cpp）：x=[3,4]、w=[1,1]、eps=0 → ms=12.5，
//       out = [3/sqrt(12.5), 4/sqrt(12.5)]。
//    🟡 典型：真实模型 eps = 1e-5（llama.attention.layer_norm_rms_epsilon），
//       远小于 ms，只起「防 0」作用。
//    🔴 极限：x 全 0 且 eps=0 → scale = inf → out = NaN；本项目默认 eps=1e-5 避开。
//    调用方：src/model.cpp 的 forward_token 每层调用两次（attn_norm / ffn_norm）。
// ============================================================================
void rmsnorm(const float * x, const float * w, float * out, int n, float eps);

// ============================================================================
// ① 定位：IEEE 754 binary16 → binary32 的位级转换（权重加载期用）
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml.c:448:ggml_fp16_to_fp32（单元素）
//                          ggml/src/ggml.c:468:ggml_fp16_to_fp32_row（整行）
//
// ③ 动机：
//    【陷阱】必须按指数域分三个分支：正规数直接拼位模式；次正规数（指数域为 0）
//      的值是 mant * 2^-24，量级 6e-8，若把「浮点结果的位模式」当整数拼进去会被
//      截断成 0，整条次正规数分支退化成恒 0（docs/03-forward.md §17.1）。
//    【陷阱】inf / nan 要走指数域为 31 的独立分支：套用正规数公式会把 exp=31
//      算成 exp=46，得到一个巨大的有限数而不是 inf。
//    【对照】上游走硬件 F16C 指令或 GGML_FP16_TO_FP32 宏，次正规数由硬件保证；
//      本实现手写位运算，是刻意不依赖 F16C 的简化。
//
// ④ demo + 多方样例：
//    🟢 正规数（tests/test_model.cpp）：0x3C00 → 1.0；0xC000 → -2.0；0x3800 → 0.5。
//    🟡 次正规数：0x0001 → 5.9604645e-8（= 2^-24），单测单独盯这一条分支。
//    🔴 边界：0x7C00 的位模式是 +inf（指数域全 1、尾数全 0）；0x7E00 是 NaN——
//       权重里混进 NaN 会一路污染到 logits。
//    调用方：src/model.cpp 的 load_tensor_f32 把 F16 张量逐元素转过来。
// ============================================================================
float f16_to_f32(uint16_t h);

// ============================================================================
// ① 定位：矩阵乘向量 y = W·x（W 按 GGML「最快维在前」存放，纯 F32）
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-cpu/ggml-cpu.c:1254:ggml_compute_forward_mul_mat
//    llama.cpp @ 88b47a755  ggml/src/ggml-cpu/vec.cpp:11:ggml_vec_dot_f32（内层点积）
//
// ③ 动机：
//    【对照】GGML 的 W(i,o) 存在 data[i + o*ne0]，与本实现 w[o*in_dim + i] 同构，
//      所以同一份权重在两边给出同一份 y。
//    【性能】本实现按 out_dim 跨步访问 W，对 CPU 缓存不友好；上游用分块 + SIMD +
//      线程池。教学版直译数学定义，不追吞吐。
//    【简化】不做分块：承重判据（04-对齐约束.md §2.2 第 1 条）——分块只改变求和
//      顺序、不改变数学，把「一次前向 = 一次投影」的对应关系留住更重要。
//
// ④ demo + 多方样例：
//    🟢 最小（tests/test_model.cpp）：单位阵 I={1,0,0,1}、x={5,7} → y={5,7}。
//    🟡 典型：TinyLlama 的 attn_q（{2048, 2048}）→ y 长度 2048。
//    🔴 极限：in_dim 很大时（本项目 attn_q 是 2048、ffn_down 是 5632）naive 循环
//       没有分块，缓存不友好——这是【性能】取舍，不影响数值结果。
//    调用方：src/model.cpp 的 forward_token 用它做无量化路径的投影。
// ============================================================================
void matvec_f32(const float * w, const float * x, float * y,
                int in_dim, int out_dim);

// ============================================================================
// ① 定位：类型化张量缓冲 —— 量化权重不展开成 FP32，只存原始字节 + 形状
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/include/ggml.h:673:struct ggml_tensor
//                          —— 上游张量同样是「类型编号 + ne[] + 一段字节」
//    llama.cpp @ 88b47a755  src/llama-model.cpp:2765:llama_model_base::create_tensor_qkv
//
// ③ 动机：
//    【概念】GGML 张量「最快维在前」：ne0 = 输入维（行内连续），ne1 = 输出行数，
//      第 o 个输出向量在 bytes[o*ne0*elem_size ...]，连续存放。
//    【对照】量化类型的字节数不是 ne0*ne1*elem_size，而是「块数 × 块字节数」，
//      所以本结构只存字节、不存元素数——元素数由 type 唯一决定。
//    【简化】只存 ne0/ne1 两维、不存 ggml 的 nb[]/shape：承重判据
//      （04-对齐约束.md §2.2）——本项目权重全是 2 维矩阵，stride 由 ne0 唯一确定；
//      以后补 nb[] 即可直接映射 ggml_tensor，那是「接得上」而非现在必需。
//
// ④ demo + 多方样例：
//    🟢 F32 权重：{2048, 2048} → bytes 长度 = 2048*2048*4 = 16 MB。
//    🟡 Q8_0 权重（块 = 32 元素 / 34 字节）：token_embd {2048, 32000} →
//       (2048/32)*34*32000 = 69,632,000 字节，而同尺寸 FP32 要 250 MB（≈3.8x）。
//    🔴 形状不整块：ne0 不是块大小整数倍 → 加载期就报错，不留到计算期越界。
//    调用方：src/model.cpp 的 matvec / tensor_row_to_f32 按 type 分派反量化。
// ============================================================================
struct TensorBuf {
    GgmlType type = GgmlType::F32;
    uint32_t ne0 = 0, ne1 = 0;
    std::vector<uint8_t> bytes;
};

// ============================================================================
// ① 定位：类型分发的矩阵乘向量 —— F32 / F16 / Q8_0 / Q6_K / Q5_K / Q4_K 各走各的
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-cpu/ggml-cpu.c:1254:ggml_compute_forward_mul_mat
//                          —— 上游按 src0 的类型选 vec_dot 内核，量化类型自带反量化
//    llama.cpp @ 88b47a755  ggml/src/ggml-quants.c:553:dequantize_row_q8_0
//                          （Q4_K/Q5_K/Q6_K 同文件同名函数）
//
// ③ 动机：
//    【对照】上游把「反量化 + 点积」融在一个内核里（边反量化边乘累加，省一次写读）；
//      本实现先反量化一列到缓冲再点积，代码意图更清晰，1B 级模型仍可接受。
//    【概念】反量化以「列」为单位（见 dequant_column：一列 = 一个输出向量 =
//      ne0 个元素）：量化块沿 ne0 方向连续，一次取一整列的块才是内存连续的走法。
//    【简化】不融合反量化与点积：承重判据（04-对齐约束.md §2.2 第 1 条）——
//      融合只影响访存次数、不影响数值结果，数学一一对应关系才是承重件。
//
// ④ demo + 多方样例：
//    🟢 最小：F32 的 {2,2} 单位阵、x={5,7} → y={5,7}。
//    🟡 典型：TinyLlama-Q8_0 的 attn_k（{2048, 256}）→ 每列 64 个 Q8_0 块
//       反量化后与 x 点积，得 256 个值。
//    🔴 极限：ne0 > 4096（如 ffn_down 的 5632）→ 栈缓冲不够，matvec 改走堆分配。
//    调用方：src/model.cpp 的 forward_token 用它做 Q/K/V 投影与 FFN 三次投影。
// ============================================================================
void matvec(const TensorBuf & w, const float * x, float * y);

// ============================================================================
// ① 定位：取张量第 row 行并反量化成 FP32 —— 给「只查一行」的场合用（embedding 查表）
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml.c:3891:ggml_get_rows
//                          —— 上游的「按行取」算子，取出的就是反量化后的 FP32 行
//    llama.cpp @ 88b47a755  src/models/llama.cpp:37（tok_embd 建张量处）
//
// ③ 动机：
//    【概念】GGML 的「行」= 一个输出向量 = ne0 个元素；token_embd 是 {n_embd, n_vocab}，
//      token 号就是行号。
//    【陷阱】行号越界必须在取之前拦：越界的行号读到的是别的张量的字节，不报错
//      但会污染整条残差流。
//    【对照】上游 ggml_get_rows 直接给出 FP32 结果；本实现复用 dequant_column，
//      与 matvec 共用同一条反量化路径，不会出现两套口径。
//
// ④ demo + 多方样例：
//    🟢 最小：F32 的 {4, 3} 张量取第 1 行 → 4 个 FP32 值。
//    🟡 典型：TinyLlama-Q8_0 的 token_embd 取第 token 行 → 长度 2048 的向量，
//       直接送进第 0 层的 attn_norm。
//    🔴 极限：row >= ne1 → 抛「tensor_row_to_f32: 行越界」（检查在函数入口）。
// ============================================================================
void tensor_row_to_f32(const TensorBuf & w, uint64_t row, float * out);

// ============================================================================
// ① 定位：RoPE（旋转位置编码）——把「位置」编码进 query / key 的维度里
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-model.cpp:LLM_ARCH_LLAMA 的 rope 类型分派
//                          → LLAMA_ROPE_TYPE_NORM（"operating on pairs of
//                            consecutive head values"）
//    llama.cpp @ 88b47a755  ggml/src/ggml-cpu/ops.cpp:GGML_ROPE_TYPE_NORMAL 分支
//                          → rotate_pairs(n_dims, 1, ...)，offset = 1（相邻对）
//
// ③ 动机：
//    【陷阱】上游当前版本对 LLaMA 系走的是「相邻配对」(2i, 2i+1)，
//    不是老教程里的「前半后半配对」(half-split / NEOX)。两者数学上不等价——
//    要互换得在加载期对 wq / wk 做行置换。与上游逐数值对拍必须用同一约定，
//    否则 logits 在「混合 token 的多位置」序列上分叉。
//    【历史】相邻配对是 GPT-J / GPT-2 的老做法；早期 llama.cpp 曾改用 half-split，
//    后来又改回相邻配对与 HF 对齐。half-split 对应 LLAMA_ROPE_TYPE_NEOX，
//    只服务 Falcon / GPT-NeoX 家族。
//
// ④ demo + 多方样例：
//    🟢 pos=0：任意配对约定都是恒等变换（所以 pos=0 的单 token 测试测不出错）
//    🟡 pos=1, h=4, theta=10000：x=[1,0,0,0] → [cos θ0, sin θ0, 0, 0]
//       其中 θ0 = 1 * theta^0 = 1
//    🔴 调用方：src/model.cpp 的 forward_token 按 head_dim 逐头调用本函数
//       —— 跨头配对会让位置信息串到别的头上（见 docs/03-forward.md §17.2）
// ============================================================================
void rope_rotate(float * x, int h, int pos, float theta);

// ============================================================================
// ① 定位：SiLU 激活 —— silu(x) = x * sigmoid(x)，就地改写数组
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml.c:2823:ggml_silu（建图算子）
//    llama.cpp @ 88b47a755  src/llama-graph.cpp:1658（build_ffn 的 LLM_FFN_SILU 分支）
//
// ③ 动机：
//    【概念】SwiGLU 里 silu 只作用在 gate 一路，up 一路保持线性，两者逐元素相乘
//      才进 down 投影（见 src/model.cpp 的 FFN 段）。
//    【对照】上游把 silu(gate)*up 融成单个算子 ggml_swiglu_split（ggml/src/ggml.c:3060）；
//      本实现拆成 silu + 逐元素乘两步，数学相同、少一层算子抽象。
//    【简化】写 x / (1 + exp(-x)) 而不是 x * sigmoid(x)：两者数学相同，但少一次
//      函数调用；x 很大时 exp(-x) 下溢到非规格化数 → 结果就是 x，不溢出。
//
// ④ demo + 多方样例：
//    🟢 最小：silu(0) = 0（tests/test_model.cpp）。
//    🟡 典型：silu(1) = 1/(1+e^-1) ≈ 0.7311；silu(-1) = -1/(1+e) ≈ -0.2689。
//    🔴 极限：x = -100 → exp(100) 溢出成 inf → x/inf = -0（结果仍有限）；
//       x = +100 → exp(-100) 落到非规格化数（约 3.7e-44）→ 结果 = 100。
//    调用方：src/model.cpp 的 forward_token 在 FFN 段对 gate 逐元素激活。
// ============================================================================
void silu(float * x, int n);

// ============================================================================
// ① 定位：数值稳定的 softmax —— 就地改成概率分布
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml-cpu/ops.cpp:5567:ggml_compute_forward_soft_max
//    llama.cpp @ 88b47a755  ggml/src/ggml-cpu/ops.cpp:5455:ggml_compute_forward_soft_max_f32
//                          —— 上游同样先取最大值再 exp
//
// ③ 动机：
//    【陷阱】不减最大值时 float32 的 exp(x) 在 x ≈ 89 就溢出成 inf，inf/inf = NaN。
//      先减 max 把最大项归到 exp(0)=1，指数范围被压到 (-inf, 0]。
//    【概念】分母是「减完最大值之后」的 exp 之和：分子分母同乘 e^-max，比值不变，
//      所以减最大值不改变结果，只是把溢出风险挪走。
//    【对照】上游 ggml_vec_soft_max_f32(ne00, dp, wp, max) 与本实现同构。
//
// ④ demo + 多方样例：
//    🟢 最小（tests/test_model.cpp）：x={3,1,-2,8,0.5} → 各项和为 1.0，
//       且序关系保持（最大项 8 在前，最小项 -2 在后）。
//    🟡 典型：注意力 scores（长度 pos+1）→ 变成权重，和为 1。
//    🔴 极限：x 全为 -1000 → max=-1000，减完全 0 → 均匀分布 1/n（不 NaN）。
//    调用方：src/model.cpp 的 forward_token 与 src/sampler.cpp 的采样链。
// ============================================================================
void softmax(float * x, int n);

// ============================================================================
// ① 定位：一层 Transformer 的全部权重 —— 两个 norm 向量 + 四个注意力投影 + 三个 FFN 投影
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/models/llama.cpp:51-77（llama_model_llama 逐层建
//                          attn_norm / attn_q / attn_k / attn_v / attn_output /
//                          ffn_norm / ffn_gate / ffn_up / ffn_down）
//    llama.cpp @ 88b47a755  src/llama-model.cpp:2765:llama_model_base::create_tensor_qkv
//
// ③ 动机：
//    【对照】上游优先找融合的 attn_qkv.weight（一个矩阵装下 Q/K/V），找不到才退回
//      三个独立张量；本项目直接用三个独立张量，与 GGUF 里 LLaMA 模型的落盘形态一致。
//    【概念】norm 恒为 F32/F16、加载期转 F32；投影矩阵可能是量化类型，所以两类
//      权重用了两种容器（std::vector<float> vs TensorBuf）。
//    【对照】张量名统一 blk.N.*（层号在名字里），与上游 LLM_TENSOR_NAMES 的
//      "blk.%d.attn_q" 形式一致（src/llama-arch.cpp:384）。
//
// ④ demo + 多方样例：
//    🟢 微型模型（tests/test_model.cpp）：1 层 → attn_norm 长度 8、wq 是 {8, 8}。
//    🟡 真实 TinyLlama-1.1B-Q8_0（22 层）：wq {2048,2048}、wk/wv {2048,256}、
//       wo {2048,2048}、w_gate/w_up {2048,5632}、w_down {5632,2048}。
//    🔴 形状不符：wk 的 ne1 必须是 kv_dim = n_head_kv * head_dim = 256，
//       加载期不满足即报错。
// ============================================================================
struct LayerWeights {
    std::vector<float> attn_norm, ffn_norm;  // norm 恒为 F32/F16，加载期转 F32
    TensorBuf wq, wk, wv, wo;                // 注意力投影（可能是量化类型）
    TensorBuf w_gate, w_up, w_down;          // FFN（可能是量化类型）
};

// ============================================================================
// ① 定位：模型对象 —— 持有超参与全部权重，并提供「全量 / 增量」两条前向入口
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-model.h:550:struct llama_model
//    llama.cpp @ 88b47a755  src/llama-context.cpp:4072:llama_decode
//                          —— 上游的增量解码入口（批次进、批次 logits 出）
//
// ③ 动机：
//    【概念】增量路径是唯一的计算路径：forward() 只是「临时 KVCache + 逐位置收集」
//      的外壳，真正的计算全在 forward_token 里，所以两条路径不可能分叉。
//    【对照】上游用 llama_batch 一次塞多个 token，本项目一次只算一个位置——
//      少一条「批次内并行」的路径，语义更窄但更直白。
//    【简化】不支持多序列 / 随机插入 / slot 复用：承重判据（04-对齐约束.md §2.2
//      第 2 条）——「缓存 = 按位置存放的历史 K/V」这一核心语义完整保留。
//
// ④ demo + 多方样例：
//    🟢 最小：Model::load(gguf) → forward({1}) → 长度 n_vocab 的 logits。
//    🟡 典型：forward_token(t, pos, kv) 逐位推进，pos 从 0 递增。
//    🔴 极限：pos 与 kv.size() 不一致 → 抛「要求 kv.size() == pos（顺序生成）」。
// ============================================================================
class Model {
public:
    // ============================================================================
    // ① 定位：从 GGUF 读出超参与全部权重，做完加载期形状校验再交出可用的模型
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  src/llama-model.cpp:1060:llama_model_base::load_hparams
    //    llama.cpp @ 88b47a755  src/llama-model.cpp:1250:llama_model_base::load_tensors
    //    llama.cpp @ 88b47a755  src/llama-model-loader.cpp:858:llama_model_loader::check_tensor_dims
    //
    // ③ 动机：
    //    【对照】上游分「读超参」与「读张量」两段，本实现合成一个 load：调用方
    //      拿到的要么是完整可用的模型，要么什么都没有。
    //    【陷阱】形状校验必须在加载期：超参与张量形状同时在手时才判得了不变式，
    //      推迟到前向等于让错误漂到运行期（docs/03-forward.md §17.5）。
    //    【对照】架构前缀来自 general.architecture（qwen3.* 与 llama.* 后缀同名），
    //      与上游 LLM_KV::operator()（src/llama-arch.cpp:873）同构。
    //
    // ④ demo + 多方样例：
    //    🟢 最小：tests/test_model.cpp 的 TinyModelBuilder 写出的 1 层微型 GGUF
    //       → Model::load 成功，n_layer=1、n_embd=8、n_head=2、n_vocab=24。
    //    🟡 典型：models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf → 加载成功，实测摘要行
    //       「层数=22 维度=2048 头=32/4 词表=32000」。
    //    🔴 极限：缺张量 / 形状不符 / ne0 不是量化块大小整数倍 → 全部在加载期
    //       抛异常并带上张量名，绝不带病进入前向。
    // ============================================================================
    static Model load(const GgufFile & gguf);

    const HParams & hparams() const { return hp_; }

    // ============================================================================
    // ① 定位：全量前向 —— 给一整段 token 序列，返回每个位置的 logits
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  src/llama-context.cpp:4072:llama_decode
    //                          —— 上游同样一次喂多个 token，返回该批次的 logits
    //    llama.cpp @ 88b47a755  src/models/llama.cpp:99:llama_model_llama::graph<embed>::graph
    //
    // ③ 动机：
    //    【概念】它是「临时缓存 + 逐位置收集」的外壳，不是第二条计算路径：内部
    //      新建 KVCache 逐位置调 forward_token，再把各行 logits 拼起来。
    //    【概念】返回行主序的 n × n_vocab：第 p 行的 [p*n_vocab, (p+1)*n_vocab)
    //      就是位置 p 的分布，调用方无需再转置。
    //    【对照】上游把整批 token 塞进一次 llama_decode，用计算图并行算完；
    //      本实现逐位置串行算，数值路径完全相同。
    //
    // ④ demo + 多方样例：
    //    🟢 最小：forward({1}) → 长度 n_vocab 的 logits；重复调用结果逐位相同
    //       （tests/test_model.cpp 的确定性断言）。
    //    🟡 典型：forward({1,5,9}) → 长度 3*n_vocab；每行只看得见自己及之前的位置
    //       （因果性）。
    //    🔴 极限：序列长度超过 n_ctx → 抛「KVCache 已满（超过 context length）」；
    //       生产路径请用 forward_token 配自己的长寿命缓存。
    // ============================================================================
    std::vector<float> forward(const std::vector<int> & tokens) const;

    // ============================================================================
    // ① 定位：增量前向 —— 只算 position=pos 这一个新 token，历史 K/V 从缓存读
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  src/llama-context.cpp:4072:llama_decode
    //    llama.cpp @ 88b47a755  src/llama-memory.h:73:struct llama_memory_i
    //                          —— 上游 KV 缓存的抽象接口，本项目 KVCache 是它的替身
    //    llama.cpp @ 88b47a755  src/llama-graph.cpp:2555:llm_graph_context::build_attn
    //
    // ③ 动机：
    //    【概念】「本位置已写入」≠「全局 size」：本层 K/V 在算完本层投影后就追加进
    //      缓存，但注意力要读的是 0..pos 全部位置——所以打分时读缓存、追加时写缓存，
    //      两件事的顺序不能颠倒（docs/03-forward.md §16.1）。
    //    【陷阱】调用前必须 kv.size() == pos：跳位置会让注意力读到空的历史，
    //      但代码不会自动补齐——顺序性是调用方的契约，错了要在入口就报。
    //    【概念】返回长度 n_vocab 的 logits：这是「一次前向 = 一个 token 进、一份
    //      logits 出」的最小契约，采样器直接消费它。
    //
    // ④ demo + 多方样例：
    //    🟢 最小：pos=0、kv 为空 → 只看自己，返回 n_vocab 个分数。
    //    🟡 典型：生成循环里 pos 从 0 递增；tests/test_generate.cpp 断言显式增量
    //       与 forward() 的全量结果逐位一致（误差 < 1e-4）。
    //    🔴 极限：pos 与 kv.size() 不一致 → 抛「要求 kv.size() == pos（顺序生成）」；
    //       token id 越界 → 抛「token id 越界」。
    // ============================================================================
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
