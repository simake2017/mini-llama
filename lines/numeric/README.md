# numeric 族 —— 数值域

> 两条线都发生在 **logits 之前**：一条把磁盘上的量化字节还原成 F32 权重，
> 一条把 logits 变成一个采样出来的 token。
> 它们是推理链上「数字 → 数字」的两端，中间夹着整个网络。

## 一 链路图

```text
 ┌───────────────────────── dequant 线 ─────────────────────────┐
 │                                                                │
 │  GGUF 张量字节                                                  │
 │      │  块布局（34 / 144 / 176 / 210 字节一块）                   │
 │      ▼                                                         │
 │  ┌──────────────┐   d / dmin / sc / m / qs / qh                │
 │  │ src/quant.cpp│   get_scale_min_k4（6bit scale 双路解包）        │
 │  └──────┬───────┘                                              │
 │         ▼  F32 权重行                                          │
 │  ┌──────────────┐   分发：F32 / F16 / Q8_0 / Q6_K / Q5_K / Q4_K  │
 │  │ src/model.cpp│   matvec → y = W x                            │
 │  └──────┬───────┘                                              │
 │         ▼                                                      │
 │    层内计算（见 infer 族）                                       │
 └────────────────────────────────────────────────────────────────┘

 ┌───────────────────────── sample 线 ──────────────────────────┐
 │                                                                │
 │  logits[0..n_vocab)                                           │
 │      │  temp <= 0 → argmax（贪心，完全不碰随机数）                │
 │      ▼                                                        │
 │  ┌──────────────┐   softmax（先减最大值，数值稳定）                │
 │  │ src/model.cpp│   ← 这一份是「未加温度」的概率                 │
 │  └──────┬───────┘                                              │
 │         ▼  概率分布                                             │
 │  ┌──────────────┐   步骤 3 top-k 截断  步骤 4 top-p 核截断（含越界那个）│
 │  │ src/sampler.cpp│  步骤 5 温度作用在存活候选上  步骤 6 按概率抽样    │
 │  └──────┬───────┘                                              │
 │         ▼                                                      │
 │      token id → 写回 KV Cache（见 infer 族）→ 下一个 token        │
 └────────────────────────────────────────────────────────────────┘
```

★ 跨层点三处：**块字节 → F32 行**（quant → model）、
**F32 行 → matvec**（quant → model）、**logits → token**（model → sampler → model）。

## 二 逐跳对照

### 线 G：`dequant`（8 跳）

| 跳 | 本仓库落点 | 上游对照（llama.cpp @ 88b47a755） | 真机命令 |
|---|---|---|---|
| 1 | 块布局常量（`src/quant.h`：34 / 144 / 176 / 210 字节每块） | `ggml/src/ggml-common.h:block_q8_0` / `block_q4_K` / `block_q5_K` / `block_q6_K` | `MINILLAMA_MODEL=… ./build/line_dequant` |
| 2 | `dequant_row_q8_0`（`src/quant.cpp`） | `ggml/src/ggml-quants.h:52:dequantize_row_q8_0` | 同上 |
| 3 | `get_scale_min_k4` + `dequant_row_q4_k`（`src/quant.cpp,41`） | `ggml/src/ggml-quants.c:880:get_scale_min_k4` + `dequantize_row_q4_K` | 同上 |
| 4 | `dequant_row_q5_k`（`src/quant.cpp`，第 5 比特在 qh[32]） | `ggml/src/ggml-quants.c:dequantize_row_q5_K` | 同上 |
| 5 | `dequant_row_q6_k`（`src/quant.cpp`，有符号 6bit，值域 -32~31） | `ggml/src/ggml-quants.c:dequantize_row_q6_K` | 同上 |
| 6 | `GgufFile::tensor_data` 取真实张量字节 → 反量化一行 | `ggml/src/gguf.cpp:1179:gguf_get_tensor_offset` + `gguf_get_tensor_size` | 同上 |
| 7 | `model.cpp:matvec`（`src/model.cpp`）按张量类型分派 | `ggml/src/ggml.c:631:type_traits` 的 `.to_float` / `.vec_dot` | 同上 |
| 8 | 量化权重进入计算图 | `src/llama-context.cpp:1693:llama_context::decode` | 同上 |

### 线 H：`sample`（8 跳）

| 跳 | 本仓库落点 | 上游对照（llama.cpp @ 88b47a755） | 真机命令 |
|---|---|---|---|
| 1 | 输入未归一化 logits（合成 `{3,2,1,0,-1}`，本线不依赖模型文件） | — （教学用合成输入） | `./build/line_sample` |
| 2 | `model.cpp:softmax`（**未加温度**的概率，`src/sampler.cpp`） | `ggml/src/ggml-cpu/ops.cpp` 的 `ggml_soft_max` | 同上 |
| 3 | 温度对分布形状的影响（temp 越大越平，`src/sampler.cpp`） | `src/llama-sampler.cpp:265:llama_sampler_temp_impl` | 同上 |
| 4 | 贪心分支 `temp<=0`（`src/sampler.cpp`） | `src/llama-sampler.cpp:935:llama_sampler_greedy` | 同上 |
| 5 | top-k 截断（`partial_sort` 取前 k，`src/sampler.cpp`） | `src/llama-sampler.cpp:321:llama_sampler_top_k_impl` | 同上 |
| 6 | top-p 核截断（**含跨阈值的那个**，`src/sampler.cpp`） | `src/llama-sampler.cpp:1343:llama_sampler_top_p` | 同上 |
| 7 | 温度作用在存活候选 + 按概率抽样（`src/sampler.cpp`） | `src/llama-sampler.cpp:1026:llama_sampler_dist` | 同上 |
| 8 | 挑出的 token id | `src/llama-sampler.cpp:880:llama_sampler_chain_add`（链的最后一节） | 同上 |

★ **节序**（本单元最容易被忽略的一处对齐）：上游默认链是
`top-k → top-p → … → temperature → dist`（`common/common.h:259` 的 samplers 列表 +
`common/sampling.cpp:376` 追加 dist），**截断发生在温度生效之前**。
本仓库的链序与之相同 —— 详见 §三 的补充问答。

## 三 学习问题的答案

> **量化块里存的是「缩放系数 + 低位量化值」，为什么每个子块要单独存 scale/min，
> 而不是整行共用一个缩放系数？**

因为**同一行内的不同位置，数值分布的尺度差别可以非常大**。

以 Q4_K 为例：一行 256 个权重被切成 4 个子块（每块 64 个元素），
每个子块各有自己的 `sc`（scale）与 `m`（min）：

```text
value[i] = d * sc[子块(i)] * nibble(i) - dmin * m[子块(i)]
```

**为什么必须分组？** 看一个具体的例子——LLM 的权重矩阵里，
**不同列（对应不同的 hidden 维度）的量级天差地别**：

| 权重类型 | 典型量级 | 4bit 能表示的档位数 |
|---|---|---|
| embedding / 大矩阵的主干 | ±0.02 | 需要精细档位 |
| 少数离群通道（ outlier channel ） | ±2.0 | 需要粗档位也不丢信息 |

如果**整行共用一个 scale**：

```text
一行共用 d，值域被 [-d*7, d*7] 覆盖
若 d 按最大值 2.0 取 → 普通权重 0.02 落在 0.02/2.0 = 1% 的量化格点上
                     → 4bit 的 16 个档位里只剩不到 0.2 个可用
                     → 普通权重全部塌到 0
```

**分组之后**（子块内自适应）：

```text
子块 A（普通权重，量级 0.02）→ d_A ≈ 0.02/7，16 个档位铺满 [-0.02, 0.02]
子块 B（离群通道，量级 2.0）→ d_B ≈ 2.0/7，16 个档位铺满 [-2.0, 2.0]
→ 每组都用自己的"刻度尺"，误差从"整行一个刻度"降到"每 64 个元素一个刻度"
```

**核心一句话：量化的误差是 `scale * 量化格距`，
scale 越小格子越细精度越高；但一组元素共享一个 scale 时，
scale 必须由这组里的**最大值**决定——所以要把量级不同的元素分到不同组，
每组才能各自用细刻度。**

**为什么 `scale` 和 `min` 要成对存？**

Q4_K 用的是**非对称量化**（affine）：

```text
value = d * sc * nibble - dmin * m
```

`nibble` 是 4bit（0~15，无符号），只能表示「非负」的一侧。
真实权重有正有负，所以要减掉一个下界 `dmin * m`。
用无符号 4bit + 下界，比用有符号 4bit（-8~7）多表达一个端点，
且**最小值附近的分辨率更高**（因为 0~15 的 16 格覆盖 [-dmin, -dmin+15*d]）。

**为什么 scale/min 要用 6bit 压进 12 字节？**

K-quants 的行是 256 元素 / 4 个子块 → 需要 4 个 sc + 4 个 m = 8 个 6bit 值 = 48 bit = 6 字节。
但存储格式只给了 12 字节（96 bit）——比需要的多，因为**实际实现里 4 个子块被展开成 8 个"半块"**
（每 32 个元素一个 scale），6bit × 8 = 48bit 的 scale + 48bit 的 min = 96 bit = 12 字节，正好。

`get_scale_min_k4` 的两路解包就是在拆这 12 字节：

```cpp
if (j < 4) {                    // 前 4 个半块：高位直接取
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
} else {                        // 后 4 个半块：低 4 位在本字节，高 2 位在 4 字节前的字节
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
}
```

**Q6_K 为什么没有 dmin？** 因为 Q6_K 用**有符号 6bit**（-32~31），
量化值本身就带符号，不需要额外的下界——省下的字节全部给了 scale（16 个 int8）。

### 补充问答：采样链的**节序**为什么不能随便改？

上游 llama.cpp 的默认链是（`common/common.h:259` 的 samplers 列表 +
`common/sampling.cpp:376` 追加 dist）：

```text
PENALTIES → DRY → TOP_N_SIGMA → TOP_K → TYPICAL_P → TOP_P → MIN_P → XTC → TEMPERATURE → DIST
```

**温度排在截断之后**。这不是随手排的——它有一个可观测的后果：

| 节序 | temp=2、top_p=0.9 时保留几个候选 |
|---|---|
| 上游（先截断，后加温） | **3 个**：未加温度的累计 0.636 / 0.871 / 0.957，第 3 个跨过 0.9 |
| 先加温再截断（HF 的常见顺序） | **4 个**：加温后的累计 0.429 / 0.689 / 0.846 / 0.942，第 4 个跨过 0.9 |

也就是说，**同样的 `--temp 2 --top-p 0.9`，上游和「先加温」的实现会选出不同的候选集合**，
生成的文本从这一步开始就分叉了。

温度是单调变换，**不改变 token 之间的排序**——所以 **top-k 的候选集在两种节序下完全相同**，
差异只出现在 top-p（因为它依赖概率的累积值，而概率的集中程度会被温度改变）。

所以本仓库的采样链**照上游的节序实现**：
`贪心 → softmax（未加温度）→ top-k → top-p → 温度 → 抽样`。
这不是「谁更主流」的问题（HF 的顺序确实更常见），而是**还原项目的对齐目标只有一个**。

## 四 认知纠错记录

| # | 曾经的错误假设 | 实际情况 | 判定依据 |
|---|---|---|---|
| 1 | 「Q4_K 的量化值是有符号 4bit（-8~7）」 | **不成立**。Q4_K 是**无符号 4bit + 减去 dmin** 的非对称量化 | `src/quant.cpp:dequant_row_q4_k` 的 `- m1` |
| 2 | 「Q6_K 的 6bit 量化值是 0~63」 | **不成立**。Q6_K 是**有符号 6bit**，解包后要 `-32` | `src/quant.cpp:dequant_row_q6_k` 的 `- 32` |
| 3 | 「top-p 截断时把跨过阈值那个 token 丢掉，分布才干净」 | **不成立**。丢掉会**永久丢掉一部分概率质量**，且分布会变成非归一化；正确做法是**保留它**（让累计概率 ≥ p），然后按截断后的总和重新归一 | `src/sampler.cpp` 的 `cut = i + 1`（含越界那个） |
| 4 | 「top-k 会改变 softmax 的结果」 | **不成立**。top-k 在 softmax **之后**做，只截断集合不改数值；真正改变分布的是温度 | `src/sampler.cpp` |
| 5 | 「贪心采样（temp=0）也会消耗随机数，所以两次结果可能不同」 | **不成立**。`temp<=0` 走 argmax 分支，完全不碰 rng | `src/sampler.cpp` |
| 6 | 「量化块的字节数可以从小数部分猜」 | **不成立**。块尺寸是**类型定义的常量**（Q8_0=34/32、Q4_K=144/256、Q5_K=176/256、Q6_K=210/256），必须查表 | `src/quant.h` 的常量 |
| 7 | 「top-p 应该在温度之后算才符合直觉」 | **就本项目的对齐目标而言不成立**。上游默认链是 top-k → top-p → temperature → dist，**截断排在温度之前**；HF 的顺序虽然更常见，但还原的目标是上游 | `common/common.h:259` 的 samplers 列表顺序；`line_sample` 第 6 跳的 temp=2 用例 |
| 8 | 「温度会改变 top-k 选出的候选」 | **不成立**。温度是单调变换，不改变排序，top-k 的候选集与温度无关 | 数学性质：logit 排序 ⟺ 概率排序 |

## 五 真机实测

机器：Linux 5.10 / 8 核 / 16GB；模型：`models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf`；
日期：2026-11-11。

```bash
$ MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_dequant
观测 : Q8_0 合成块：d=1.0，qs=1..32 → y = 1..32
观测 : Q4_K 合成块：d=1.0 dmin=0 sc=1 m=0 nibble=1 → y 全 1.0
观测 : Q5_K 合成块：qh=0 → y 全 1.0
观测 : Q6_K 合成块：量化值 0 → -32，y 全 -32
观测 : 真实 token_embd 第 0 行第 1 块的 F32 值域 = [-1.51396e-05, 1.46627e-05]
观测 : matvec(Q8_0) 的行和 = 32
== 线 dequant 走完：10/10 项通过
```

```bash
$ ./build/line_sample
观测 : 输入 logits = [3, 2, 1, 0, -1]
观测 : temp=1.0 的概率 = [0.636409, 0.234122, 0.086129, 0.031685, 0.011656]（和 = 1）
观测 : temp=0.25 → P(0) = 0.981684
观测 : temp=2.0  → P(0) = 0.428656
观测 : top_k=1 → 200 次采样全部命中 token 0
观测 : top_p=0.9 → 200 次采样命中 token 0/1/2，token 3/4 从未被抽中
观测 : temp=2.0 时 token 3 是否被抽中 = 否   ← 截断排在温度之前
观测 : 相同 seed 两次采样结果一致
== 线 sample 走完：11/11 项通过
```

**耗时**：dequant 线 0.03 秒（不跑前向）；sample 线同量级（纯数值，不依赖模型文件）。

## 六 差异注明

| # | 差异 | 我的实现 | 真机（llama.cpp @ 88b47a755） | 为什么可接受 |
|---|---|---|---|---|
| 1 | 反量化粒度 | 一次解一整行到 F32 缓冲 | `to_float` 同样是整行；计算内核另有 `vec_dot` 直算路径 | 语义相同；本仓库走 to_float 路径，因为要复用同一份 F32 行做矩阵乘 |
| 2 | 量化类型覆盖面 | 只有 Q8_0 / Q6_K / Q5_K / Q4_K 四种 | 全部 GGML 量化类型（含 IQ 系、Q3_K、Q2_K 等） | 本项目只需要覆盖 TinyLlama 家族用到的类型（见 `docs/05-quantization.md`） |
| 3 | 采样链 | 固定顺序的单个函数 | `llama_sampler_chain` 可任意增删节 | **节序已与上游默认链对齐**（top-k → top-p → temperature → dist），各节语义相同；少了任意组合的自由度（`docs/constraints/04-对齐约束.md` 已登记为实现档可替换） |
| 4 | top-p 判据 | 累加超过 p 即停（**含**越界那个） | `llama_sampler_top_p` 同样是「保留跨阈值那个」 | 一致 |
| 5 | 抽样分布 | `uniform_real_distribution(0, total)` 线性扫描累加 | `llama_sampler_dist` 用同样的逆 CDF 思路 | 语义一致；分布函数的具体实现不同（实现档） |
| 6 | 温度为 0 | 单独走 argmax 分支 | `llama_sampler_greedy` 作为独立后端 | 语义一致 |
| 7 | 量化后的训练 | 无（只读不写） | `quantize_row_*_ref` 也有 | 本项目只做推理，不做量化 |

## 七 可观测点

| 量 | 本实现的接口 | 真机对应（llama.cpp @ 88b47a755） |
|---|---|---|
| 块尺寸表 | `src/quant.h` 常量 / `ggml_block_params` | `ggml/src/ggml.c:type_traits` 的 `.blck_size` / `.type_size` |
| 反量化结果 | `dequant_row_q8_0` 等的 `y[]` | `dequantize_row_q8_0` 等的 `y[]` |
| 类型名 | `ggml_type_name(GgmlType)` | `ggml/src/gguf.cpp:1003:gguf_type_name` |
| 张量字节数 | `GgufFile::tensor_nbytes` | `gguf_get_tensor_size` |
| 矩阵乘向量 | `model.cpp:matvec` | `ggml_compute_forward_mul_mat` / `ggml_vec_dot_q8_0` |
| 概率分布 | `model.cpp:softmax` | `ggml_soft_max` |
| 采样结果 | `Sampler::sample(logits, n)` | `llama_sampler_sample` |
| 采样参数 | `SamplerParams{temp, top_k, top_p, seed}` | `llama_sampler_chain_params` 的对应字段 |

> 真机没有「教学线」这种观测入口——它的所有量都藏在
> `llama_sampler` 的中间数组里，只能靠加日志或 gdb 看。
> 本仓库把「概率分布」「截断后的候选集」直接打印出来，
> 是为了**让采样链的每一步都能被肉眼检查**——这是教学便利代码，不是性能优化。
