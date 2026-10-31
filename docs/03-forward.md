# 单元 3：模型结构与前向计算 —— RMSNorm / RoPE / GQA / SwiGLU 的朴素实现

（对应真实 llama.cpp 的 src/llama-model.cpp + src/llama-graph.cpp + ggml CPU 内核）

## ① 我实现了什么

- **超参数加载**：从 GGUF 读 block_count / embedding_length / head_count / head_count_kv / feed_forward_length / rms_epsilon / rope.freq_base。key 带架构前缀（`llama.*`、`qwen3.*`），取 `general.architecture` 作前缀，Qwen2/3 因架构派生自 LLaMA 可直接复用
- **权重加载**：LLaMA 主干 9 种张量 + embedding + output_norm + output（缺省自动 tied embeddings），F32 原样 / F16 逐元素转 F32，**量化类型明确报错指向单元 5**
- **全量前向**（无 KV cache 教学版）：
  - 每层物化全序列 X(n×d)，注意力对每个位置 p 只可见 0..p（因果掩码）
  - RMSNorm → Q/K/V 投影 → RoPE（相邻元素对旋 (2i,2i+1)，与 llama.cpp master 的 `LLAMA_ROPE_TYPE_NORM` 对齐）→ **GQA**（KV 头按需索引，不物理复制）→ 输出投影 + 残差 → SwiGLU FFN + 残差
  - 输出头：最终 RMSNorm + 词表投影
- **基础算子独立可测**：rmsnorm / matvec / rope_rotate / silu / softmax / f16_to_f32

关键设计决策：

- **单元 3 故意不用 KV cache**：O(n²) 的全量注意力是最容易对照数学公式的形态。单元 4 引入缓存时做"增量 vs 全量逐位一致"的回归测试，把优化前后的等价性钉死——这是本仓库最重要的一个测试。
- **RoPE 用相邻配对 (2i,2i+1)**：llama.cpp master（88b47a7）`llama-model.cpp` 中 `LLM_ARCH_LLAMA` 返回 `LLAMA_ROPE_TYPE_NORM`（"operating on pairs of consecutive head values"，即 GPT-J/GPT-2 风格交错旋转），**不是**老教程和旧版 llama.cpp 的 half-split（两半对旋，ggml 叫 NEOX，实际 Falcon/GPT-NeoX 才用）。两种约定数学上不等价（要对 wq/wk 做相应行置换才可互换），与 llama.cpp 逐数值对照必须用同一种。这个坑的排查过程见 ④。
- **GGUF 布局"最快维在前"**：`matvec` 明确注释 `y[o] = Σ x[i]·W(i,o)`，`W(i,o)` 在 `data[i + o·in_dim]`，避免行列主序混淆。

## ② 真实 llama.cpp 对应源码

| mini-llama | 真实版 | 说明 |
|---|---|---|
| `Model::load` | `llama-model-loader.cpp` + `llama-hparams.cpp` | hparams 键名映射逻辑一致（LLM_KV_NAMES） |
| `rmsnorm` | `ggml-cpu` 的 `ggml_rms_norm` 内核 | naive 版 |
| `matvec` | `ggml_mul_mat` 的 CPU 实现 | ggml 分块+SIMD+线程，我们是三层循环 |
| `rope_rotate` | `llama-model.cpp` 的 rope 类型分发 → `ggml_rope` | LLaMA 系走 `LLAMA_ROPE_TYPE_NORM`（相邻对旋）；ggml 还有 NEOX（半切，Falcon 用）等风格 |
| GQA 注意力 | `build_attn()` + `llama-kv-cache` 的 repeat | 真实版物化复制 KV，我们按索引共享 |
| SwiGLU FFN | `build_ffn()`（ggml 里是 mul_sigmoid_mul 组合） | 数学等价 |
| 全量前向 | `llama-graph.cpp` 整张计算图 | ggml 先建图再执行；我们是直接循环 |
| f16_to_f32 | `ggml_compute_fp16_to_fp32` | 位操作版 |

## ③ 真实版比我多做了什么、为什么

| 真实版多做 | 为什么可以砍 |
|---|---|
| **GGML 计算图抽象**（先建图、后执行、可复用） | 这是性能/工程架构，不是数学；直接循环更适合教学 |
| **KV cache** | 刻意推迟到单元 4，用它讲"算法不变式驱动的优化" |
| 几十种架构变体（Gemma q-norm、MoE、注意力 sink、RWKV 循环结构等） | 每个变体都是局部的数学替换，主干掌握后都是查表工作 |
| 线程池/SIMD/缓存分块 | 纯性能优化，naive 版 + 注释指路即可 |
| rope _scaling（YaRN/NTK 外推）、clip、各种掩码类型 | 长上下文扩展技术，7B 以内用不到 |
| 权重 mlock / NUMA / 张量并行 | 工程运维特性 |

## ④ 踩坑记录

- **F16 次正规数转换 bug**：`sign | (uint32_t)(float_val)` 把浮点值的截断结果当位模式用，次正规数全变 0。教训：位操作和数值运算混在一行必须逐分支测（0x0001 这个用例就是为此补的）。
- **RoPE 跨头配对**：最初实现把 q 的 2048 维当作一个整体做半切旋转（pair i, i+1024，跨 32 个头），正确做法是按 head_dim=64 逐头旋转。pos=0 时 RoPE 是恒等变换，所以单元 3/4 的全部单 token 测试都漏检了它——**任何位置相关算子用 pos=0 测等于没测**。
- **RoPE 配对约定与 llama.cpp master 不一致**（排查耗时最长的一个坑）：单 token、同 token 重复序列（如 [1576,1576]）下任意配对约定结果都相同（恒等 / v 相同权重无关），只有"混合 token 的多位置"用例才能暴露。排查链路：ours 与独立 numpy 仲裁者（从 GGUF 原始字节手写解析+前向，零复用本仓库代码）逐位一致 → 证明我们的实现自洽 → 用仲裁者做 RoPE 模式扫描（恒等 / 相邻对 / 半切）发现 llama.cpp 走相邻对 → 查源码证实 `LLM_ARCH_LLAMA → LLAMA_ROPE_TYPE_NORM`。教训：与上游逐数值对照前先确认"上游当前版本的行为约定"，教程/旧源码里的"标准做法"可能已经变了；构造对照实验时要设计能区分各假设的用例。
- **qwen3 超参 key 是 `qwen3.*` 不是 `llama.*`**：llama.cpp 用 LLM_KV_NAMES 按架构映射键名，直接抄了 llama 的 key 名导致"超参不完整"。修复方式（前缀参数化）正好复刻了真实版的映射结构。
- **张量尺寸校验放在加载期**：宁可启动时报"张量尺寸不符: blk.0.attn_k"，不要前向到一半段错误——错误信息要带层号和张量名。

## ⑤ 验收记录

- 合成用例 4 组全绿：算子数学（RMSNorm 手算值 / f16 抽查 / RoPE 恒等+保模长 / softmax 归一）、前向形状与确定性、**因果性逐位验证**（改未来 token 不影响过去 logits）、F16+tied+双层、量化拒绝
- 真实模型冒烟：qwen3 1.7B Q4_K_M 正确报"量化类型 Q4_K，FP32 前向不支持（单元 5 实现量化推理）"
- ctest 3 个单元全绿
