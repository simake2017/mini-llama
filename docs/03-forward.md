# 技术文档：单元 3 模型结构与前向计算

- 读者：想弄懂 LLaMA 系模型「权重怎么进、注意力怎么算」的人，以及要对照 llama.cpp 源码核查约定差异的维护者
- 审法：先过 §16.1 的术语对齐（确认两边说的是同一件事），再按 §17 逐坑自检，最后用 §9 的数据形状核对每个张量的 ne0/ne1
- 名词版本：v1

## 变更记录

| 日期 | 版本 | 改了哪条 |
|---|---|---|
| 2026-11-11 | v1 | 从旧五段式迁到 mini-restore v1.2 骨架（§13~§17 补齐） |

> 参考实现 = llama.cpp b10143（commit 88b47a755）
> 来源     = /root/ai/llama.cpp @ 88b47a755
> 技能版本 = mini-restore v1.2

## 第一部分：概要设计

### §1 概念

#### 1.1 实体

| 实体 | 业务含义 | 谁写 | 谁读 | 空值语义 |
|---|---|---|---|---|
| `HParams`（src/model.h） | 模型形状与数值超参 | `Model::load` | `forward_token` / CLI / 测试 | `n_layer`/`n_embd`/`n_head`/`n_ff` 任一为 0 即抛「模型超参数不完整」；`n_head_kv` 缺省取 `n_head`；`n_vocab` 由嵌入表 ne1 推出 |
| `LayerWeights`（src/model.h） | 单层 9 个权重：2 个 norm 向量 + 4 个注意力投影 + 3 个 FFN 投影 | `Model::load` | `forward_token` | 无空值：张量缺失即抛「模型缺少张量: <名字>」 |
| `TensorBuf`（src/model.h） | 类型化张量缓冲（type + ne0/ne1 + 原始字节） | `load_tensor_buf` | `matvec` / `tensor_row_to_f32` | `ne0/ne1` 为 0 表示未装载；`bytes` 为空表示空张量 |
| `Model`（src/model.h） | 已加载、可前向的模型 | `Model::load` | `forward` / `forward_token` / CLI | 构造失败即抛异常，不存在「空模型」状态 |

#### 1.2 枚举值对象

| 枚举 | 取值 | 出现位置 |
|---|---|---|
| `GgmlType`（本单元用到的子集） | F32 / F16 / Q4_K / Q5_K / Q6_K / Q8_0 | 权重加载按类型分支：F32 原样、F16 转 F32、其余在单元 3 报错 |
| 异常类别（`std::runtime_error`，共 13 处抛出点） | 「模型超参数不完整」「embedding_length 必须能被 head_count 整除」「模型缺少张量」「张量 ... 必须是 2 维矩阵」「... 的输入维 ... 不是块大小 ... 的整数倍」「... 是量化类型 ...，FP32 前向不支持」「张量尺寸不符」「norm 张量尺寸不符」「output.weight 尺寸与词表不符」「tensor_row_to_f32: 行越界」「matvec: 未支持的量化类型」「token id 越界」「forward_token: 要求 kv.size() == pos（顺序生成）」 | 逐条见 §8 |

#### 1.3 名词

| 名词 | 一句大白话 |
|---|---|
| `gguf` | 一个模型文件：头部元数据 KV + 张量表 + 张量原始字节 |
| `hparams` | 模型形状参数（层数/维度/头数/eps/theta），不占权重存储空间 |
| `head_dim` | 一个注意力头占多少维 = `embedding_length / head_count` |
| `GQA` | 多查询注意力：query 头多、KV 头少，一组 query 头共用一个 KV 头 |
| `RMSNorm` | 只除以均方根再乘可学习权重，没有均值项、没有 bias |
| `RoPE` | 旋转位置编码：按位置把每对相邻维度转一个角度 |
| `SwiGLU` | 双通道 FFN：`silu(gate) * up` 再投影回去 |
| `tied embeddings` | 输出投影复用输入嵌入表，文件里不单独存 `output.weight` |
| `ne0` 最快维在前 | GGUF/GGML 张量布局：ne0 是行内连续维（输入维），ne1 是行数（输出维） |

### §2 正常流程

```text
[解析 GGUF 头] -> [按架构前缀读超参] -> [读嵌入表 + 每层 9 权重 + 输出头]
                       |                          +--> [校验：缺失/尺寸/量化 -> 抛异常]
                       +---------------------------------> [逐位置前向：注意力 -> FFN] -> [最终 RMSNorm + 词表投影]
```

1. 从 GGUF 头读 `general.architecture` 得到架构前缀（`llama` / `qwen3` / …），拼到每个超参 key 的后缀前面 —— 【为什么】上游用 `LLM_KV_NAMES` 的 `%s.` 占位 + `LLM_KV::operator()`（llama.cpp @ 88b47a755 的 src/llama-arch.cpp:873）做同一件事：不同架构的 key 后缀相同、前缀不同。
2. 按前缀读形状参数（`block_count` / `embedding_length` / `attention.head_count` / `attention.head_count_kv` / `feed_forward_length` / `context_length` / `attention.layer_norm_rms_epsilon` / `rope.freq_base`），并校验：四个必填项任一为 0 报错；`embedding_length` 必须能被 `head_count` 整除（否则 head_dim 不是整数）。
3. 读 `token_embd.weight`，用它的 ne1 反推词表大小；再逐层读 9 个张量（`blk.N.attn_norm/attn_q/attn_k/attn_v/attn_output/ffn_norm/ffn_gate/ffn_up/ffn_down`）：norm 展开成 F32 向量，矩阵保留原始类型。
4. 逐张量校验 ne0/ne1 与超参一致 —— 【为什么】加载期是唯一同时知道「超参」和「张量形状」的地方，在这里报错能带层号和张量名。
5. 读输出头：有 `output.weight` 就独立加载，没有就与嵌入表绑定（tied embeddings）；然后逐位置推进前向：查嵌入行 → 每层跑注意力子层和 FFN 子层 → 最终 RMSNorm + 词表投影。

### §3 异常处理

| # | 异常 | 如何处理 |
|---|---|---|
| 1 | GGUF 文件本身损坏（魔数/版本/越界） | 不可重试；单元 1 的解析器抛异常，模型对象从未构造，CLI 退出码 1 |
| 2 | 超参缺失或架构前缀取错 | 不可重试；抛「模型超参数不完整（需要 <前缀>block_count/embedding_length/attention.head_count/feed_forward_length）」；模型对象未构造 |
| 3 | 张量缺失 | 不可重试；抛「模型缺少张量: <名字>」，名字里带 `blk.N.` 层号，便于定位是哪一层 |
| 4 | 张量尺寸与超参不符（含 ne0 不是量化块大小整数倍、output 尺寸与词表不符） | 不可重试；加载期抛，错误信息带层号、张量名与期望/实际 ne0/ne1；模型对象未构造，不留半加载状态 |
| 5 | 权重是量化类型（Q4_K 等） | 不可重试（单元 5 才支持）；抛「张量 <名字> 是量化类型 Q4_K，FP32 前向不支持（单元 5 实现量化推理）」；模型对象未构造 |
| 6 | 前向时 token id 越界 | 不可重试；抛「token id 越界」；不写 KV cache，残差流未产生 |
| 7 | 前向时 `kv.size() != pos` | 不可重试；抛「forward_token: 要求 kv.size() == pos（顺序生成）」；不写 KV cache，缓存保持原状 |

### §4 Out of Scope

| 砍掉的能力 | 为什么可以砍 | 承重判断（详见 §16.3） |
|---|---|---|
| GGML 工程架构（计算图 + 线程池/SIMD/缓存分块） | 这是性能/工程架构，不是数学；直接循环更适合教学 | 第 1、4 行：可砍 |
| KV cache 的完整形态（多实现、slot 复用、seq_id） | 刻意推迟到单元 4，用它讲「算法不变式驱动的优化」 | 第 2 行：可砍（推迟到单元 4） |
| 几十种架构变体（Gemma q-norm、MoE、注意力 sink、RWKV 循环结构等） | 每个变体都是局部的数学替换，主干掌握后都是查表工作 | 第 3 行：可砍 |
| 长上下文与工程扩展（`rope_scaling`、logit soft cap、掩码类型、`mlock` / NUMA / 张量并行） | 7B 以内用不到；且不影响数值 | 第 5、6 行：可砍 |
| RoPE 配对约定的「可切换」（NORM / NEOX 二选一开关） | 配对约定是模型语义本体，切换等于换模型 | 第 7 行：**承重不可砍**（只固定 NORM） |

### §5 观测点

| 时机 | 打的键、携带的字段 |
|---|---|
| CLI 加载完成 | `层数` / `维度` / `头`（格式 `nh/nkv`）/ `词表` —— 见 src/main.cpp |
| 单元 3 合成用例 | 合成用例 4 组全绿：算子数学（RMSNorm 手算值 / f16 抽查 / RoPE 恒等+保模长 / softmax 归一）、前向形状与确定性、因果性逐位验证（改未来 token 不影响过去 logits）、F16+tied+双层、量化拒绝 |
| 真实模型冒烟 | qwen3 1.7B Q4_K_M 正确报「量化类型 Q4_K，FP32 前向不支持（单元 5 实现量化推理）」 |
| ctest | `model` / `generate` / `quant` 三个单元测试全绿 |

## 第二部分：详细设计

### §6 整体流程图

```text
[子流程 1 模型加载] -> [子流程 2 注意力子层] -> [子流程 3 FFN 子层] -> [子流程 4 输出头]
       | 失败出口            | 注意力输出 + 残差      | 残差流                  | logits
       v
  [抛异常，模型未构造]
```

失败出口的形状：子流程 1 的任何一条失败（§3 异常 1~5）都在「模型对象构造完成」之前抛出，流程终止在子流程 1 内部；子流程 2~4 的失败（§3 异常 6~7）在单次前向内抛出，缓存不前进。

### §7 子流程 + 业务说明

#### 7.1 模型加载

`[读架构前缀] -> [读超参 + 形状自洽校验] -> [读嵌入表] -> [逐层读 9 张量并校验] -> [读输出头 / 绑定 tied]`

1. 看 `general.architecture` 拿前缀（本单元只支持 LLaMA 主干系 `llama` / `qwen3`，因为 Qwen2/3 派生自 LLaMA、张量布局与主干完全一致），再按前缀拼出每个超参 key 的完整名字去查；查不到取缺省值（`n_head_kv` 缺 `n_head`、`n_ctx` 缺 4096、`rms_eps` 缺 1e-5、`rope_theta` 缺 1e4）；最后检查四个必填项和整除关系，任何一条不满足立刻停下。
2. 矩阵类张量走 `load_tensor_buf`（保留原始类型与字节），norm 类小向量走 `load_tensor_f32`：F32 原样拷、F16 逐元素转 F32、其余类型报「量化类型不支持」；逐层读 9 个张量并立即校验形状，量化张量额外要求 ne0 是量化块大小的整数倍，否则反量化会越界。
3. 输出头：有 `output.weight` 用它，没有就把 `token_embd` 复制一份共享并置 `tied_output = true`。

#### 7.2 注意力子层（每层一次）

`x -> [RMSNorm] -> [Q/K/V 投影] -> [RoPE 逐头旋转] -> [写缓存 / 读历史 K/V] -> [GQA 打分 + softmax + 加权] -> [输出投影 + 残差累加]`

1. 对残差流做 RMSNorm（`out = x / sqrt(mean(x²) + eps) * weight`）—— 【为什么】LLaMA 系没有 bias 也没有减去均值；再用同一份归一化结果投影出 query、key、value 三条向量。
2. 对 query 的每个头、key 的每个头**分别**做 RoPE，作用范围严格限定在一个头内（`head_dim` 维）—— 【为什么】跨头配对会让位置信息串到别的头上，数学上直接错。
3. 把本层的 key/value 追加进 KV cache；打分时只读 0..pos 的历史，实现因果性 —— 【为什么】位置 p 的输出只允许看到 0..p；query 头数 `nh` 比 KV 头数 `nkv` 多时，query 头 `hq` 用 KV 头 `hq / (nh / nkv)`（按需索引共享，不物理复制 KV 头）。
4. 每个 query 头的分数向量做数值稳定 softmax（先减最大值），再对 value 加权求和；所有 query 头拼起来过输出投影矩阵，投影结果累加回残差流。

#### 7.3 FFN 子层（SwiGLU）

`x -> [RMSNorm] -> [gate 投影] + [up 投影] -> [silu(gate) * up] -> [down 投影] -> [残差累加]`

1. 先做一次 FFN 前的 RMSNorm，再用两个矩阵分别算出门控通道和上通道 —— 【为什么】这是 LLaMA 相对原始 Transformer ReLU FFN 的主要结构差异；门控通道过 SiLU（`x * sigmoid(x)`）后逐元素乘以上通道，再过 down 投影回到残差维度并累加回残差流。

#### 7.4 输出头

1. 所有层跑完后，对最终残差流做最后一次 RMSNorm，再用输出矩阵投影到词表维度得到该位置的 logits（长度 = 词表大小）；tied 模式下这个矩阵就是嵌入表。

### §8 异常流转表

| 异常 | 触发点 | 状态/字段变化 | 恢复路径 |
|---|---|---|---|
| GGUF 文件损坏 | 单元 1 的解析器读魔数/版本/偏移时 | 模型对象从未构造；没有任何可前向的状态 | 不可恢复：换一份合法 GGUF 重来 |
| 超参不完整或整除关系不成立 | `Model::load` 读完超参后立刻校验 | `Model` 对象半构造随即被析构，权重一个字节都没读 | 不可恢复：换一份 key 齐全的 GGUF 重来 |
| 张量缺失 | `load_tensor_f32` / `load_tensor_buf` 在张量表里找不到名字 | 已读入的层权重随对象析构释放；层号可从张量名 `blk.N.` 读出 | 不可恢复：文件与本实现支持的架构不匹配 |
| 张量尺寸与超参不符 | `Model::load` 里的形状校验（`expect` lambda）；量化张量另查 ne0 是否为块大小整数倍；输出头另查与词表是否一致 | 报错信息带层号、张量名、期望与实际尺寸；模型对象未完成构造 | 不可恢复：文件与超参自相矛盾 |
| 权重是量化类型而前向是 FP32 | `load_tensor_f32` 遇到 F32/F16 之外的类型 | 报错指向单元 5；模型对象未完成构造 | 不可恢复（本期不支持）：等单元 5 打开量化路径 |
| 前向时 token id 越界 | `forward_token` 入口检查 token 是否在词表范围内 | 不写 KV cache，缓存 `size` 不变，残差流未产生 | 不可恢复：调用方传入非法 token |
| 前向时 `kv.size() != pos` | `forward_token` 入口检查顺序生成前置条件 | 不写 KV cache，缓存 `size` 不变，残差流未产生 | 不可恢复：调用方必须按 pos 递增顺序调用 |

### §9 数据形状

```jsonc
// HParams（src/model.h）—— 对应 llama.cpp @ 88b47a755 的 llama_hparams
// n_layer/n_embd/n_head/n_ff 必填；n_head_kv 缺省 = n_head；n_ctx 缺省 4096；n_vocab 由嵌入表 ne1 推出
{ "n_layer": 0, "n_embd": 0, "n_head": 0, "n_head_kv": 0, "n_ff": 0,   // u32 <前缀>.*
  "n_ctx": 4096, "n_vocab": 0, "rms_eps": 1e-5, "rope_theta": 1e4 }    // u32 / u32 / f32
// TensorBuf（src/model.h）—— GGUF/GGML 的「最快维在前」；W(i,o) 存在 data[i + o*ne0]
{ "type": "GgmlType::F32",   // F32 / F16 / Q4_K / Q5_K / Q6_K / Q8_0
  "ne0": 0, "ne1": 0, "bytes": [] }   // u32 输入维 / u32 输出维 / u8[] 原始字节（F32 时 ne0*4*ne1）
// 每层 9 个权重 + 3 个全局张量（d = n_embd, kvd = kv_dim, ff = n_ff）；命名对应上游 LLM_TENSOR_NAMES（src/llama-arch.cpp:377）的 blk.%d.*
{ "blk.N.attn_norm.weight": {"ne": [d]},     "blk.N.attn_q.weight": {"ne": [d, d]},
  "blk.N.attn_k.weight": {"ne": [d, kvd]},   "blk.N.attn_v.weight": {"ne": [d, kvd]},
  "blk.N.attn_output.weight": {"ne": [d, d]}, "blk.N.ffn_norm.weight": {"ne": [d]},
  "blk.N.ffn_gate.weight": {"ne": [d, ff]},  "blk.N.ffn_up.weight": {"ne": [d, ff]},
  "blk.N.ffn_down.weight": {"ne": [ff, d]},  "token_embd.weight": {"ne": [d, n_vocab]},
  "output_norm.weight": {"ne": [d]}, "output.weight": {"ne": [d, n_vocab]} } // output 缺省时与 token_embd 绑定
// KVCache 布局（src/kvcache.h）：k_[((layer*cap) + pos) * dim_ + i]，v_ 同构
{ "n_layer": 0, "kv_dim": 0, "cap": 0, "size": 0 }  // u32/u32/u64/u64；size 必须等于下一次调用的 pos
```

## 开发流水线扩展节

### §13 编排锚点与挂载点

| 角色 | 路径 | 承载什么 |
|---|---|---|
| 加载期编排 | src/model.cpp `Model::load` | 超参 → 嵌入表 → 逐层权重 → 输出头 → 尺寸校验，全流程唯一入口 |
| 运行期编排 | src/model.cpp `Model::forward` → src/model.cpp `Model::forward_token` | 逐位置推进；`forward` 只是「临时缓存 + 逐位置收集」的外壳，计算全部在 `forward_token`；单元 3/4/5 共用这一条路径 |
| 算子层 | src/model.cpp/27/50/125/143/149 | rmsnorm / f16_to_f32 / matvec_f32 / rope_rotate / silu / softmax，独立成函数便于单测 |

**删掉锚点流程还跑不跑得起来**：`forward_token` 是唯一的计算路径，删掉它 `forward`、CLI、单元 3/4/5 的全部测试都编译不过，流程立刻断；`rope_rotate` 同样如此，它把「配对约定」这一语义收在一个函数里，是全文最容易写错也最不该散开的地方。

挂载点表：

| 文件:行 | 改什么 | 为什么必须在这儿 |
|---|---|---|
| src/model.cpp `Model::load` / :231-235 `prefix` + `key()` lambda；src/model.h / :68 `HParams` / `TensorBuf`；src/gguf.h `GgufFile` | 架构前缀、超参 key、缺省值；超参字段与类型化缓冲；元数据与张量的只读入口 | GGUF 的键名映射只在这里做一次（前缀参数化复刻了上游 `LLM_KV::operator()` 的「后缀相同、前缀随架构变」结构）；`HParams` / `TensorBuf` 是上游 `llama_hparams` / `llama_tensor` 的对应物，单元 5 量化反量化复用同一份形状契约；`GgufFile` 是单元 1 的只读契约 |
| src/model.cpp `expect` lambda | 张量形状校验 | 加载期是唯一同时握有超参和张量形状的时机 |
| src/model.cpp `Model::forward_token` | 注意力 + FFN 的全部数学 | 单元 3/4/5 共用这一条路径，等价性测试钉在这里 |
| src/model.cpp `rope_rotate` | RoPE 配对约定与频率 | 约定写死在单个函数里；改它等于改全模型的数值语义 |
| src/kvcache.h `KVCache` | K/V 的追加与按 (层, 位置) 读取 | 单元 4 增量路径的存储契约；本单元作为「历史 K/V 只读」的消费方 |

### §14 概要→详细映射表

| 概要条款 | 详细落点 | 参考实现对应 |
|---|---|---|
| §1.1 `HParams` / `LayerWeights` / `TensorBuf` | §7.1 步骤 1-3、§9 | `LLM_KV_NAMES` 的 `%s.` 前缀占位 + `LLM_KV::operator()` 按架构名替换；`LLM_TENSOR_NAMES` 的 `blk.%d.*` 命名；`ggml_tensor` 的 ne[]/type/数据指针三件套 |
| §1.3 `GQA` | §7.2 步骤 3 | `llama_hparams::n_gqa` 给出组数，组内头共享由 mul_mat 的 `r2 = ne12 / ne02` 索引广播实现 |
| §2 步骤 1-4 | §7.1 步骤 1-3 | `llama_model_loader` 的 `get_key` 按架构查元数据（键名表在 `LLM_KV_NAMES`）+ 加载期统一形状校验，缺张量即抛、错误信息带张量名 |
| §2 步骤 5 | §7.2-§7.4 | 计算图节点：norm → 投影 → rope → 注意力 → FFN → 输出投影 |
| §2 因果性 | §7.2 步骤 3 | 只读 0..pos 的缓存槽位，等价于图上的 causal mask |
| §3 异常 1 | §8 第 1 行 | GGUF 解析层的边界检查与异常 |
| §3 异常 2 | §8 第 2 行 | 必填 key 缺失即报「key not found in model」 |
| §3 异常 3 | §8 第 3 行 | 缺张量即报，名字带层号 |
| §3 异常 4 | §8 第 4 行 | `llama_model_loader::check_tensor_dims` 的形状比对，错误信息含期望与实际 |
| §3 异常 5 | §8 第 5 行 | 本实现按类型分发，量化类型在单元 3 明确拒绝并指向单元 5 |
| §3 异常 6 | §8 第 6 行 | 前向入口的 token 范围检查 |
| §3 异常 7 | §8 第 7 行 | 增量路径要求顺序生成，`kv.size()` 与 `pos` 必须相等 |
| §5 观测点 | §5 | 加载期摘要一行 + 单元 3 合成用例 + 真实模型冒烟 + ctest |

### §15 裁量区与待拍板

| AI 可以自决 | 必须对齐参考实现 | 必须问人 |
|---|---|---|
| 教学便利代码的实现方式：`tests/gguf_writer.h` 的字节流写入器、`tests/test_model.cpp:81` 的微型模型构造器、`tests/arbiter.py` 的独立仲裁者怎么写 | 术语（名字）：hparams / head_dim / GQA / RMSNorm / RoPE / SwiGLU / tied embeddings / ne0 最快维在前 | 砍掉承重件：GGML 计算图、KV cache、架构变体、线程池与 SIMD、rope_scaling、mlock 与 NUMA 这六项的砍掉结论 |
| 测试夹具的形状选择（n_embd=8、head=2、kv_head=1、ff=16、词表=24） | 结构档：分层（HParams / LayerWeights / TensorBuf / Model）+ 字段职责 | 新增参考实现没有的概念：本单元无新增（`TensorBuf`、`KVCache` 是上游 `llama_tensor` / `llama_memory_i` 的教学替身） |
| 教学注释的详略与排版 | 结构档：本单元覆盖到的调用链骨架（load → forward → forward_token → 每层两段） | 改对齐粒度定义：把 §16.1 的「对齐到什么粒度」从字段职责提升到逐位一致 |
| ASCII 流程图的框数与措辞 | 实现档的替换（判据：能力不减 + 以后接得上）：计算图→直接循环、ggml 分块 SIMD→三层循环、K-V 物理复制→按需索引、F16 位转换→逐元素转换 | RoPE 配对约定固定为 NORM 是否需要向上游提 issue（上游本身不提供切换） |

## 还原模式扩展节

### §16 对齐面

> 参考实现：llama.cpp @ 88b47a755
> 本文只对「本单元涉及的部分」负责。

#### 16.1 概念对齐

| 参考实现的术语 | 本实现的对应物 | 对齐到什么粒度 | 备注 |
|---|---|---|---|
| `hparams` / `llama_hparams` | `HParams`（src/model.h） | 字段职责 | 与上游一一对应；上游有 per-layer 数组，本实现只有单层值 |
| `n_head` / `n_head_kv` / `n_gqa` / `head_dim` | `HParams::n_head` / `n_head_kv`，组数现算 `nh / nkv`（src/model.cpp），头宽 `HParams::head_dim()`（src/model.h） | 字段职责 + 机制 | 上游 `llama_hparams::n_head(il)` / `n_head_kv(il)` / `n_gqa(il)`（src/llama-hparams.cpp:50/58/74）返回同样的量；`head_dim` 定义同为 `embedding_length / head_count`，上游另有 `n_embd_head_v`（src/llama-hparams.cpp:123）区分 K 头与 V 头 |
| `RMSNorm` / `LLM_NORM_RMS` | `rmsnorm()`（src/model.cpp） | 机制 + 调用链骨架 | 对应 ggml 的 `ggml_compute_forward_rms_norm`（ggml/src/ggml-cpu/ops.cpp:3860） |
| `RoPE` / `llama_rope_type` | `rope_rotate()`（src/model.cpp） | 机制（配对约定逐位一致） | **相邻配对 (2i, 2i+1)**，对应上游 `LLAMA_ROPE_TYPE_NORM`（src/llama-model.cpp:2494，架构分支在 2457 行，注释在 2456 行「operating on pairs of consecutive head values」），**不是** `LLAMA_ROPE_TYPE_NEOX`（半切配对，Falcon/GPT-NeoX 家族用） |
| `GQA` | `hkv = hq / q_per_kv`（src/model.cpp） | 机制 | 按需索引共享 KV 头；上游同样不物化复制（见 §16.2） |
| `SwiGLU` / `LLM_FFN_SILU` | `silu()`（src/model.cpp）+ gate/up/down 三次投影 | 机制 | 上游 `llm_graph_context::build_ffn`（src/llama-graph.cpp:1564）的 gate/up 分支，数学等价 |
| `tied embeddings` | `Model::tied_output_`（src/model.h） | 字段职责 | 缺 `output.weight` 时输出投影与 `token_embd` 共享 |
| `ne0` 最快维在前 / `ggml_tensor::ne[]` | `TensorBuf::ne0/ne1`（src/model.h） | 字段职责 + 布局 | `W(i,o)` 存在 `data[i + o*ne0]`，即 `y[o] = Σ x[i]·W(i,o)`；`TensorBuf` 是 `llama_tensor` 的结构档替身，缺 nb[]/shape 等 stride 元数据 |
| `llama_memory_i` / KV cache cell | `KVCache`（src/kvcache.h） | 结构档：调用链骨架 | 上游接口在 src/llama-memory.h:73；单元 4 交付，本单元作为「历史 K/V 只读」的消费方 |

#### 16.2 设计对齐

「结构档的偏离」表：

| 参考实现的机制 | 本实现的位置 | 偏离点 | 为什么可以偏离 |
|---|---|---|---|
| `llama_model_rope_type` 按架构分发 rope 类型（src/llama-model.cpp:2430） | src/model.cpp `rope_rotate` | 概念偏离：只实现 LLaMA 系的 `LLAMA_ROPE_TYPE_NORM` 一种 | LLaMA 主干与派生架构（qwen3）共用；引入 NEOX/MROPE 只会重演 §17.3 的约定错配 |
| `LLM_KV::operator()` 的架构前缀映射（src/llama-arch.cpp:873） | src/model.cpp `prefix` + `key()` lambda | 细节简化：手写前缀拼接，不做 map 查表 | 名字空间小；但「后缀相同、前缀随架构变」这一结构被完整复刻 |
| `llama_model_loader::check_tensor_dims` 在加载期比对形状 | src/model.cpp `expect` lambda | 无偏离 | 时机与信息量一致：都在加载期、错误信息都带张量名与期望/实际 |
| `llama_kv_cache::get_k` 返回 n_head_kv 头的 view，注意力阶段按批次广播 | src/model.cpp `hkv = hq / q_per_kv` | 无偏离 | 上游同样不物化复制：ggml 的 mul_mat 用 `r2 = ne12 / ne02`、`i02 = i12 / r2` 做索引广播（ggml/src/ggml-cpu/ggml-cpu.c:1185）；旧文档曾写作「真实版物化复制 KV」，在 88b47a755 上核对后更正 |
| GGML 计算图（先建图后执行、图可复用） | src/model.cpp 直接循环 | 机制替换：计算图 → 直接循环 | 数学等价；直接循环更适合教学（读代码 = 读数学） |

「实现档的替换」表：

| 参考实现用什么 | 最小版用什么 | 能力是否等价 |
|---|---|---|
| 容器：ggml 张量 + 计算图（`ggml_tensor` / `ggml_context`） | `std::vector<float>` + `TensorBuf` 字节缓冲（src/model.h） | 能力不减：类型、形状、字节都在；以后接得上：给 `TensorBuf` 补 `nb[]` 即可映射 `ggml_tensor` |
| 算法：`ggml_compute_forward_mul_mat` 的分块 + SIMD + 线程池（ggml-cpu.c:1254），内层用 `ggml_vec_dot_f32`（vec.cpp:11） | `matvec` 三层循环（输出行 → 元素累加，src/model.cpp）与 `matvec_f32` 朴素累加（src/model.cpp） | 能力不减：同为 `y[o] = Σ x[i]·W(i,o)`，结果逐位一致到浮点误差；以后接得上：内层换成一次 `ggml_mul_mat` 调用即可 |
| 容器/算法：K-V 物理复制或 ggml repeat 广播 | 按需索引 `hkv = hq / q_per_kv`（src/model.cpp） | 能力不减：头共享语义完全一致，且省内存；以后接得上：换成上游的 view 广播即可 |
| 类型转换：`ggml_fp16_to_fp32_row` 批量逐元素转换（ggml/src/ggml.c:468） | `f16_to_f32` 逐元素位转换（src/model.cpp） | 能力不减：正规数/次正规数/inf-nan 三分支全覆盖；以后接得上：换成 `ggml_fp16_to_fp32` 即可（本实现是手写位运算，不能用硬件 F16C 指令的简化写法，见 §17.1） |
| 错误处理：ggml 的 `GGML_ASSERT` / `GGML_ABORT`；并发：ggml 线程池；日志：`LLAMA_LOG_INFO` 打印整套 hparams | `std::runtime_error` + 消息带层号与张量名；单线程；CLI 打印一行摘要（src/main.cpp） | 能力不减：不可恢复错误同样立即中止、数值结果不变、层数/维度/头数/词表都在；接得上：消息格式可对齐，并行化在 `matvec` 一层做即可，摘要可扩成完整日志 |

#### 16.3 砍掉清单的承重判断

| 砍掉什么 | 真实版怎么做 | 概念测试 | 行为测试 | 术语测试 | 结论 |
|---|---|---|---|---|---|
| GGML 计算图 | 先建图后执行，图可复用（`llm_graph_context`） | 直接循环的段落与图节点一一对应：norm、QKV 投影、rope、注意力、FFN、输出投影各一段，可逐段对公式 | 合成用例 logits 逐位确定；因果性逐位验证通过 | 真实版用「计算图 / 节点 / 算子」，本实现用「子层 / 步骤 / 算子函数」，边界在 §16.2 已声明 | 可砍 |
| KV cache（完整形态：多实现、slot 复用、seq_id、随机插入） | `llama_memory_i`（src/llama-memory.h:73）多个实现 + slot 调度（src/llama-kv-cache.cpp） | 注意力只需要「历史 K/V 只读」这一条契约，`KVCache` 提供的就是它 | 单元 4 的「增量 vs 全量逐位一致」测试（tests/test_generate.cpp:134）已把等价性钉死 —— 单元 3 的教学形态本是全量 O(n²) 注意力（每层物化整段序列，位置 p 只可见 0..p），重构后两条路径共用同一个 `forward_token` | 沿用上游 `KV cache cell` / `llama_memory_i` 的称呼，本实现类名 KVCache | 可砍（刻意推迟到单元 4） |
| 几十种架构变体（Gemma q-norm、Qwen2MoE、注意力 sink、RWKV 循环结构等） | `LLM_ARCH_*` 每种一套图（src/llama-arch.cpp 的 `LLM_ARCH_NAMES`） | 只覆盖 LLaMA 主干；Qwen2/3 派生自 LLaMA，主干数学一致 | 真实模型冒烟跑到「量化不支持」这一步，说明架构前缀与张量名已对上 | 变体名保留上游叫法（Gemma q-norm / MoE / 注意力 sink） | 可砍 |
| 线程池 / SIMD / 缓存分块 | ggml 分块 + SIMD + 多线程（ggml-cpu.c 的 mul_mat 分块） | 并行只改执行顺序不改数学，本实现把并行性完全剥离 | 单线程下 logits 逐位确定；并行化需另做一致性测试 | 保留上游「线程池 / 分块 / SIMD」词 | 可砍 |
| `rope_scaling`（YaRN / NTK 外推）、logit soft cap、各种掩码类型 | `LLAMA_ROPE_SCALING_TYPES`（src/llama-model.cpp:846）+ `ggml_rope_yarn_corr_dims`（ggml/src/ggml-cpu/ops.cpp:6011） | 本实现只支持 base theta 的线性频率 `theta^(-2i/h)`，扩展频率不改主干 | RoPE 恒等 + 保模长 + 相邻配对数值三测通过 | 用上游 `rope_scaling` 一词指代被砍部分 | 可砍 |
| 权重 `mlock` / NUMA 亲和 / 张量并行 | 驻留控制与调度策略（llama.cpp 的 mmap/mlock/NUMA 亲和） | 只影响驻留与调度，不进入数值路径 | 加载期校验 + 前向逐位确定，说明与驻留无关 | 保留上游 `mlock` / `NUMA` / `tensor parallel` 词 | 可砍 |
| RoPE 配对约定的「可切换」（NORM / NEOX 二选一开关） | 按架构固定一种：LLaMA 系 `LLAMA_ROPE_TYPE_NORM`（src/llama-model.cpp:2494），Falcon 系 `LLAMA_ROPE_TYPE_NEOX`（同文件 2496 行注释：pairs offset by n_rot/2） | 配对约定是模型语义本体：两种约定给出不同的旋转矩阵，注意力分数随之不同，切换等于换模型 | 用独立仲裁者做模式扫描（恒等 / 相邻对 / 半切）才能分辨，说明「可切换」必然带来不可判定的结果 | 用上游 `LLAMA_ROPE_TYPE_NORM` / `LLAMA_ROPE_TYPE_NEOX` 两个术语 | 承重不可砍：只固定 NORM，且不提供切换开关 |

#### 16.4 教学便利代码清单

| 多出来的 | 在哪 | 为什么需要 | 谁 import 它 |
|---|---|---|---|
| GGUF 字节流写入器 `minitest::Writer` | tests/gguf_writer.h:13 | 与 src/gguf.cpp 的解析器互为逆运算，把 GGUF 格式契约变成可执行文档；单元 3 的微型模型靠它落盘 | tests/test_gguf.cpp、tests/test_tokenizer.cpp、tests/test_model.cpp、tests/test_generate.cpp、tests/test_quant.cpp |
| 微型模型构造器 `TinyModelBuilder` | tests/test_model.cpp:81 | 真实版要下几个 GB 权重才能测；这里现场合成 1~2 层微型 LLaMA（n_embd=8、head=2、kv_head=1、ff=16、词表=24），权重用确定性伪随机 | tests/test_model.cpp |
| 独立仲裁者（numpy，float64 累加） | tests/arbiter.py:1 | 从 GGUF 原始字节手写解析 + 前向，零复用本仓库任何代码；用来裁决 ours-vs-llama.cpp 的发散归属，RoPE 配对约定就是靠它做模式扫描定性的 | 无（命令行人工运行，不 import 本仓库模块） |
| CLI 入口 | src/main.cpp | 真实版是 examples/ 下几十个程序；这里一个 main 覆盖「加载 → 编码 → prefill → 采样 → 生成」，是模型能力的可见出口 | （独立可执行文件） |
| Unicode 区间表生成器 | tools/gen_unicode_tables.py | 上游 unicode-data.cpp（164K）同样由脚本生成；本仓库用 Python 标准库现场生成区间表 | 构建前手动执行（属单元 2 范围，在此登记在册） |

### §17 踩坑记录

### 17.1 F16 次正规数被截断成 0 【坑型：平台或类型差异】
**症状**：F16 权重转 F32 后，最小的一批非零值全部变成 0；正规数（如 0x3C00 = 1.0）转换正确，所以大部分测试都绿，只有抽查极小值时才发现。
**根因**：最初的写法把浮点值的截断结果当位模式用 —— `sign | (uint32_t)(float_val)`。F16 的次正规数（指数域为 0）值是 `mant * 2^-24`，量级在 6e-8 附近，转换成 float 后其位模式的下半段被截断，整数化后恒为 0，整条次正规数分支退化成「所有次正规数 → 0」。
**修法**：按 IEEE 754 的三个分支分别处理（src/model.cpp `f16_to_f32`）：指数域为 0 时用 `mant * (1 / 2^24)` 算出真实数值再拼符号位；指数域为 31 时拼 inf/nan；其余情况按 `exp + (127 - 15)` 左移 23 位、尾数左移 13 位直接拼位模式。测试里补了 `f16_to_f32(0x0001) == 5.9604645e-8` 专门盯这条分支（tests/test_model.cpp:186）。
**真实版**：上游走 `ggml_fp16_to_fp32`（ggml/src/ggml.c:448），用编译器的 F16C 硬件转换或 `GGML_FP16_TO_FP32` 宏，次正规数由硬件/宏保证正确。本质相同 —— 都是「按指数域分三支」，差别只是我们手写了硬件替身。
**教学点**：位操作和数值运算混在一行时，「看起来对」的样本不能代表全分支正确。只要涉及 IEEE 754 的特殊区域（次正规数、inf、nan、负零），就必须逐分支测，否则被吞掉的永远是最小的那批值。

### 17.2 RoPE 跨头配对，把 32 个头旋成了一团 【坑型：概念层级错位】
**症状**：注意力输出明显不对，但所有单 token 测试都通过；单测日志一片绿。
**根因**：最初把 query 的 2048 维当作一整块做半切旋转（配对 `i` 与 `i+1024`），一次跨过了 32 个头。RoPE 的作用范围是**一个头内部**（`head_dim` = 64 维），跨头配对等于把位置信息串到别的头上。更隐蔽的是：pos = 0 时 RoPE 是恒等变换（旋转角为 0），所以单元 3/4 的全部单 token 测试都漏检了它。
**修法**：按 `head_dim` 逐头切段调用旋转 —— query 有 `nh` 段、key 有 `nkv` 段（src/model.cpp），每段长度严格等于 `head_dim`；并补一条 pos = 7 的保模长测试和一条 pos = 1 的相邻配对数值测试（tests/test_model.cpp:205-224）。
**真实版**：上游 `ggml_compute_forward_rope_flt`（ggml/src/ggml-cpu/ops.cpp:5953）按 `attn-heads` 维度逐头循环，head 内再按 `rotate_pairs` 的 offset 配对（ggml/src/ggml-cpu/ops.cpp:5934），语义与本实现一致。
**教学点**：**任何位置相关算子用 pos = 0 测等于没测** —— 恒等变换会让一整类错误隐身。凡是「按位置/按序号改变结果」的算子，测试里必须至少有一条 pos != 0 的用例。

### 17.3 RoPE 配对约定与上游 master 不一致（排查最久的一条） 🔴 【坑型：混合】
**症状**：与上游逐数值对照时 logits 发散；但本地实现自洽 —— 用同一个实现自比永远对得上，看不出任何异常。
**根因**：单 token 序列、以及同一 token 重复的序列（如 `[1576,1576]`）下，任意配对约定都给出相同结果（恒等 / 同 token 的权重无关），只有「混合 token 的多位置」用例才能暴露差异。真正的错源是约定本身：本实现最初按 half-split（两半对旋）旋转，那是老教程和旧版 llama.cpp 的做法；上游 master（88b47a755）已把 LLaMA 系改成**相邻配对**（src/llama-model.cpp:2457 的 `case LLM_ARCH_LLAMA:` → 2494 行 `return LLAMA_ROPE_TYPE_NORM;`，注释在 2456 行「operating on pairs of consecutive head values」），而 half-split 对应 `LLAMA_ROPE_TYPE_NEOX`（2496 行注释「the pairs of head values are offset by n_rot/2」），只服务 Falcon / GPT-NeoX 家族。两种约定数学上不等价，除非对 `wq` / `wk` 做相应的行置换才能互换。
**修法**：三步定案 —— ① 用独立 numpy 仲裁者（从 GGUF 原始字节手写解析 + 前向，零复用本仓库代码）逐位比对，确认本实现自洽、问题在约定而非实现；② 让仲裁者做 RoPE 模式扫描（恒等 / 相邻对 / 半切），发现上游走的是相邻对；③ 查上游源码证实 `LLM_ARCH_LLAMA → LLAMA_ROPE_TYPE_NORM`。随后把 `rope_rotate` 改成相邻配对 (2i, 2i+1)，在 tests/test_model.cpp:215-224 钉死数值，并同步修掉 src/model.h 那段仍写 half-split 的过期注释（以实现的相邻配对 + 上游锚点为准）。
**真实版**：上游按架构分发 rope 类型（`llama_model_rope_type`，src/llama-model.cpp:2430），LLaMA 系走 `LLAMA_ROPE_TYPE_NORM`；ggml 侧的分支在 ggml/src/ggml-cpu/ops.cpp:6070 —— `GGML_ROPE_TYPE_NORMAL` 调 `rotate_pairs(n_dims, 1, ...)`（offset = 1，相邻对），`GGML_ROPE_TYPE_NEOX` 调 `rotate_pairs(n_dims, n_dims/2, ...)`（offset = 半切）。本质相同：上游把「约定」显式化成架构到枚举的映射，本实现把它固定成一种。
**教学点**：与上游逐数值对照前，先确认「上游当前版本的行为约定」—— 教程和旧源码里的「标准做法」可能已经变了；过期注释比没有注释更危险，因为它读起来像是有依据的。构造对照实验时，用例必须能区分各个假设（恒等 / 相邻对 / 半切要能分开），否则实验再严谨也只是自证。

### 17.4 qwen3 的超参 key 是 `qwen3.*` 不是 `llama.*` 【坑型：名字遮蔽】
**症状**：加载 qwen3 模型时报「模型超参数不完整」，但同一个文件在 llama.cpp 里能正常加载。
**根因**：超参 key 名带架构前缀 —— `llama.block_count` 与 `qwen3.block_count` 后缀相同、前缀不同（上游 `LLM_KV_NAMES` 的 `%s.` 占位 + `LLM_ARCH_NAMES` 的架构名，见 src/llama-arch.cpp:149 与 :8）。直接把 llama 的 key 名抄过来，等于用 `llama.*` 这组名字把文件里真实的 `qwen3.*` 遮住了：查得到的是空，查不到时又因为有缺省值而安静地取 0，最终以「超参不完整」的形式爆出来，而不是「key 不存在」。
**修法**：加载时先读 `general.architecture` 拿到架构名，再把它拼成前缀（src/model.cpp），后面所有超参都按 `prefix + suffix` 查。修复方式（前缀参数化）正好复刻了真实版的映射结构，而不是打补丁式地多查几组 key。
**真实版**：上游是 `LLM_KV::operator()`（src/llama-arch.cpp:873）用 `::format(LLM_KV_NAMES.at(kv), LLM_ARCH_NAMES.at(arch))` 做同样的前缀替换；架构名表在 src/llama-arch.cpp:8。本质相同：都是「后缀固定、前缀按架构解析」。
**教学点**：静默缺省值是错误信息的最大伪装 —— 「不报错但取 0」比「直接报错」难查十倍。凡是带缺省值的读取，缺省值必须能区分「合法缺省」和「查错了名字」，否则问题会被推迟到很远的地方才爆。

### 17.5 张量尺寸校验放在加载期 【坑型：不变式被打破】
**症状**：早期把形状检查放在前向循环里，结果是前向跑到一半越界，读到的是别人的内存，输出是一堆 NaN，没有任何可读的错误信息。
**根因**：形状与超参之间有一条不变式 —— `wq` 必须是 `{d, d}`、`wk` / `wv` 必须是 `{d, kvd}`、`ffn_down` 必须是 `{ff, d}`（kvd = `n_head_kv * head_dim`）。这条不变式只在「超参」和「张量形状」都在场时才能判定，而加载期正是两者同时在手的地方；把它推迟到前向，等于让错误漂移到运行期，而运行期已经写坏了一堆中间量。
**修法**：在 `Model::load` 里加一个 `expect` lambda，逐张量比对 ne0/ne1，不符就抛「张量尺寸不符: blk.0.attn_k 期望 {…} 实际 {…}」（src/model.cpp）。错误信息同时带层号和张量名 —— 只说「尺寸不符」等于没说。
**真实版**：上游在加载期做同一件事，`llama_model_loader::check_tensor_dims`（src/llama-model-loader.cpp:858）在读张量时比对期望维度，不符即报「tensor '%s' has wrong shape; expected %s, got %s」。本质相同：加载期校验 + 错误信息带张量名与期望/实际。
**教学点**：**宁可启动时报「张量尺寸不符: blk.0.attn_k」，也不要前向到一半段错误**。错误发生的层次要尽量靠近错误产生的地方；边界检查放在边界上，才不会让错误漂移到下游。
