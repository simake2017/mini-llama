# 技术文档：单元 4 KV Cache + 生成循环 + 采样 —— 从「会算」到「会说」

- 读者：想弄懂「为什么加了 KV cache 推理就快」以及「logits 怎么变成一个 token」的人，以及要对照 llama.cpp 源码核查约定差异的维护者
- 审法：先过 §16.1 的术语对齐，再按 §17 逐坑自检，最后用 §9 的数据形状核对缓存布局
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
| `KVCache`（src/kvcache.h） | 按 `[layer][pos][dim]` 存放的历史 K/V 缓冲 | 增量前向逐层 `append`（src/model.cpp） | 增量前向注意力段的 `k()` / `v()`（src/model.cpp / :576） | 构造参数任一为 0 抛「KVCache 参数不能为 0」；`size_ == 0` 表示还没推进过任何位置；`clear()` 只把 `size_` 归零，缓冲不释放 |
| `SamplerParams`（src/sampler.h） | 采样策略四元组：温度 / top-k / top-p / 种子 | CLI（src/main.cpp） | `Sampler` 构造 | `temp <= 0` 退化为贪心；`top_k <= 0` 不限数量；`top_p <= 0` 或 `>= 1` 不截断 |
| `Sampler`（src/sampler.h） | 把一段 logits 变成一个 token id | 构造（src/main.cpp） | 生成循环（src/main.cpp） | 内部持有 `std::mt19937 rng_`；`reset_seed`（src/sampler.h）可换种子；`sample` 遇空 logits 抛「sample: 空 logits」 |

#### 1.2 枚举与异常类别

本单元没有新增枚举：采样策略用四个标量字段表达（上游的 sampler 是「一节一个类」，本仓库压成固定顺序的一段函数）。异常类别共 7 处抛出点，全部是 `std::runtime_error`：

| 抛出点 | 消息 | 触发条件 |
|---|---|---|
| src/kvcache.cpp | KVCache 参数不能为 0 | 构造时层数 / kv_dim / 容量任一为 0 |
| src/kvcache.cpp | KVCache 已满（超过 context length） | `size_ >= cap_` |
| src/kvcache.cpp / :62 | KVCache 访问越界: k(pos) / v(pos) | `pos >= cap_` |
| src/model.cpp / :482 | token id 越界；forward_token: 要求 kv.size() == pos（顺序生成） | token id 不在 `[0, n_vocab)`；调用方乱序推进 |
| src/sampler.cpp | sample: 空 logits | 词表长度 `<= 0` |

#### 1.3 名词

| 名词 | 一句大白话 |
|---|---|
| KV cache | 缓存每个历史位置的 key / value，避免每生成一个 token 就重算全部历史 |
| cell / slot | 上游 KV 存储的最小单元（cell，有独立身份，不依附于它属于哪个位置）与「某条序列当前占用哪批 cell」的记录（slot），上游按 slot 分配与复用 cell |
| seq_id | 序列编号；上游一条 batch 里可以带多条 seq_id，各占各的 slot |
| prefill | 把 prompt 的所有 token 一次性推进 KV Cache，得到末位 logits |
| decode | 自回归阶段：每步采一个 token，把它前向后写回 KV Cache |
| greedy | 温度 <= 0 时的退化分支：直接取 argmax，完全不消耗随机数 |
| temperature | 把 logits 除以一个正数再 softmax；越大分布越平（多样性 ↑、稳定性 ↓） |
| top-k | 只保留概率最高的 k 个候选 |
| top-p（核采样） | 保留累积概率达到 p 的最小候选集合，**含把累计推过阈值的那一个** |
| kv_dim | 一个位置上的 K（或 V）的总维度 = `n_head_kv * head_dim` |

### §2 正常流程

```text
[构造 KV 缓存] -> [prefill: 逐位置推进] -> [采样] -> [新 token 前向] --+
                       |                     |                        |
                [每层写入本位置 K/V]   [按概率挑一个 token]           |
                [注意力读回 0..pos]  ->  回到采样  ------------------+
```

1. 构造 KV 缓存：一次分配「层数 × 容量 × kv_dim」两块 float 缓冲（K 一块、V 一块）。
2. prefill：对 prompt 的每个位置 p 调一次增量前向 —— 【为什么】位置 p 的注意力要看到 0..p 的全部历史 K/V，所以必须按位置递增推进，不能乱序。
3. 每一层做完 Q/K/V 投影与 RoPE 后，把本层的 K/V 追加进缓存；注意力段再从缓存里读回 0..pos 的历史 K/V —— 【为什么】**缓存里存的是「历史位置的 K/V」，不是「历史位置的 logits」**：位置 p 的输出只由「缓存里的历史 K/V」和「本位置自己的 K/V」决定，与历史位置当时怎么算出来的无关。这条结构性事实正是「增量 ≡ 全量」的论证基础（见 §7.2）。
4. 采样：按「贪心 → softmax（未加温度）→ top-k → top-p → 温度 → 抽样」产出下一个 token，再把它前向到位置 n（n = 已推进的位置数）、K/V 写回缓存 —— 【为什么】下一个 token 的注意力需要看到它；循环直到命中 EOS 或达到生成上限。

### §3 异常处理

| # | 异常 | 如何处理 |
|---|---|---|
| 1 | KV 缓存构造参数为 0 | 不可重试；抛「KVCache 参数不能为 0」，对象未构造、无内存分配 |
| 2 | 缓存写满（位置数超过 context length） | 不可重试；`append` 抛「KVCache 已满（超过 context length）」；已写入的历史位置保持不变，本位置未写入 |
| 3 | 缓存读取越界（pos ≥ 容量） | 不可重试；`k()` / `v()` 抛「KVCache 访问越界: k(pos) / v(pos)」；**校验用容量而非 size**（见 §7.1） |
| 4 | 调用方乱序推进（`kv.size() != pos`） | 不可重试；增量前向抛「forward_token: 要求 kv.size() == pos（顺序生成）」；缓存一个字节都不会被改 |
| 5 | token id 越界 | 不可重试；增量前向抛「token id 越界」；缓存不变 |
| 6 | 采样器收到空 logits | 不可重试；`sample` 抛「sample: 空 logits」；随机数状态不变（未消费） |

### §4 Out of Scope

| 砍掉的能力 | 为什么可以砍 | 承重判断（详见 §16.3） |
|---|---|---|
| **KV cache 的 slot / cell 复用与 seq_id 多序列** | 对话服务器场景的需求；单序列顺序生成用不到 | 第 1、2 行：**承重不可砍 → 降级还原**（降级为纯顺序追加，已登记于 `docs/constraints/04-对齐约束.md` §2.2） |
| 4 种 KV cache 实现（默认 / DSA / ISWA / dsv4） | 前 3 种是缓存压缩研究（attention sink、dramatic shift），非主干 | 第 3 行：可砍 |
| 采样链 10+ 节（min-p、重复惩罚、grammar 约束、mirostat 等） | 每节都是「变换 logits 或挑 token」的局部逻辑，主干四节覆盖主流用法 | 第 4 行：可砍 |
| `llama_batch` / ubatch 并行解码 | 吞吐优化，正确性无关 | 第 5 行：可砍 |
| 投机解码 | 加速特性，不进入数值语义 | 第 6 行：可砍 |
| 层卸载（ngl） | 显存特性，不进入数值语义 | 第 7 行：可砍 |

### §5 观测点

| 时机 | 打的键、携带的字段 |
|---|---|
| CLI 加载完成 | `模型: <路径> 层数=… 维度=… 头=… 词表=…`（src/main.cpp） |
| prefill 完成 | `prefill 完成: N tokens`（src/main.cpp） |
| 生成结束 | `生成 N tokens，耗时 Xs（Y tok/s）`（src/main.cpp） |
| 采样线观测 | 贪心下标 / top_k=1 下标 / 200 次采样的命中集合 / temp=2 时 token 3 是否被抽中（lines/numeric/sample_line.cpp 第 4~6 跳） |
| 验收记录 | 合成用例 3 组全绿：KV 等价性（含 clear 复用、顺序违规检测）、采样器（贪心 / top_k=1 / 统计尖峰 / 种子复现 / top_p）、生成循环（贪心可复现、id 合法）；ctest 4 个单元全绿（仓库后来扩到 7 个用例，当前 7/7 Passed）；`mini-llama` CLI 编译通过（端到端真实模型验证待单元 5 量化支持后用 qwen3 Q4_K_M 完成）；真实模型冒烟 `mini-llama --model models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf --prompt "Hi" --n 8` 正常生成 |

## 第二部分：详细设计

### §6 整体流程图

```text
[构造 KV 缓存] -> [prefill: 逐位置前向] -> [采样] -> [新 token 前向] --+
                        |                    |                         |
                  [KV 缓存写满]        [token id]                    |
     +---------------------------------------------------------------+
     |  命中 EOS 或达到上限时退出
```

失败出口的形状：所有异常都在「写缓存」之前或「写了一半」的位置抛出；其中 §3 的第 4、5 条抛在增量前向的入口，缓存一个字节都不会被改。**本单元的核心不变式**：增量前向（`forward_token`）与全量前向（`forward`）逐位置 logits 一致（1e-4 容差）—— 这不是一条普通测试，而是「缓存只是省掉重算、不改变语义」这条设计的**验收条件**：一旦不成立，说明缓存里存的东西或读取顺序出了错，后面的采样与生成全部失去意义。论证见 §7.2，测试见 tests/test_generate.cpp:136。

### §7 子流程 + 业务说明

#### 7.1 KV Cache 的存储与追加

`[构造: 分配 层数×容量×kv_dim 的 K 块与 V 块] -> [追加一个位置的一层] -> [按 (层, 位置) 读回]`

1. 布局是 `k_[((layer * cap) + pos) * dim + i]`（src/kvcache.h）—— 【为什么】按层分块，同一层的 K 在内存里连续，注意力内层循环（遍历历史位置）顺序访问。
2. 追加写的位置是**当前的 `size_`**，只有当 `layer == n_layer - 1` 时才把 `size_` 加一（src/kvcache.cpp）—— 【为什么】一次增量前向要逐层写同一个位置；「这个位置写完了没有」由「最后一层有没有写」判定，`size_` 因此表示**已完成的位置数**，而不是「已写入的层数」。
3. 由此带来一条反直觉的规则：**读缓存时的边界校验必须用容量 `cap_` 而不是 `size_`** —— 【为什么】逐层推进的过程中，本位置的 K/V 正在被写、也马上要被读，此时 `size_` 还没涨，用 `size_` 校验会误报越界（详见 §17.2）。而 `clear()` 把 `size_` 归零、缓冲不释放（src/kvcache.h）—— 【为什么】换一轮对话时复用同一块内存，不必重新分配。

#### 7.2 增量前向与「增量 ≡ 全量」

`[取嵌入行] -> [逐层: RMSNorm→QKV→RoPE→写入缓存→注意力→残差→FFN→残差] -> [最终 RMSNorm + 词表投影]`

1. 全量前向不是另一套实现 —— 【为什么】它内部建一个临时 KV 缓存并逐位置调增量前向（src/model.cpp），**增量路径是唯一的计算路径**；全量前向只是「把每行 logits 收集起来」的外壳。这样「优化前后共享一份代码」这条纪律在结构上就被保证了。
2. 位置 p 的输出只依赖两样东西：**缓存里 0..p-1 的 K/V** 和 **位置 p 自己的 K/V**。而 K/V 只依赖 `(token, pos)` —— 【为什么】RoPE 的频率只由 `pos` 与 `theta` 决定，投影只由 token 决定。所以「增量」与「全量」在数学上是同一件事的两种写法，**逐位一致是结构性的，不是数值巧合**；因果性也由此自动成立 —— 改末位 token 不影响前 n-1 个位置的 logits。
3. 单元 4 的核心不变式测试（tests/test_generate.cpp:136）把这条性质钉死：逐位置增量前向的每一行都要与全量前向的对应行在 1e-4 容差内相等（:153）；`clear()` 后重跑第 0 个位置结果不变（:157）；跳 pos 调用必须抛异常（:166）。

#### 7.3 采样链

`[贪心?] -> [softmax（未加温度）] -> [top-k] -> [top-p] -> [温度] -> [逆 CDF 抽样]`

1. `temp <= 0` 直接取 argmax（src/sampler.cpp）—— 【对照】上游的 `temp <= 0` 由 `llama_sampler_temp_impl`（src/llama-sampler.cpp:265 @ 88b47a755）内部处理：把除最大值外的 logit 全设成 -inf，效果等价于 argmax；上游另有独立的 `llama_sampler_greedy`（:935）后端供显式选择。
2. 截断用的概率分布是**未加温度**的那一份（src/sampler.cpp）—— 【对照】上游默认链的节序是 `top-k → top-p → … → temperature → dist`（examples 层 common/common.h:259 的 samplers 列表 + common/sampling.cpp:376 追加 dist），即**温度排在截断之后**。本仓库照此对齐，详见 §17.3。
3. top-k 用 `partial_sort` 取前 k（src/sampler.cpp）—— 【性能】词表有 3~15 万项，`partial_sort` 比全排序快。
4. top-p 按累积概率截断，**保留把累计推过 p 的那一个**（src/sampler.cpp）—— 【对照】上游 `llama_sampler_top_p_apply`（src/llama-sampler.cpp:1355 @ 88b47a755）判据是 `cum_sum >= p && i + 1 >= min_keep`，注释写明「the current iterate should be included in the set」。丢掉它会把分布砍得比 p 还狠。
5. 温度作用在存活候选的 logit 上，随后在存活集合内重新归一（softmax 天然归一）—— 【为什么】被裁掉的那部分概率质量就此丢弃，剩余候选重新构成一个合法的概率分布；最后用逆 CDF 抽样（画一个 `[0,1)` 的随机数，看它落在谁的累计区间里，src/sampler.cpp）—— 【对照】上游 `llama_sampler_dist`（src/llama-sampler.cpp:1026 @ 88b47a755）思路相同，差别只在它用 double 累加。

#### 7.4 生成循环

`[prefill: 逐位置前向] -> [采样] -> [前向到位置 n] -> [回显] -> ...`

1. prefill 把 prompt 的每个 token 推进缓存，保留**最后一个位置的 logits**（src/main.cpp）；循环体是「采样 → 若命中 EOS 则停 → 回显文本 → 把该 token 前向到位置『已推进位置数』」（src/main.cpp）—— 【为什么】位置用 `ids.size() + gen` 推进，`gen` 是本轮已生成的 token 数，正好等于「当前位置减去 prompt 长度」。

### §8 异常流转表

| 异常 | 触发点 | 状态/字段变化 | 恢复路径 |
|---|---|---|---|
| KV 缓存参数为 0 | 构造函数（src/kvcache.cpp） | 对象未构造，没有发生任何内存分配 | 不可恢复：调用方参数错，只能改调用参数 |
| 缓存写满 | 追加（src/kvcache.cpp） | 已写入的历史位置保持不变，本位置未写入 | 不可恢复：超出 context length，应在调用方提前停 |
| 缓存读取越界 | `k()` / `v()`（src/kvcache.cpp） | 缓存内容不变 | 不可恢复 |
| 顺序违规（`kv.size() != pos`） | 增量前向入口（src/model.cpp） | 缓存不变，残差流未产生 | 不可恢复：调用方必须按 pos 递增调用 |
| token id 越界 | 增量前向入口（src/model.cpp） | 缓存不变 | 不可恢复 |
| 空 logits | `Sampler::sample` 入口（src/sampler.cpp） | 随机数状态不变（未消费） | 不可恢复 |

### §9 数据形状

```jsonc
// KVCache 私有字段（src/kvcache.h）—— 对应上游 llama_kv_cache 的 cell 池（降级还原，见 §16.3）
{ "n_layer": 22, "dim": 256, "cap": 2048, "size": 0 }   // u32 / u32 / u64 / u64
//   布局：k_[((layer*cap) + pos) * dim + i]，v_ 同构；size = 已完成的位置数（最后一层追加之后才 +1）
// SamplerParams（src/sampler.h）：temp<=0 走贪心；top_k<=0 不限数量；top_p<=0 或 >=1 不截断
{ "temp": 0.8, "top_k": 40, "top_p": 0.95, "seed": 1234 }  // f32 / i32 / f32 / u32
// 生成循环的位置关系（src/main.cpp）：gen_N_pos = ids.size() + N
{ "prompt_tokens": 2, "kv.size()": 2, "gen_0_pos": 2, "gen_1_pos": 3 }
```

## 开发流水线扩展节

### §13 编排锚点与挂载点

| 角色 | 路径 | 承载什么 |
|---|---|---|
| 前向编排 | src/model.cpp `Model::forward_token` / :463 `Model::forward` | 增量前向的唯一实现（注意力 + FFN + 输出头，全部数学）；`forward` 只是「临时缓存 + 逐位置收集」的外壳，**不含任何独立计算逻辑** |
| 存储契约 | src/kvcache.h `KVCache` / src/kvcache.cpp `append` | K/V 的追加与按 (层, 位置) 读取 |
| 采样与生成循环 | src/sampler.cpp `Sampler::sample` / src/main.cpp | 贪心 / softmax / top-k / top-p / 温度 / 抽样；prefill → 采样 → 前向 → 回显，直到 EOS 或上限 |

**删掉锚点流程还跑不跑得起来**：`forward_token` 是唯一的计算路径，删掉它 `forward`、CLI、单元 3/4/5 的全部测试都编译不过；`Sampler::sample` 是 logits 与 token 之间唯一的桥，删掉它生成循环立刻断。

| 文件:行 | 改什么 | 为什么必须在这儿 |
|---|---|---|
| src/kvcache.h 构造 / :83 `append`；src/kvcache.cpp `size_` 递增 / :56 与 :62 边界校验 | 布局与追加协议；「最后一层写完才算一个位置完成」；边界校验用容量而非 size | 「位置」的语义在这里定义；逐层推进的中间态里本位置正在被写（见 §17.2） |
| src/model.cpp `Model::forward_token` / :481-483 顺序校验 | 注意力 + FFN + 输出头；「必须按 pos 递增调用」的前置条件 | 单元 3/4/5 共用这一条路径，等价性测试钉在这里；顺序校验是「缓存只追加」协议的唯一守卫 |
| src/sampler.cpp 未加温度的 softmax | 截断的依据分布 | 节序对齐上游的关键一处（见 §17.3） |
| src/sampler.cpp top-p 判据 | `cum >= p` 且含跨阈值那一个 | 与上游 `llama_sampler_top_p_apply` 逐条对应 |
| src/main.cpp 位置推进 | `ids.size() + gen` | 位置算错会让下一次增量前向的顺序校验抛异常 |

### §14 概要→详细映射表

| 概要条款 | 详细落点 | 参考实现对应 |
|---|---|---|
| §1.1 `KVCache` / `SamplerParams` / `Sampler` | §7.1、§7.3、§9 | 上游 `llama_kv_cache` 的「按位置存放历史 K/V」语义 + `llama_sampler_chain` 的逐节 logits 变换顺序（每节一次 `apply`，**顺序即语义**） |
| §1.3 `cell` / `slot` / `seq_id` / `prefill` / `decode` | §7.1、§7.4、§16.1 | 上游 `llama_kv_cells` 的 cell 身份 + `find_slot` 的「按序列分配可复用 cell」机制；prefill / decode 是「一批装入 prompt」与「一批只带 1 个 token」两种批形态 |
| §2 步骤 1-2 | §7.1、§7.2 | 上游「按位置分配 cell、再把本位置 K/V 写回 cell 池」的追加协议 |
| §2 步骤 3 | §7.2 步骤 1-3 | 上游把 K/V 拷进 cell 池张量、注意力段再按 cell 读回；因果性由「只读 0..pos 的 cell」保证 |
| §2 步骤 4 | §7.3、§7.4 | 上游 `llama_sampler_chain` 的逐节顺序（截断节在温度节之前，末节是随机抽样）+ examples 层「采样 → 前向 → 再采样」的自回归循环骨架 |
| §3 异常 1-3 | §8 前 3 行 | 上游在 cell 池耗尽 / 找不到可复用 slot 时同样中止 |
| §3 异常 4 | §8 第 4 行 | 上游用 `llama_batch` 的 pos 数组表达位置，不检查「顺序」；本实现用「size == pos」的顺序协议替代 |
| §3 异常 5 | §8 第 5 行 | 上游在 token 化阶段校验 token id，不在 decode 阶段 |
| §3 异常 6 | §8 第 6 行 | 上游 sampler 对空候选集返回 -1 |
| §5 观测点 | §5 | 上游 CLI 打印 `llama_perf_context` 的 prefill / decode 分项计时；本仓库打印总耗时与 tok/s |

### §15 裁量区与待拍板

| AI 可以自决 | 必须对齐参考实现 | 必须问人 |
|---|---|---|
| 教学便利代码的实现方式：微型模型构造器的形状（层数 / 维度 / 词表）、独立仲裁者用 numpy 还是纯 python、Unicode 区间表由脚本生成还是手写 | 术语（名字）：KV cache / cell / slot / seq_id / prefill / decode / greedy / temperature / top-k / top-p | 砍掉承重件：slot / cell 复用与 seq_id 多序列 —— 当前按「降级还原」处理，扩场景前必须先问 |
| 教学便利代码的错误消息措辞与是否带位置号 | 结构档：K/V 的 `[layer][pos][dim]` 分层与字段职责；增量路径是唯一计算路径 | 新增参考实现没有的概念：本实现不引入上游之外的新概念，若要引入必须先问 |
| 教学便利代码的测试夹具形状选择与断言写法 | 结构档：采样链的**节序**（top-k → top-p → temperature → dist）与各节判据 | 改对齐粒度定义：从「逐 token 顺序」扩到「多序列并行」必须先问 |

## 还原模式扩展节

### §16 对齐面

> 参考实现：llama.cpp @ 88b47a755；本文只对「本单元涉及的部分」负责。

#### 16.1 概念对齐

| 参考实现的术语 | 本实现的对应物 | 对齐到什么粒度 | 备注 |
|---|---|---|---|
| `llama_kv_cache` + `cpy_k` / `cpy_v`（写回缓存） | `KVCache`（src/kvcache.h）与 `KVCache::append`（src/kvcache.cpp） | 结构档：调用链骨架 + 机制 | 上游的对外接口是 `llama_memory_i`（src/llama-memory.h:73），有 4 个实现；上游把 K/V 拷进 cell 池的 ggml 张量，本实现拷进 float 缓冲 |
| cell / slot（`llama_kv_cells`，src/llama-kv-cells.h:32；`llama_kv_cache::slot_info`，src/llama-kv-cache.h:34） | `KVCache` 的 `[layer][pos]` 定位；无 slot 层 | **概念偏离（已登记）** | 上游 cell 有独立身份、由 `find_slot` 分配（src/llama-kv-cache.cpp:958），slot 是「某条序列当前占用哪批 cell」的记录；本实现的 cell 身份就是 `pos` —— 见 §16.3 第 1 行 |
| `llama_seq_id` / seq_id（include/llama.h:70） | 无对应物 | **概念偏离（已登记）** | 上游一条 batch 里可带多条 seq_id，各占各的 slot；本实现恒为单序列 —— 见 §16.3 第 2 行 |
| `llama_memory_seq_pos_max`（include/llama.h:785） | `KVCache::size()`（src/kvcache.h） | 机制 | 上游是「某条序列已占用的最大位置」，本实现是「已完成的位置数」，单序列下同义 |
| `llama_context::decode`（src/llama-context.cpp:1693） | `Model::forward_token`（src/model.cpp） | 机制 + 调用链骨架 | 上游一次 decode 推进一整个 batch，本实现一次推进一个位置 |
| prefill / decode | 同一函数的两种调用方式（src/main.cpp / :101-104） | 术语 + 调用链骨架 | 上游是「装入整批 prompt」与「batch 只带 1 个 token」两种批形态，本实现是同一个函数的不同调用方式 |
| `llama_sampler_chain` | `Sampler::sample`（src/sampler.cpp） | 机制 + **节序** | 上游每节是独立对象可任意组合，本实现压成固定顺序的一段函数，**节序与上游默认链一致**（见 §16.2） |
| `llama_sampler_greedy`（src/llama-sampler.cpp:935） | `temp <= 0` 的贪心分支（src/sampler.cpp） | 机制 | 上游另有一个显式的 greedy 后端；本实现的 temp<=0 与它等价 |
| temperature（`llama_sampler_temp_impl`，src/llama-sampler.cpp:265） | 步骤 5 温度缩放（src/sampler.cpp） | 机制（作用位置一致） | 上游在截断之后对存活候选做温度缩放，本实现同序 |
| top-k（`llama_sampler_top_k`，src/llama-sampler.cpp:321） | 步骤 3 top-k 截断（src/sampler.cpp） | 机制（判据逐条一致） | 上游用 `llama_sampler_top_k_apply` 取前 k，本实现用 `partial_sort` 取前 k |
| top-p（`llama_sampler_top_p_apply`，src/llama-sampler.cpp:1355） | 步骤 4 top-p 截断（src/sampler.cpp） | 机制（判据一致，归一基准有偏差） | 「含跨阈值那一个」一致；归一基准的偏差见 §16.2 |

#### 16.2 设计对齐（先「结构档的偏离」表，后「实现档的替换」表）

| 参考实现的机制 | 本实现的位置 | 偏离点 | 为什么可以偏离 |
|---|---|---|---|
| cell 池 + `find_slot` 分配 + seq_id 多序列（src/llama-kv-cache.cpp:958） | src/kvcache.h `KVCache` 纯顺序追加 | **概念偏离（承重不可砍 → 降级还原）** | 详见 §16.3 第 1 行；已登记在 `docs/constraints/04-对齐约束.md`，**不允许在文档里说成「等价」** |
| 采样链可任意组合（`llama_sampler_chain_add`，src/llama-sampler.cpp:880） | src/sampler.cpp 固定顺序的函数 | 细节简化：少了组合自由度，**节序已对齐上游默认链** | 上游默认链的节序是 `top-k → top-p → … → temperature → dist`（common/common.h:259 + common/sampling.cpp:376）；本实现照此排（见 §17.3） |
| top-p 的截断口径：上游在**候选集内**归一（`llama_sampler_softmax_impl(cur_p, false)`，src/llama-sampler.cpp:1362），`p <= 0` 时第一次累计就满足 `cum_sum >= p`（只剩 1 个候选），并有 `min_keep` 下限 | src/sampler.cpp 在**全词表**上归一；:108 `top_p > 0 && top_p < 1` 才截断 | 机制替换：归一基准不同（细节简化：`top_p <= 0` 在本实现是「不截断」，且无 `min_keep`） | 累计概率的口径不同：实测 20 个递减 logits、top_k=10、top_p=0.90 时上游保留 7 个、本实现保留 8 个；top_p=0.95 时上游 8 个、本实现 10 个。**top_k 未启用时两者一致**（候选集 = 全词表）；`top_p <= 0` 在单序列生成里不会被触发 |
| `llama_context::decode` 一次推进一整个 batch（src/llama-context.cpp:1693） | src/model.cpp 一次推进一个位置 | 机制替换：batch → 单 token | 正确性等价（本单元的等价性测试逐位比对过）；丢掉的是并行解码的吞吐 |
| 上游 `llama_sampler_dist` 用 double 累加 + `uniform_real_distribution<double>`（src/llama-sampler.cpp:1040） | src/sampler.cpp 用 float 累加 | 细节简化：float vs double | 分布形状一致；概率极小的尾项在 float 下有 1e-7 量级误差 |
| 采样链的节序（top-k → top-p → temperature → dist） | src/sampler.cpp | **无偏离** | 上游默认链的节序（common/common.h:259 + common/sampling.cpp:376）与本实现逐节一致；同样的 (temp=0.8, top_p=0.9) 输入下两者截断出的候选集相同 |

| 参考实现用什么 | 最小版用什么 | 能力是否等价 |
|---|---|---|
| 链式采样器：上游每节一个对象，由 `llama_sampler_chain_add`（src/llama-sampler.cpp:880）串成一串 | 固定顺序的一段函数（src/sampler.cpp 的步骤 1~6） | 能力不减：各节判据与节序与上游默认链一致；以后接得上：拆成独立对象即可 |
| 多 KV cache 实现：上游 `llama_memory_i`（src/llama-memory.h:73）下有默认 / DSA / ISWA / dsv4 四种 | 纯顺序版（src/kvcache.h 只有一种） | **有损** —— 已按承重不可砍登记为降级还原，不得在文档里说成「等价」；单序列顺序生成的能力不减 |
| 批接口与并发：上游 `llama_batch` / `llama_ubatch`（src/llama-batch.h:15）多 slot 并发解码 | 逐 token 调用（src/model.cpp 一次推进一个位置）、单线程单序列 | 能力不减：数值语义一致；以后接得上：把「一位置一批」扩成「一批多位置」是加法 |
| 容器：上游 `llama_kv_cells` 的 cell 池 + ggml 张量 | `std::vector<float>` 两块缓冲（src/kvcache.h） | 能力不减（单序列场景）：K/V 的存取语义完全一致；以后接得上：把索引公式换成 cell 查找即可 |
| 错误处理：上游返回 `nullptr` / `-1` + `GGML_LOG_ERROR` | 抛 `std::runtime_error` | 能力不减：不可恢复错误同样立即中止；接得上：可加一层 try 包装成上游风格 |

#### 16.3 砍掉清单的承重判断

| 砍掉什么 | 真实版怎么做 | 概念测试 | 行为测试 | 术语测试 | 结论 |
|---|---|---|---|---|---|
| **slot / cell 管理与复用** | `llama_kv_cache::find_slot`（src/llama-kv-cache.cpp:958）在 cell 池里挑一个可复用位置；cell 有独立身份（`llama_kv_cells`，src/llama-kv-cells.h:32） | **不通过** —— `find_slot` 的存在意义是「在多序列 / 乱序到达时决定 K/V 落在哪个 cell」。砍掉后 **cell 这个概念塌了**：本实现里 K/V 的位置就是 `pos`，没有独立于位置的 cell 身份 | **通过** —— 单序列顺序生成的典型路径跑通，logits 逐位不变 | **通过** —— 用户仍叫它 KV cache | **承重，不可砍 → 降级还原**：降级为「纯顺序追加（pos == size）」的单序列缓存。「缓存 = 按位置存放的历史 K/V」核心语义完整保留，丢掉的是 cell 身份与跨 batch 复用。**已登记为已知简化** |
| **seq_id 多序列** | 上游 `llama_batch` 的一条 batch 可带多条 seq_id，各占各的 slot（include/llama.h:70） | **不通过** —— 多序列是「同一批 cache 服务多条会话」的概念；砍掉后本实现连「有几条序列」这个问题都不存在 | **通过** —— 单序列顺序生成的典型路径跑通 | **通过** —— 单会话场景下用户不会提 seq_id | **承重，不可砍 → 降级还原**：与上一行同一降级方案（纯顺序追加，单序列）；丢掉的是并发会话，**已登记为已知简化** |
| 4 种 KV cache 实现（默认 / DSA / ISWA / dsv4） | 上游 `llama_memory_i` 下的四种实现，差异在缓存压缩策略（attention sink、dramatic shift） | **通过** —— 四种实现都只改「缓存里存什么」，不改「缓存 = 历史 K/V」这个概念 | **通过** —— 单序列顺序生成路径的 logits 与压缩策略无关 | **通过** —— 保留上游「DSA / ISWA / dsv4」词 | **可砍** —— 主干只留默认实现 |
| 采样链 10+ 节（min-p、重复惩罚、grammar、mirostat 等） | src/llama-sampler.cpp 每节一个 backend | **通过** —— 每节都是「变换 logits 或挑 token」的局部逻辑；主干四节（temperature / top-k / top-p / dist）已覆盖「裁剪分布 + 抽样」这个概念 | **通过** —— 单元 4 的采样器测试（贪心 / top_k=1 / 尖峰 / 种子复现 / top_p）覆盖主干路径 | **通过** —— 沿用上游各节的名字 | **可砍** |
| `llama_batch` / ubatch 并行解码 | `llama-batch.cpp` + `llama_context::decode` 的批处理 | **通过** —— 并行只改执行顺序不改数学 | **通过** —— 单 token 路径的 logits 与全量路径在 1e-4 容差内一致 | **通过** —— 保留上游 `batch` / `ubatch` 词 | **可砍** |
| 投机解码 | `common/speculative.cpp` 的 `common_speculative_draft`（:2545）+ `common_speculative_accept`（:2626）：先起草多个 token 再批量验证 | **通过** —— 投机只改「一次产出几个 token」，不改「token 从哪来」的语义 | **通过** —— 不影响 logits，落点仍在采样器 | **通过** —— 保留上游「speculative」词 | **可砍** —— 加速特性，不进入数值语义 |
| 层卸载（ngl） | `llama-model-loader` 的设备管理：把部分层放到 GPU | **通过** —— 卸载只改「权重放哪」，不改前向的数学 | **通过** —— 不影响 logits | **通过** —— 保留上游「ngl / offload」词 | **可砍** —— 显存特性，不进入数值语义 |

#### 16.4 教学便利代码清单

| 多出来的 | 在哪 | 为什么需要 | 谁 import 它 |
|---|---|---|---|
| 微型模型构造器 `TinyModelBuilder` | tests/test_generate.cpp:58 | 真实版要下几个 GB 权重才能测；这里现场合成 1~2 层微型 LLaMA，权重用确定性伪随机 | tests/test_generate.cpp |
| GGUF 字节流写入器 `minitest::Writer` | tests/gguf_writer.h:13 | 与 src/gguf.cpp 的解析器互为逆运算，把 GGUF 格式契约变成可执行文档 | tests/test_generate.cpp（及单元 1~5 的测试） |
| 独立仲裁者（numpy） | tests/arbiter.py:1 | 从 GGUF 原始字节手写解析 + 前向，零复用本仓库任何代码，用来裁决 ours-vs-llama.cpp 的发散归属 | 无（命令行人工运行） |
| Unicode 区间表生成器 | tools/gen_unicode_tables.py | 上游 unicode-data.cpp（164K）同样由脚本生成；本仓库用 Python 标准库现场生成区间表 | 构建前手动执行（属单元 2 范围，在此登记在册） |
| CLI 入口 `main` | src/main.cpp | 真实版是 examples/ 与 tools/ 下几十个程序；这里一个 main 覆盖「加载 → 编码 → prefill → 采样 → 生成」。**它是教学便利代码、不是核心路径**：真实版对应 `examples/simple-chat` / `tools/cli` 这类**应用外壳**而非库本体，本仓库核心库（src/model.cpp / src/kvcache.cpp / src/sampler.cpp）不依赖 src/main.cpp，删掉 CLI 后单元 1~5 的测试仍全部通过 | （独立可执行文件） |

### §17 踩坑记录

### 17.1 头文件互相 include，编译直接死锁在模板实例化 【坑型：概念层级错位】

**症状**：`model.h` 里要前向声明 `KVCache`，`kvcache.h` 里要引用 `HParams`，于是两边互相 `#include`，编译器报「include 后再 include 不生效」或模板重复实例化。
**根因**：两个头文件被放在了**同一层**，但它们的依赖是单向的 —— `kvcache.h` 只用基本类型（`uint32_t` / `size_t` / `float`），**根本不需要 `model.h`**。层次放错之后，一个本来可以单向的依赖被写成了环。
**修法**：删掉 `kvcache.h` 里的 `#include "model.h"`，依赖变成单向 `model.h → kvcache.h`。
**真实版**：上游用前向声明 + `src/llama-memory.h` 单独定义 `llama_memory_i` 接口，`llama-context.h` 再包含它；同一套「接口头不依赖实现头」的规矩。本质相同：分层是为了让依赖单向。
**教学点**：**头文件之间的包含关系必须是一条 DAG**。发现环的时候，答案几乎总是「有一层放错了」而不是「得用前向声明糊过去」。前向声明能绕过编译错误，但会把层次错误藏起来 —— 真正的修复是把那一层挪到正确的位置。

### 17.2 边界校验用了 `size_`，逐层推进时误报越界 🔴 【坑型：不变式被打破】

**症状**：第一次跑带 KV cache 的前向就抛「KVCache 访问越界」，但明明只是刚写了第一个位置。
**根因**：追加的协议是「逐层写同一个位置，写完最后一层才把 `size_` 加一」（src/kvcache.cpp）。于是一次增量前向的执行过程中存在一个**中间态**：

```text
第 0..n-2 层：本位置的 K/V 已写入，size_ 还没涨
最后一层：写完，size_ += 1
```

注意力段要在**每一层内部**读回本位置的 K/V，此时 `size_` 还没涨。若 `k()` / `v()` 用 `pos >= size_` 校验，第一个位置就会被判越界。
**修法**：边界校验改用**容量** `cap_`（src/kvcache.cpp / :62）—— 真正要防的是「读到分配之外」，而不是「读到还没写完」；「还没写完」由增量前向入口的 `kv.size() == pos` 协议来管。
**真实版**：上游不用「size」这个概念，而是每个 cell 有独立的 `pos` 标记（`llama_kv_cells`），读之前按 cell 状态判断，不存在这个中间态 —— 但它同样用「池容量」而不是「已用数量」做边界。本质相同：**边界是分配边界，不是进度边界**。
**教学点**：**「已完成的位置数」和「已写入的层数」是两个数**。凡是「分步完成一个整体」的数据结构，都要先问一句：进度计数在中间态里代表什么？用错计数做校验，报错会指向一个完全无辜的位置 —— 而且只在第一次跑时出现，因为第一次的中间态最明显。

### 17.3 采样链的节序被当成了「实现细节」 🔴 【坑型：文档纪律】

**症状**：与上游逐 token 对拍时，采样结果在第 2 个 token 就分叉了；而单看每一节（温度 / top-k / top-p / 抽样）的实现都是对的。
**根因**：本实现原先的节序是 `温度 → softmax → top-k → top-p → 抽样`，**温度排在截断之前**；上游默认链（examples 层 common/common.h:259 的 samplers 列表 + common/sampling.cpp:376 追加 dist）是 `top-k → top-p → … → temperature → dist`，**温度排在截断之后**。两者的差别在 top-p 上是可见的：

```text
logits = {3, 2, 1, 0, -1}，temp = 2，top_p = 0.9
上游（先截断）：未加温度的累计 = 0.636 / 0.871 / 0.957 → 保留 3 个
本实现（先加温）：加温后的累计 = 0.429 / 0.689 / 0.846 / 0.942 → 保留 4 个
→ 候选集合不同 → 从这一步开始生成的文本就分叉
```

更糟的是，旧文档里写着「top_p 截断……与 llama.cpp 行为一致」——**这句话只对了一半**：「含越界那一个」确实一致，但**节序**不一致，而节序同样属于行为。
**修法**：把 `Sampler::sample` 改成上游的节序 —— ① 贪心分支 → ② 未加温度的 softmax（截断依据）→ ③ top-k → ④ top-p → ⑤ 温度作用在存活候选上 → ⑥ 逆 CDF 抽样（src/sampler.cpp）；并把 top-p 的判据对齐上游的 `cum >= p`（src/sampler.cpp）。同时在 `lines/numeric/sample_line.cpp` 第 6 跳加了一条可观测断言：**temp=2 时 token 3 仍然一次都抽不到**（截断在温度之前）。
**真实版**：上游的节序写在 examples 层的默认 samplers 列表里（common/common.h:259），**不是**库层的硬约束 —— 上游允许 `--samplers` 任意重排。但「默认链的节序」就是 llama.cpp 的事实行为，本项目作为还原必须与它一致。
**教学点**：**「实现细节」和「行为语义」的边界，取决于对齐目标是什么**。同样是「温度除以 logits」，放在截断前还是后面，在上游是两种可观测的行为。旧文档里那句「与 llama.cpp 行为一致」不是小瑕疵 —— 它是**用一句没被验证过的话，掩盖了一个真实的语义差异**。凡是写「与上游一致」，就要能指出是**哪一条判据**、**哪一个顺序**一致；否则这句话就是在给自己埋雷。
