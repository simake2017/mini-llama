# 单元 4：KV Cache + 生成循环 + 采样 —— 从"会算"到"会说"

（对应真实 llama.cpp 的 src/llama-kv-cache*.cpp + src/llama-sampler.cpp + examples CLI）

## ① 我实现了什么

- **KVCache**：按层分块的 K/V 缓冲 `[layer][pos][dim]`，顺序追加、按 (layer,pos) 读取、clear 复用
- **前向重构**：`forward_token(token, pos, kv)` 增量前向成为唯一计算路径，`forward(tokens)` 只是"临时缓存逐位置推进"的语法糖——**优化前后共享一份代码**
- **核心不变式测试**：增量前向与全量前向逐位置 logits 一致（1e-4 容差），把"缓存只是避免重算"这一语义钉死
- **采样器**：greedy（temp≤0）→ 温度缩放 → top-k 截断 → top-p（核采样）截断 → 按概率抽样，固定种子可复现
- **生成循环**：prefill 逐 token 推进缓存 → 采样 → 前向 → 直到 EOS/上限；CLI `mini-llama --model --prompt [--temp --top-k --top-p --n --seed]`

关键设计决策：

- **KVCache 访问校验用容量而非 size_**：`forward_token` 内逐层追加时，前 n-1 层的本位置数据已写入、但 size_ 要末层才递增——这是顺序追加协议的一个真实陷阱，注释里写明了。
- **top_p 截断包含"越界那一个 token"**（cum > p 时保留当前 token），与 llama.cpp 行为一致；截断后部分概率质量被丢弃，抽样前重新归一。
- **采样器保留链式语义注释**：真实版每节是独立对象可任意组合，我们把固定顺序写成一个函数，主干逻辑相同。

## ② 真实 llama.cpp 对应源码

| mini-llama | 真实版 | 说明 |
|---|---|---|
| `src/kvcache.h` | `src/llama-kv-cache.cpp`（llama_kv_cache） | 真实版还有 slot/cell 复用、seq_id 多序列；我们是纯顺序版 |
| `forward_token` | `llama-context.cpp` 的 decode 路径 + `llama-graph` 单 batch 图 | 等价 |
| `src/sampler.h` | `src/llama-sampler.cpp`（llama_sampler_chain） | 链式 vs 固定顺序函数 |
| 生成循环 | `examples/simple-chat` / `llama-cli` | 骨架一致 |
| top-k/top-p | `llama_sampler_top_k` / `llama_sampler_top_p` | 语义一致 |

## ③ 真实版比我多做了什么、为什么

| 真实版多做 | 为什么可以砍 |
|---|---|
| 4 种 KV cache 实现（默认 / DSA / ISWA / dsv4） | 前 3 种是缓存压缩研究（attention sink、dramatic shift），非主干 |
| slot/cell 管理、seq_id 多并发序列、跨 batch 复用 | 对话服务器场景需求，单序列生成用不到 |
| 采样链 10+ 节（min-p、重复惩罚、grammar 约束、mirostat 等） | 每节都是"变换 logits 或挑 token"的局部逻辑，主干四节已覆盖 95% 用法 |
| `llama_batch` / ubatch 并行解码 | 吞吐优化，正确性无关 |
| 投机解码、llama-ngl 层卸载 | 加速/显存特性 |

## ④ 踩坑记录

- **头文件循环包含**：`model.h ↔ kvcache.h` 互相 include。解法：`kvcache.h` 其实不依赖 model.h（只用基本类型），删掉即可——头文件依赖要保持最薄。
- **KVCache 的 size 校验时机**：如上，逐层追加协议下"本位置是否已写入"和"全局 size"是两回事，校验目标选错会误报越界。

## ⑤ 验收记录

- 合成用例 3 组全绿：KV 等价性（含 clear 复用、顺序违规检测）、采样器（贪心/top_k=1/统计尖峰/种子复现/top_p）、生成循环（贪心可复现、id 合法）
- ctest 4 个单元全绿；`mini-llama` CLI 编译通过（端到端真实模型验证待单元 5 量化支持后用 qwen3 Q4_K_M 完成）
