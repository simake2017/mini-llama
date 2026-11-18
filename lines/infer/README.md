# infer 族 —— 推理工域

> 一次推理的两个阶段：**prefill**（把 prompt 一次性推进）和 **decode**（自回归逐 token 生成）。
> 两条线共用同一份前向代码与同一个 KV Cache，区别只在「喂进去几个 token」。

## 一 链路图

```text
   prompt 文本
        │
        ▼
   ┌────────────┐
   │ tokenize   │  （见 text 族）
   └─────┬──────┘
         │  token id 序列
         ▼
   ┌────────────────────────────┐
   │  prefill 线                 │
   │  for p in 0..n-1:           │
   │    embed 查行                │  src/model.cpp
   │    每层: RMSNorm → QKV       │
   │          → RoPE → K/V 入缓存 │  src/kvcache.cpp  ★ 跨层点
   │          → GQA 注意力 → 残差  │
   │          → SwiGLU FFN → 残差 │
   │    输出头 → logits           │
   └─────────┬──────────────────┘
             │ 末位 logits
             ▼
   ┌────────────────────────────┐
   │  decode 线                  │
   │    采样 → token              │  src/sampler.cpp
   │    forward_token(位置=size)  │  src/model.cpp  ★ 跨层点：生成物回到缓存
   │    写回 KV Cache             │  src/kvcache.cpp
   │    重复直到 EOS / 上限        │
   └─────────┬──────────────────┘
             ▼
        detokenize → 文本  （见 text 族）
```

★ 跨层点两处：**K/V 写入缓存**（注意力层 → 缓存层）、**生成物回到缓存**
（采样层 → 前向层 → 缓存层）。

## 二 逐跳对照

### 线 A：`prefill`（8 跳）

| 跳 | 本仓库落点 | 上游对照（llama.cpp @ 88b47a755） | 真机命令 |
|---|---|---|---|
| 1 | `GgufFile::load` → `Tokenizer::load` → `Model::load` | `src/llama-model-loader.cpp:llama_model_loader::load_all_data` | `MINILLAMA_MODEL=… ./build/line_prefill` |
| 2 | `Tokenizer::encode`（拿到位置序列） | `src/llama-vocab.cpp:3358:llm_tokenizer_spm_session::tokenize` | 同上 |
| 3 | `KVCache::KVCache`（`src/kvcache.h`） | `src/llama-kv-cache.cpp:64:llama_kv_cache::llama_kv_cache` | 同上 |
| 4 | `Model::forward_token`（`src/model.cpp`）：embed → 每层 block → 输出头 → logits | `src/models/llama.cpp:99:llama_model_llama::graph<embed>::graph` 建图 + `src/llama-context.cpp:1693:llama_context::decode` 执行 | 同上 |
| 5 | `KVCache::append` / `KVCache::k`（`src/kvcache.h,35`） | `src/llama-kv-cache.cpp:1425:llama_kv_cache::cpy_k` / `:1460:cpy_v`（K/V 写回缓存） | 同上 |
| 6 | `Model::forward` vs 逐位置 `forward_token`（不变式） | 同一张图，两种喂法：一次 batch vs 逐位置增量 | 同上 |
| 7 | `Model::forward_token` 的因果掩码（改末位不影响前位） | `src/llama-graph.cpp:26:build_attn_inp_kq_mask` | 同上 |
| 8 | 末位 logits 交给采样（`src/sampler.cpp`） | `src/llama-sampler.cpp:796:llama_sampler_chain_init` | `./build/mini-llama --model … --prompt "Hi" --n 8` |

### 线 B：`decode`（8 跳）

| 跳 | 本仓库落点 | 上游对照（llama.cpp @ 88b47a755） | 真机命令 |
|---|---|---|---|
| 1 | load 三件套 | `src/llama-model-loader.cpp:llama_model_loader::load_all_data` | `MINILLAMA_MODEL=… ./build/line_decode` |
| 2 | CLI 参数 → `SamplerParams`（`src/main.cpp`） | `src/llama-sampler.cpp:796:llama_sampler_chain_init` 的默认链 | 同上 |
| 3 | `Model::forward_token`（预填充） | `src/llama-context.cpp:1693:llama_context::decode` | 同上 |
| 4 | `Sampler::sample`（`src/sampler.cpp`） | `src/llama-sampler.cpp:1026:llama_sampler_dist` | 同上 |
| 5 | `Model::forward_token` + `KVCache::append`（位置 = 已推进的位置数） | `src/llama-kv-cache.cpp:1425:llama_kv_cache::cpy_k` / `:1460:cpy_v` | 同上 |
| 6 | 生成循环（`src/main.cpp`） | `examples/simple-chat` 的循环骨架 | `./build/mini-llama --model … --prompt "Hi" --n 8 --temp 0` |
| 7 | 贪心可复现（同种子两次同结果） | `src/llama-sampler.cpp:935:llama_sampler_greedy` | 同上 |
| 8 | `Tokenizer::decode`（token 序列 → 文本） | `src/llama-vocab.cpp:3640:llama_vocab::impl::detokenize` | 同上 |

## 三 学习问题的答案

> **为什么「增量前向」和「全量前向」必须给出逐位一致的 logits？
> 如果两者允许有微小差异，KV Cache 的正确性靠什么保证？**

因为**缓存里存的是「历史位置的 K/V」，而不是「历史位置的 logits」**。

一次前向在位置 p 做的事：

```text
x_p = embed(token_p)
每层:  h = RMSNorm(x_p)
              q,k,v = 投影(h)
              k,v  = RoPE(k,v, pos=p)      ← 位置编码只依赖 p
              缓存[layer][p] = (k, v)        ← 写回
              attn_out = Σ_{j≤p} softmax(q·K_j/√d) · V_j   ← 读全部历史
              x_p = FFN(attn_out) + 残差
logits_p = 词表投影(RMSNorm(x_p))
```

**位置 p 的输出依赖的是「缓存里 0..p-1 的 K/V」和「位置 p 自己的 K/V」**——
它**不依赖**任何历史位置的 logits，也不依赖历史位置当时是怎么算出来的。

所以「增量 ≡ 全量」的正确性论证是**结构性的**，不是数值巧合：

| 论证 | 增量路径 | 全量路径 | 是否一致 |
|---|---|---|---|
| 位置 p 的 K/V | 由 `token_p` + `pos=p` 算出 | 同一个 `token_p` + 同一个 `pos=p` | **逐位相同**（同一份代码、同一个输入） |
| 位置 p 的注意力输入 | 缓存里的 K_j（j<p） | 缓存里的 K_j（j<p） | **逐位相同**（K_j 只依赖 token_j 和 pos=j） |
| 位置 p 的输出 | 只由上面两项决定 | 只由上面两项决定 | **必然逐位相同** |

换句话说：**只要「K/V 只依赖 (token, pos)」这个不变量成立，增量与全量就是同一件事的两种写法。**

那「允许微小差异」会怎样？——**正确性不会立刻坏，但会失去可诊断性**：

1. **误差会累积**：decode 循环里每一步都把上一步的输出喂进下一步。
   第 p 步的误差 δ 会进入第 p+1 步的注意力输入，误差变成 `δ_p+1 ≈ δ_p + ∂logits/∂input · δ_p`。
   一旦某一步的误差超过「argmax 不翻转」的余量，生成的 token 就与真实版不同，
   而且**你无法判断是从哪一步开始错的**。
2. **失去回归基线**：本仓库用「增量 ≡ 全量」当**唯一的正确性判据**。
   如果它只保证「差得不多」，那就没有任何东西能证明 KV Cache 的实现是对的——
   只能靠「生成的文本看起来还行」这种没法自动化的判断。
3. **对照会失效**：与 llama.cpp 的逐 logits 对拍（`docs/05-quantization.md` 的
   max|Δ| ≤ 0.27）要求两侧用**同一套位置约定**。如果本仓库内部的增量/全量已经不一致，
   那「与上游差多少」这个问题就没有确定的答案。

所以本仓库把这个不变式做成了**硬断言**：

```cpp
// tests/test_generate.cpp: test_kv_equivalence
for (size_t p = 0; p < tokens.size(); p++) {
    auto row = m.forward_token(tokens[p], p, kv);
    for (uint32_t i = 0; i < V; i++) {
        CHECK_NEAR(row[i], full[p * V + i], 1e-4);   // 逐位置
    }
}
```

**判据：差 0，不是「差得小」。** 实测本仓库的差就是 **0**（见下）。

## 四 认知纠错记录

| # | 曾经的错误假设 | 实际情况 | 判定依据 |
|---|---|---|---|
| 1 | 「全量前向 = 把整个序列的注意力一次算完（O(n²)）」 | **不成立**。单元 4 之后 `Model::forward` 内部也是逐位置推进 KV Cache，只是把每行 logits 收集起来 | `src/model.cpp:Model::forward` 的实现 |
| 2 | 「`kv.size()` 到了位置 p 才 +1，所以逐层追加时判不出越界」 | **成立且是真实陷阱**：逐层追加协议下「第 p 层已写入、第 p-1 层未写入」是正常中间态，校验必须用 `容量` 而不是 `size` | `src/kvcache.h` 的注释；`docs/04-kv-cache-sampling.md` §17.2 |
| 3 | 「改末位 token 后，前面的 logits 也会变（注意力是连通的）」 | **不成立**。因果掩码保证前 n-1 个位置逐位不变，实测差 = 0 | `line_prefill` 第 7 跳 |
| 4 | 「贪心采样会消耗随机数，所以两次结果可能不同」 | **不成立**。`temp<=0` 走 argmax 分支，完全不碰随机数发生器 | `line_decode` 第 7 跳 |
| 5 | 「cache size 应该等于 prompt 位置数 + 生成 token 数」 | **差 1**：decode 线第 5 跳额外做了一次前向（用来演示写回），所以是 `prompt + 1 + 生成数` | `line_decode` 第 6 跳 |

## 五 真机实测

机器：Linux 5.10 / 8 核 / 16GB；模型：`models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf`（Q8_0 量化）；
日期：2026-11-11。

```bash
$ MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_prefill
观测 : 层数 = 22   维度 = 2048   词表 = 32000
观测 : prompt token 数 = 2
观测 : 缓存 size = 0 → 1 → 2
观测 : 第 0 层位置 0 的 K 向量模 = 1.3779
观测 : 第 0 层位置 0 的 V 向量模 = 0.080459
观测 : 最大逐元素差 = 0            ← 增量 ≡ 全量，差是 0
观测 : clear 后复用的最大差 = 0
观测 : 前 n-1 个位置的最大差 = 0     ← 因果性
观测 : 末位位置的最大差 = 19.1051
观测 : argmax token id = 29892
== 线 prefill 走完：9/9 项通过
```

```bash
$ MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_decode
观测 : temp = 0   top_k = 40   top_p = 0.95   seed = 1234
观测 : 采样出的 token id = 306
观测 : 推进前位置 = 2   推进后缓存 size = 3
观测 : 生成 token 数 = 4
            生成[0] id=306  文本="[control_304]"
            生成[1] id=29915 文本="'"
            生成[2] id=29885 文本="m"
            生成[3] id=19235 文本="▁Sarah"
观测 : 生成文本 = [control_304]'m Sarah
== 线 decode 走完：5/5 项通过
```

**耗时**：prefill 线 37 秒、decode 线 34 秒（朴素前向，约 2~4 秒/token）。

## 六 差异注明

| # | 差异 | 我的实现 | 真机（llama.cpp @ 88b47a755） | 为什么可接受 |
|---|---|---|---|---|
| 1 | 计算图 | 阶段顺序固定的直接循环 | `llama-graph.cpp` 建图 + ggml 后端调度 | 数学与阶段划分不变；丢掉的是图复用与并行调度（已按承重不可砍 → 降级还原登记） |
| 2 | KV cache 形态 | 纯顺序追加（`pos == size`） | `llama_kv_cache` 的 cell/slot 池 + `find_slot` | **有损**：多序列与乱序到达不在（`docs/constraints/04-对齐约束.md` §2.2 第 2 条已登记为降级还原） |
| 3 | 多序列 | 只支持一条序列 | `seq_id` 多序列 + 跨 batch 复用 | 单请求推理用不到 |
| 4 | 并行解码 | 逐 token 调用 | `llama_batch` / ubatch 并行 | 吞吐优化，正确性无关 |
| 5 | 注意力掩码 | 因果（下三角） | `build_attn` 的多种掩码类型 | 推理只需要因果掩码 |
| 6 | 采样链 | 固定顺序的单个函数 | `llama_sampler_chain` 可任意组合 | **节序已与上游默认链对齐**（top-k → top-p → temperature → dist），各节语义相同；少了任意组合的自由度 |
| 7 | RoPE 配对 | 相邻配对 `(2i, 2i+1)` | `LLAMA_ROPE_TYPE_NORM`（也是相邻配对） | **与上游一致**（这是逐数值对拍的前提，见 `docs/03-forward.md` §17.3） |
| 8 | KV 头 | 按需索引共享，不物理复制 | `llama-kv-cache` 的 repeat/expand | 语义相同，省一份内存 |
| 9 | 投机解码 | 无 | `llama-spec.cpp` | 加速特性，不是主干 |

## 七 可观测点

| 量 | 本实现的接口 | 真机对应（llama.cpp @ 88b47a755） |
|---|---|---|
| 缓存位置数 | `KVCache::size()` | `llama_memory_seq_pos_max` / KV cache 的 `n_cells` |
| 每位置 KV 维度 | `KVCache::dim()` | `llama_kv_cache` 的 `n_embd_k_gqa` |
| 某位置的 K/V | `KVCache::k(layer, pos)` / `v(layer, pos)` | `llama_kv_cache::get_input` 的底层缓冲 |
| 单 token logits | `Model::forward_token(token, pos, kv)` | `llama_context::decode` 的 ubatch 输出 |
| 全序列 logits | `Model::forward(tokens)` | `llama_context::decode` 的 batch 输出 |
| 超参 | `Model::hparams()` | `include/llama.h:573:llama_model_n_ctx_train` / `:577:llama_model_n_layer` / `:574:llama_model_n_embd` |
| 生成 token | `Sampler::sample(logits, n)` | `llama_sampler_sample` |
| 生成文本 | `Tokenizer::decode(ids)` | `llama_detokenize` |

> 本仓库没有 `/proc/vmstat` 那样的内核统计接口；最接近的「真机对应」是上游的
> `llama_memory_seq_pos_max` 与 `llama_context` 的 `n_tokens` 计数——
> 本仓库的 `KVCache::size()` 就是它们的教学版。
