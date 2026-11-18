# text 族 —— 文本域

> 文本 ↔ token 序列。**正变换**（tokenize）与**逆变换**（detokenize）是两条独立的线，
> 因为它们的失败方式完全不同：正变换错在「切错」，逆变换错在「渲染错」。

## 一 链路图

```text
   "Hello world 2026, 你好！"
              │
              ▼
   ┌──────────────────────┐
   │ 归一化                │  src/tokenizer.cpp
   │ SPM: 句首补空格 + ▁   │  ★ 跨层点：文本形态变了
   │ BPE: bytes_to_unicode │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ 切词                  │  src/unicode.cpp（预分词扫描器）
   │ BPE: 正则分支         │
   │ SPM: 整段不切          │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ 合并核心              │  src/tokenizer.cpp
   │ symbol 链表           │  ★ 跨层点：字符 → 词表条目
   │ 按 score 优先合并      │
   │ 字节兜底 → BYTE token │
   └──────────┬───────────┘
              ▼
        token id 序列 [1, 15043, 8879, …]
              │
              ▼  （detokenize 线反向走）
   ┌──────────────────────┐
   │ 按 token 类型决定      │  src/tokenizer.cpp
   │ 渲不渲染              │
   │ BYTE token → 原始字节 │
   │ ▁ → 空格             │
   └──────────┬───────────┘
              ▼
        " Hello world 2026, 你好！"
```

★ 跨层点三处：**归一化**（文本形态变了）、**合并核心**（字符 → 词表条目）、
**按类型渲染**（token → 字符）。

## 二 逐跳对照

### 线 A：`tokenize`（7 跳）

| 跳 | 本仓库落点 | 上游对照（llama.cpp @ 88b47a755） | 真机命令 |
|---|---|---|---|
| 1 | `GgufFile::load` → `Tokenizer::load` | `src/llama-vocab.cpp:1921:llama_vocab::impl::load` | `MINILLAMA_MODEL=… ./build/line_tokenize` |
| 2 | `Tokenizer::encode` 的 BOS 段 | `src/llama-vocab.cpp:569:llm_tokenizer::append_bos` | 同上 |
| 3 | `Tokenizer::encode` 的 work 准备段（▁ 转义 / bytes_to_unicode） | `src/llama-vocab.cpp:3285:llama_escape_whitespace` + `:3869:unicode_byte_to_utf8` | 同上 |
| 4 | `unicode.cpp:pre_tokenize`（BPE 走扫描器，SPM 整段不切） | `src/llama-vocab.cpp:605:unicode_regex_split` | 同上 |
| 5 | `Tokenizer::merge_tokenize`（symbol 链表 + 按 score 优先的相邻合并） | `src/llama-vocab.cpp:3358:llm_tokenizer_spm_session::tokenize` / `:3435:llm_tokenizer_bpe_session::tokenize` | 同上 |
| 6 | `Tokenizer::byte_token` 三级查找链（残段逐字节落 BYTE token） | `src/llama-vocab.cpp:3869` 的 `token_to_id.at(unicode_byte_to_utf8(ch))` | 同上 |
| 7 | `Tokenizer::decode`（往返对照） | `src/llama-vocab.cpp:3640:llama_vocab::impl::detokenize` | 同上 |

### 线 B：`detokenize`（6 跳）

| 跳 | 本仓库落点 | 上游对照（llama.cpp @ 88b47a755） | 真机命令 |
|---|---|---|---|
| 1 | `GgufFile::load` → `Tokenizer::load` | `src/llama-vocab.cpp:1921:llama_vocab::impl::load` | `MINILLAMA_MODEL=… ./build/line_detokenize` |
| 2 | `Tokenizer::encode`（取合法 id 序列） | `src/llama-vocab.cpp:3358:llm_tokenizer_spm_session::tokenize` | 同上 |
| 3 | `Tokenizer::decode` 的类型判定（CONTROL / UNKNOWN 跳过，USER_DEFINED 照原样输出） | `src/llama-vocab.cpp:3640:llama_vocab::impl::detokenize` → `:3519:llama_vocab::impl::token_to_piece` | 同上 |
| 4 | `Tokenizer::decode` 的 BYTE 分支 | `src/llama-vocab.cpp:3089:llama_vocab::impl::token_to_byte` | 同上 |
| 5 | `Tokenizer::decode` 尾部的 `replace_all`（▁ → 空格） | `src/llama-vocab.cpp:3289:llama_unescape_whitespace` | 同上 |
| 6 | `Tokenizer::decode`（消费方） | `src/main.cpp` 的生成循环 | `./build/mini-llama --model … --prompt "Hi" --n 8` |

## 三 学习问题的答案

> **为什么 SPM 和 BPE 两种分词器要共用同一个「按 score 合并」的核心，
> 而真实版把它们写成了两份几乎一样的代码？**

因为**两者在「词内合并」这一步上是同一套算法**，差别只在**合并的边界**：

| | SPM（`model="llama"`） | BPE（`model="gpt2"`） |
|---|---|---|
| 切词 | **不切**，整段文本作为一条待合并链 | 先按预分词规则切成「词」，**词内**才合并 |
| 合并 | 整段内的相邻 symbol 按 score 合并 | 每个词内相邻 symbol 按 score 合并，**词间不合并** |
| 空白 | 句首补空格，空格转义成 `▁` | 原始字节 → bytes_to_unicode 映射 |

「按 score 优先的相邻合并」这套机制本身是**词表训练时学到的规则在推理侧的应用**：
词表里每个多字符 token 的 score 就是它参与合并的优先级。所以两边共用同一个
`symbol 链表 + 按 score 最大堆 + 陈旧条目校验` 的核心是自然的。

llama.cpp 之所以写成两份，是因为它们是**在不同时间、为不同模型家族分别加进去的**
（SPM 支持 Llama-1/2，Qwen2 的 BPE 支持后来才加），改造成本低于抽象成本。
本仓库把两份合成一份之后，对照反而更清楚：**一眼能看出「BPE = SPM + 一个预分词切词步骤」**。

> 教学点：合并不是「查词表」，而是**在词表约束下做贪心合并**——
> 一个字符串可能被切成很多种方式，score 决定优先选哪种。

## 四 认知纠错记录

| # | 曾经的错误假设 | 实际情况 | 判定依据 |
|---|---|---|---|
| 1 | 「EOS 是 CONTROL 类 token，解码时会被跳过」 | **不成立**。TinyLlama 把 `</s>` 标成 `USER_DEFINED(4)`，会被**原样渲染** | `line_detokenize` 第 3 跳的类型直方图；llama.cpp 的 `attr_special = UNKNOWN\|CONTROL` 只跳过这两类 |
| 2 | 「所有词表都有 BYTE token」 | **不成立**。llama3 词表没有显式 BYTE token，字节兜底要退化为「bytes_to_unicode 映射字符本身查词表」 | `src/tokenizer.cpp:Tokenizer::byte_token` 的三级查找链 |
| 3 | 「SPM 的句首空格由 `add_special` 控制」 | **不成立**。句首补空格与 `add_special` 无关，是词表类型的历史规范 | `docs/02-tokenizer.md` §17.4 |
| 4 | 「预分词会吞掉空白」 | **不成立**。llama3 模式下空白单独成片段（`" world"`、`" "`），拼接长度 == 原文本长度 27 | `line_tokenize` 第 4 跳 |
| 5 | 「数字会按单个字符切」 | **不成立**。llama3 模式按 **1~3 位**分组（`"202"` + `"6"`），qwen2 模式才是单个数字 | `src/unicode.cpp` 的 `PreTokenizer` 分派 |

## 五 真机实测

机器：Linux 5.10 / 8 核 / 16GB；模型：`models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf`（SPM 词表）；
日期：2026-11-11。

```bash
$ MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_tokenize
观测 : 词表类型 = llama
观测 : 词表大小 = 32000
观测 : 样本 = Hello world 2026, 你好！
观测 : add_bos = 1        观测 : 首 token id = 1
            片段[0] = "Hello"
            片段[1] = " world"
            片段[2] = " "
            片段[3] = "202"          ← llama3：数字按 1~3 位分组
            片段[4] = "6"
            片段[5] = ","
            片段[6] = " 你好"
            片段[7] = "！"
            片段拼接长度 27 / 原文本长度 27
观测 : 编码后 token 数 = 15
            token[0] id=1     文本="<s>"      ← BOS
            token[1] id=15043 文本="▁Hello"
            token[2] id=8879  文本="▁wo"
            token[3] id=2096  文本="rl"
            token[4] id=29881 文本="d"
            token[5] id=29871 文本="▁"
            token[6] id=29906 文本="2"
            …
观测 : 解码回来 = <s> Hello world 2026, 你好！
== 线 tokenize 走完：3/3 项通过
```

```bash
$ MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_detokenize
            类型 1 × 31229（1=NORMAL 3=CONTROL 4=USER_DEFINED 6=BYTE）
            类型 4 × 771
            本模型的 EOS 标成 **USER_DEFINED(4)** 而不是 CONTROL(3)——所以它会被原样渲染
观测 : 解码结果 =  Hello world 2026, 你好！
== 线 detokenize 走完：4/4 项通过
```

> 注意：SPM 词表解码回来**带句首空格**（`" Hello…"`），这不是 bug，是 SPM 的规范行为。

## 六 差异注明

| # | 差异 | 我的实现 | 真机（llama.cpp @ 88b47a755） | 为什么可接受 |
|---|---|---|---|---|
| 1 | 分词器家族 | SPM + BPE 两套 | 五套（SPM / BPE / UGM / WPM / RWKV） | 其余三套是同一套词表结构的不同变体；Llama 全系只用到前两套 |
| 2 | 预分词引擎 | 手写 6 分支扫描器 | 通用正则引擎 `unicode_regex_split` | 模式固定 → 手写更快、零依赖、可逐分支注释；**切分语义等价** |
| 3 | Unicode 类别 | 只生成 `\p{L}` / `\p{N}` 两类 | `unicode-data.cpp` 全类别 | 这两类是 llama3/qwen2 两个模式的全部需求 |
| 4 | 归一化 | 不做 NFKC / 大小写折叠 | `unicode_regex_split` 的归一化步骤 | 这两个预分词器的正则里是显式 `[sS]`，用不到归一化 |
| 5 | special token 片段 | 不从正文里切出 `<\|end_of_text\|>` | `tokenizer_st_partition` | 处理「文本里内嵌特殊串」；教学主干可后补 |
| 6 | BYTE 文本格式 | 三种都认（`<0xXX>` / 单字节 / 映射字符） | 同样三种（`unicode_byte_to_utf8` 的产物） | **完全一致** |
| 7 | 合并核心 | SPM / BPE 共用一份 | 两份几乎相同的实现 | 结构同构；合一份之后「BPE = SPM + 切词」更直观 |
| 8 | 控制 token 渲染 | 跳过 `CONTROL`/`UNKNOWN`/`UNUSED`/`UNDEFINED` | 跳过 `attr_special`（= UNKNOWN\|CONTROL），`USER_DEFINED` 照原样输出 | **行为一致**（`UNUSED`/`UNDEFINED` 在真机里也不渲染） |

## 七 可观测点

| 量 | 本实现的接口 | 真机对应（llama.cpp @ 88b47a755） |
|---|---|---|
| 词表类型 | `GgufFile::get_string("tokenizer.ggml.model")` | `include/llama.h:593:llama_vocab_type` / `:595:llama_vocab_n_tokens` |
| 词表大小 | `Tokenizer::n_tokens()` | `llama_vocab_n_tokens(vocab)` |
| BOS / EOS | `Tokenizer::bos_id()` / `eos_id()` | `llama_vocab_bos` / `llama_vocab_eos` |
| 是否加 BOS | `Tokenizer::add_bos()` | `tokenizer.ggml.add_bos` 元数据 |
| token 文本 | `Tokenizer::token_text(id)` | `llama_token_get_text(vocab, id)` |
| 预分词结果 | `unicode.cpp:pre_tokenize` | `llama_vocab::impl::tokenizer_st_partition` + 正则引擎 |
| 编码结果 | `Tokenizer::encode(text, add_special)` | `llama_tokenize(vocab, text, …)` |
| 解码结果 | `Tokenizer::decode(ids)` | `llama_detokenize(vocab, ids, …)` / `llama_token_to_piece` |
| 字节映射 | `unicode.cpp:byte_to_unicode_cp` | `src/unicode.cpp:unicode_byte_to_utf8` |
