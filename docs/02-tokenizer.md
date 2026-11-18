# 技术文档：单元 2 分词器 —— 从字节序列到 token 序列

- 读者：要复刻 llama.cpp 词表层（vocab / tokenizer）的人，以及做 mini-llama 单元 2 验收的人
- 审法：先读 §16.3 的承重结论定「哪些能砍」，再读 §17 的四个坑，最后拿 §5 的验收数字复跑
- 名词版本：v1

## 变更记录

| 日期 | 版本 | 改了哪条 |
|---|---|---|
| 2026-11-11 | v1 | 从旧五段式迁到 mini-restore v1.2 骨架（§13~§17 补齐） |

> 参考实现 = llama.cpp b10143（commit 88b47a755，2026-07-27，master 分支）
> 来源     = /root/ai/llama.cpp @ 88b47a755
> 技能版本 = mini-restore v1.2

## 第一部分：概要设计

### §1 概念

#### 1.1 实体

| 实体 | 业务含义 | 关键字段（谁写 / 谁读） | 空值语义 |
|---|---|---|---|
| `Tokenizer`（src/tokenizer.h） | 一个 GGUF 词表在运行期的视图：既持有词表，又是编码/解码入口；唯一外部输入是单元 1 的 `GgufFile`（src/gguf.h），本单元只 `find` / `get_*` | `tokens_`、`text_to_id_`、`type_`、`pre_` 由 `Tokenizer::load` 写；`encode`/`decode` 只读 | `bos_id_` / `eos_id_` 为 -1 表示该词表没有这个特殊 token |
| `byte_to_id_[256]` / `id_to_byte_`（src/tokenizer.h） | 原始字节 ↔ BYTE token id 的双向兜底通道 | load 写；encode 的兜底分支与 decode 的 BYTE 分支读 | `has_byte_map_ == false` 表示词表没有 BYTE token，兜底退化为「按文本形状猜」 |
| `TokenData`（src/tokenizer.h） | 词表里的一条 token：文本、合并分数、类型 | `text` / `score` / `type` 三字段按 **token id 同序** 从 GGUF 三个数组读出 | `score` 缺失取 0.0f；`type` 缺失按「文本长度 == 1 即 BYTE」猜 |

#### 1.2 枚举值对象

| 枚举 | 取值（穷举） | 对齐上游 |
|---|---|---|
| `TokenType`（src/tokenizer.h） | UNDEFINED=0, NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, UNUSED=5, BYTE=6 | `enum llama_token_type`，include/llama.h:91，逐值同 |
| `VocabType`（src/tokenizer.h） | SPM, BPE | `LLAMA_VOCAB_TYPE_SPM` / `_BPE`，include/llama.h:72 的子集 |
| `PreTokenizer`（src/unicode.h） | LLAMA3, QWEN2 | `LLAMA_VOCAB_PRE_TYPE_LLAMA3` / `_QWEN2`，llama-vocab.cpp:283 / :373 |

#### 1.3 名词

| 名词 | 一句大白话（术语沿用上游叫法） | 名词 | 一句大白话 |
|---|---|---|---|
| vocab | 词表：token 文本 ↔ id 的对照表，从 `tokenizer.ggml.tokens` 读出 | token | 词表里的一项，id 就是它在数组里的下标 |
| score | token 的合并优先级；越大越先合并，等于 BPE 训练学到的合并规则在推理侧的应用 | merge | 把两个相邻 symbol 合成一个 token 的动作，按 score 从大到小贪心执行 |
| BYTE token | 词表里代表单个原始字节的兜底 token，让任意二进制输入都能被编码 | SPM | sentencepiece 风格词表（`model = "llama"`），空格用 `▁` 转义 |
| BPE | byte-level BPE 词表（`model = "gpt2"`），先预分词、再词内合并 | pre-tokenizer | 把整段文本切成「词」的组件；Llama-3 系与 Qwen2 系是同一套 6 分支模式，只有数字分组不同 |
| bytes_to_unicode | GPT-2 的字节→可见字符映射：0~255 映射到 256 个 Unicode 码点，使词表只含文本 token | `▁`（U+2581） | SPM 词表里代表空格的转义符；编码时 `' '` → `▁`，解码时反向还原 |

### §2 正常流程

```text
[读 GGUF 元数据] → [判词表类型] → [建反向索引 + 字节映射] → [改写文本] → [切词] → [按 score 合并] → [残段落 BYTE token] → [拼 BOS/EOS]
```

1. 读 `tokenizer.ggml.model` 判词表类型：`llama` → SPM，`gpt2` → BPE；BPE 路径再读 `tokenizer.ggml.pre` 判预分词器：`llama-bpe` / `llama3` → LLAMA3 模式，`qwen2` → QWEN2 模式。【为什么】类型决定后面每个分支（补不补空格、转不转义、做不做字节映射、要不要预分词），必须最先判；两套预分词模式的 6 个分支里只有「数字分组」不同，模式选错会切错词边界。
2. 按 token id 同序读 `tokenizer.ggml.tokens` / `scores` / `token_type` 三个数组，逐条拼成 `TokenData`；再建 `text_to_id_` 反向索引；再扫一遍 BYTE token，建 `byte_to_id_` / `id_to_byte_` 双向映射。【为什么】合并循环要反复查「这两个 symbol 拼起来是不是词表里的 token」，反向索引是热路径。
3. 文本改写：SPM 句首补空格 → 空格转 `▁`（SPM 词表）→ 原始字节映射到 bytes_to_unicode 可见字符（BPE 词表）。
4. SPM：整段文本直接进合并核心；BPE：先按 6 分支扫描器切词，再**逐词**进合并核心。【为什么】BPE 词间不合并，词界由 pre-tokenizer 定。
5. 合并核心：把文本切成 UTF-8 字符级 symbol 链表，把所有「相邻 symbol 拼起来在词表里」的候选按 score 放进最大堆，反复取最高分合并，直到堆空。
6. 收尾时逐个 symbol 查词表；查不到的按 UTF-8 字符逐字节落到 BYTE token。【为什么】词表覆盖不到的字节必须有去处，否则整段文本会丢字符。首尾按 `add_special` 与词表开关拼 BOS / EOS。

### §3 异常处理

| # | 异常 | 如何处理 |
|---|---|---|
| 1 | GGUF 缺 `tokenizer.ggml.model` | `get_string` 抛 std::runtime_error；**不可重试**；`Tokenizer::load` 中止，调用方 src/main.cpp 的整条推理链退出 |
| 2 | `model` 不是 llama/gpt2（如 rwkv、bert），或 BPE 路径的 `pre` 不是 llama-bpe/llama3/qwen2 | 分别抛「不支持的词表类型 tokenizer.ggml.model: ...」与「不支持的预分词器 tokenizer.ggml.pre: ...」；**不可重试**；GGUF 句柄随栈展开释放 |
| 3 | 缺 `tokenizer.ggml.tokens`，或它是空数组 | `get_array` 抛 / 抛「tokenizer.ggml.tokens 为空」；**不可重试** |
| 4 | `scores` / `token_type` 数组长度小于 `tokens` | **当前实现不校验长度**，`(*scores)[i]` 越界读（未定义行为）；**不可重试**。上游同位置抛「Index out of array bounds for scores」（llama-vocab.cpp:2415 / :2425）—— 偏离点见 §16.2 |
| 5 | 词表缺 `tokenizer.ggml.token_type` / `bos_token_id` / `eos_token_id` | 不抛错：类型按「文本长度 == 1 即 BYTE」猜，id 保持 -1（encode 不加 BOS/EOS）；**可重试**（换带这些 key 的词表即恢复） |
| 6 | 字节兜底时词表里没有对应 BYTE token | 该字节被静默丢弃（src/tokenizer.cpp 只在 `bid >= 0` 时 push）；**不可重试**（词表本身缺通道）；decode 侧同样跳过 |

### §4 Out of Scope

| 砍掉的能力 | 为什么可以砍 | 承重判断（见 §16.3） |
|---|---|---|
| UGM / WPM / RWKV / PLaMo2 四套 tokenizer | 教学目标是 Llama 全系主干（SPM + BPE）；四套各自的算法都不在 Llama 链路上 | 第 1 行：可砍 |
| special token 片段提取（`tokenizer_st_partition`） | 文本内嵌 `<\|end_of_text\|>` 这类串在 Llama 主干 prompt 里不出现 | 第 2 行：可砍 |
| 通用正则引擎（std::regex / std::wregex 兜底路径） | 上游 @ 88b47a755 对 llama3 / qwen2 这两条模式**自己也是手写扫描器**，`std::regex` 只给没被特判的模式兜底 | 第 3 行：可砍 |
| 全量 Unicode 类别表 + NFD 归一化 | 上游的 NFD 归一化只被 WPM 用（llama-vocab.cpp:821）；SPM/BPE 的 6 分支模式只需要字母类与数字类 | 第 4 行：可砍 |
| `ignore_merges` / double-bos 告警 / unused token 抑制等 flag | 边界行为微调，不改编码主流程的 token 序列 | 第 5 行：可砍（有已知差异，见 §16.3 该行「行为测试」列） |
| tiktoken 风格的离线批量 token 化接口、token attr 查询 API | 性能 / 易用性接口，编码结果不受影响 | 第 6 行：可砍 |

### §5 观测点

本实现当前**未落任何日志**，下表是排障时约定的检索词与字段，以及在现有代码里取到等价信息的位置。

| 时机 | 打的键、携带的字段 | 当前实现的对应事实 |
|---|---|---|
| 词表加载完成 / 编码入口 | `vocab`：n_tokens、bos_id、eos_id、词表类型、预分词器；`tokenize`：文本长度、add_special、escape_whitespaces、byte_encode | `Tokenizer::bos_id()` / `eos_id()` / `n_tokens()`（src/tokenizer.h）；三个开关是 `Tokenizer` 私有成员（src/tokenizer.h） |
| 字节兜底命中 / 落空 / 解码完成 | `byte_fallback`：原始字节（十六进制）、命中的 BYTE token id；`byte_fallback_miss`：原始字节（十六进制）；`detokenize`：token 数、输出字节数 | `Tokenizer::byte_token`（src/tokenizer.cpp）返回点；src/tokenizer.cpp 判 `bid >= 0` 为假即静默丢弃，这是唯一线索；`Tokenizer::decode`（src/tokenizer.cpp）返回点 |

**验收记录（旧 §⑤，数字原样保留）**

| 词表 | token 数 | 往返结果 |
|---|---|---|
| llama-bpe | 128256 | `"Hello world"` → 3 tokens（含 BOS）→ 完整还原 |
| llama-spm | 32000 | → `" Hello world"`（句首空格是 SPM 规范行为） |
| qwen2 | 151936 | → 2 tokens → 完整还原 |

合成用例 5 组全绿：预分词扫描器（含切分不变式 fuzz）、SPM 合并优先级 / 字节兜底、qwen2 映射词表、llama-bpe BOS+映射合并、缺 key / 类型错误。本仓复跑：`./build/test_tokenizer` → `passed=5 failed=0`；真实词表冒烟用环境变量 `TEST_TOKENIZER_REAL_FILE=<path>` 挂载。**本仓自带词表的实测现状**：`models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf`（n_tokens=32000，bos=1，eos=2）的 `tokenizer.ggml.token_type` 里 `<s>` 是 4（USER_DEFINED），本实现与上游 `llama_vocab::impl::token_to_piece`（llama-vocab.cpp:3519，跳过掩码只有 UNKNOWN|CONTROL）一样会把它渲染出来：`encode("Hello world")` 得 5 tokens，`decode` 得 `"<s> Hello world"`。tests/test_tokenizer.cpp:315 的断言只接受 `"Hello world"` / `" Hello world"`，故该断言在此文件上失败——是断言过窄，不是实现偏离上游。

## 第二部分：详细设计

### §6 整体流程图

```text
[GgufFile 解析] →(原始 KV)[判词表类型 + 判预分词器] →(词表三数组)[建反向索引 + 字节映射]
                                                              │ Tokenizer
                        ┌─────────────────────────────────────┴──────────────────────────┐
                        ▼                                                                ▼
        [SPM 整段 / BPE 切词后逐词] →(symbol 链表)[按 score 合并 → 残段落 BYTE]   [加 BOS → 文本改写]
                                                                                      │
                                                                    token id 序列 ←───┘ → [加 EOS → 输出]
```

失败出口：判类型 / 判预分词器 / 读 tokens 三处抛异常（§3 的 1~3），流程在「建索引」之前终止。

### §7 子流程图 + 业务说明

#### 7.1 词表加载

流程：`model → 类型`；`pre → 预分词器`；`tokens/scores/token_type → TokenData[] → text_to_id_ → byte_to_id_/id_to_byte_ → bos_id/eos_id/开关`
1. 拿 `tokenizer.ggml.model` 决定走哪条编码路径，同时定下这一族的默认开关：SPM 默认句首补空格、加 BOS；BPE 默认不做空格转义。若是 BPE，再拿 `tokenizer.ggml.pre` 决定预分词模式，并据此定下「是否按 bytes_to_unicode 映射字节」「是否默认加 BOS」。
2. 三个数组按 id 同序读出 `TokenData` 的 text / score / type；`token_type` 缺失时按文本长度兜底猜。
3. 建 `text_to_id_`：文本重复时保留**首个** id（比上游保守，上游用 `[]` 覆盖成最后一个；真实词表本身无重复，不影响结果）。
4. 扫一遍 BYTE token，把三种文本格式都识别成原始字节，建 `byte_to_id_` / `id_to_byte_`；再读 `bos_token_id` / `eos_token_id`（缺失留 -1）与 `add_bos` / `add_eos` 覆盖开关（缺失保留第 1 步定下的默认值）。

#### 7.2 文本改写与预分词

流程：`原文 → [补空格] → [空格转 ▁] → [字节映射到可见字符] → [切词] → 逐词`
1. SPM 家族在句首补一个空格。【为什么】SPM 词表用 `▁` 标记词边界，不补空格首词和后续词无法区分——这条与 `add_special` 无关，见 §17.4。
2. SPM 家族把空格换成 `▁`（U+2581）。BPE 家族不做这步，它的词表直接含原始空格；而 BPE 家族把每个原始字节映射成 bytes_to_unicode 的可见字符，空格因此变成 `Ġ`（U+0120）。【为什么】BPE 词表只含可打印 token，二进制输入靠这层映射兜住，见 §17.1。
3. BPE 家族切词：6 个分支按声明顺序试，先命中先走——英文缩略、可选前缀+字母串、数字、可选空格+标点串+尾随换行、空白+换行、空白兜底；字母/数字判定走 Unicode 类别区间表的二分查找（`in_ranges`，src/unicode.cpp）。其中空白分支有贪心回溯（等价于上游正则「空白串后面还跟非空白时不吃掉最后一个空格」的否定前瞻写法）：段空白后面还跟非空白时，只吃掉「除最后一个空格以外」的空白，把最后一个空格留给后一个词当前缀。`"  word"` 因此切成 `[" ", " word"]` 而不是 `["  ", "word"]`。SPM 家族不切词，整段进 7.3。

#### 7.3 按 score 合并（SPM / BPE 共享核心）

流程：`切字符 → 邻接对入堆(按 score) → 取最高分合并 → 重挂邻接对 → 直到堆空`
1. 把文本按 UTF-8 字符切成 symbol，用「数组 + prev/next 下标」模拟双向链表；合并只改下标和长度，物理字符串永不移动。再把每一对相邻 symbol 拼起来查词表：查得到就以该 token 的 score 入最大堆，查不到就不入。
2. 反复取堆顶（score 最大）合并：右 symbol 并入左 symbol，右 symbol 摘链，然后重新检查左右两个新的邻接对。
3. 堆里的条目可能在弹出前就失效（相邻 symbol 已被别的合并吞掉），所以弹出时必须校验「两段当前长度之和 == 入队时记的长度」，不等就跳过；同分时按入队序号决定先后，保证结果确定可复现。

#### 7.4 字节兜底与解码

流程：`symbol 查不到词表 → 逐 UTF-8 字符拆成原始字节 → byte_token() 三级查找`
1. 合并结束后沿链表输出；每个 symbol 先按整段文本查词表，查到即输出该 id；查不到的 symbol 按 UTF-8 字符逐个拆成原始字节（BPE 家族把可见字符反查回字节，SPM 家族直接取单字节）。
2. 每个字节走三级查找：先查显式 BYTE token 映射；没有就查 bytes_to_unicode 映射字符本身在不在词表里（llama3 式词表没有显式 BYTE token）；再查单字节原始文本。
3. 三级都查不到就丢弃该字节（无异常、无日志，见 §5 的 `byte_fallback_miss`）；解码方向：跳过 CONTROL / UNKNOWN / UNUSED / UNDEFINED，渲染 NORMAL / USER_DEFINED；BYTE token 还原成原始字节；BPE 家族把可见字符反查回字节；SPM 家族把 `▁` 换回空格。

### §8 异常流转表

| 异常 | 触发点 | 状态/字段变化 | 恢复路径 |
|---|---|---|---|
| 词表类型 / 预分词器不支持 | `Tokenizer::load` 判 `model` 与判 `pre`（两段都在 src/tokenizer.cpp 的 load 里） | 抛异常前 `type_` 仍是默认值 SPM、`pre_` 仍是默认 LLAMA3，无字段被写坏 | 换用 `model` 为 llama/gpt2、`pre` 为 llama-bpe/llama3/qwen2 的词表后重新加载；本进程内不可恢复（对象未构造成功） |
| 词表数组为空 / 缺失，或 scores 数组偏短 | `Tokenizer::load` 读 `tokens` 与 `scores`（两段都在 src/tokenizer.cpp 的 load 里） | 前者 `tokens_` 为空、`text_to_id_` 未建；后者已读入的 `TokenData::score` 里越界部分是未定义值，词表状态不可信 | 换词表；本进程内不可恢复；**当前实现未加长度校验**，属已知缺陷（§16.2） |
| 字节兜底落空 | `Tokenizer::merge_tokenize` 收尾（src/tokenizer.cpp） | 该字节不进入输出，其余 token 不受影响 | 换带 BYTE token 的词表；文本内容会有丢失，不可静默忽略 |
| 解码遇到越界 id | `Tokenizer::decode`（src/tokenizer.cpp） | 越界 id 被 `continue` 跳过，其余 token 照常渲染 | 无需处理，调用方不应传越界 id |

### §9 数据形状

```jsonc
// 上半：GGUF 里本单元读到的 key（单元 1 解析出的原始 KV）；下半：内存里的词表视图（src/tokenizer.h）
{
  "tokenizer.ggml.model":       "llama",               // string，必填："llama"=SPM / "gpt2"=BPE
  "tokenizer.ggml.pre":         "default",             // string，BPE 路径必填："llama-bpe"|"llama3"|"qwen2"
  "tokenizer.ggml.tokens":      ["<unk>", "<s>", "…"],  // array<string>，必填；token id = 数组下标
  "tokenizer.ggml.scores":      [0.0, 0.0, -1000.0],   // array<float32>，可缺（缺则全 0.0）
  "tokenizer.ggml.token_type":  [4, 4, 1],             // array<int32>，可缺（缺则按长度猜）
  "tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 2,  // uint32，可缺（缺则 -1 = 无）
  "tokenizer.ggml.add_bos":     true,                  // bool，可缺（本实现读这个名字，上游是 add_bos_token）

  "tokens_": [ { "text": "<s>", "score": 0.0, "type": "USER_DEFINED" },   // id = 下标
                { "text": "▁", "score": -1.0, "type": "NORMAL" } ],
  "text_to_id_": { "<s>": 0, "▁": 1 },         // 文本 -> id；重复文本保留首个
  "byte_to_id_": [ -1, -1, 3, 4 ],             // 原始字节 -> BYTE token id；-1 = 无
  "id_to_byte_": [ -1, -1, -1, 0, 1 ],         // BYTE token id -> 原始字节；-1 = 非 BYTE
  "has_byte_map_": true,                      // 词表里到底有没有 BYTE token
  "type_": "SPM", "pre_": "LLAMA3",           // VocabType / PreTokenizer（pre 仅 BPE 路径有意义）
  "bos_id_": 0, "eos_id_": -1,                // -1 = 该词表没有这个特殊 token
  "add_bos_": true, "add_eos_": false, "escape_whitespaces_": true,   // ' ' <-> ▁(U+2581)
  "add_space_prefix_": true,                  // SPM 句首补空格
  "byte_encode_": false                       // GPT-2 bytes_to_unicode
}
```

## 开发流水线扩展节

### §13 编排锚点与挂载点

| 角色 | 路径 | 承载什么 |
|---|---|---|
| 词表加载入口 | src/tokenizer.cpp `Tokenizer::load` | 判类型与预分词器、读三个数组、建反向索引与字节双向映射、定特殊 token 与开关 |
| 编码入口 | src/tokenizer.cpp `Tokenizer::encode` | BOS/EOS + 文本改写 + 分派 SPM/BPE |
| 合并核心 / 预分词扫描器 | src/tokenizer.cpp `Tokenizer::merge_tokenize`；src/unicode.cpp `pre_tokenize` | symbol 链表 + score 最大堆 + 陈旧条目校验 + 字节兜底；6 分支确定性切词（SPM 路径不经过它） |
| 编排挂载点 | src/main.cpp / :66 / :78 | 加载词表 → 把 prompt 编成 token 序列 → 回显与逐 token 解码 |

**删掉锚点流程还跑不跑得起来**：删掉 `Tokenizer::load` 的判类型分支，BPE 词表会被当成 SPM 走（补空格 + 转义），编码结果全错但不会崩——即「跑得起来但结果错」，是最坏的一种。删掉 `merge_tokenize` 则整条链断，src/main.cpp 的 encode 直接空转。

挂载点表：

| 文件:行 | 改什么 | 为什么必须在这儿 |
|---|---|---|
| src/tokenizer.cpp / :71 / :96-109 / :120-129 / :132-144 | 读 `tokenizer.ggml.model` 判词表类型；读 `tokenizer.ggml.pre` 判预分词器；三个数组按 id 下标对齐读成 `TokenData`；建 `byte_to_id_` / `id_to_byte_` 双向映射；读 bos/eos id 与 add_bos/add_eos 覆盖 | 词表类型决定后面所有分支（补空格 / 转义 / 字节映射 / 是否预分词），必须最先判；只有 BPE 路径需要 `pre`（SPM 路径不读）；token id 就是数组下标，三数组必须同序同长，否则 score/type 会错位；字节兜底通道必须在 encode 之前就绪，否则残段无处可去；词表级开关叠加在「模型族默认值」之上，位置错了就永远读不到覆盖值 |
| src/unicode.cpp | 预分词扫描器入口 | BPE 路径的词边界唯一来源；分支顺序即正则交替的优先级 |

### §14 概要→详细映射表

| 概要条款 | 详细落点 | 参考实现对应 |
|---|---|---|
| §1.1 词表实体与三个数组、§1.2 三组枚举 | §7.1 | `llama_vocab::impl::load`（llama-vocab.cpp:1921）按 id 同序读 tokens/scores/token_type 三个数组；枚举取 `enum llama_vocab_type`（include/llama.h:72）与 `enum llama_token_type`（include/llama.h:91）的子集 |
| §2 步骤 1 判类型 / §3 异常 1 / 3 | §7.1 | `llama_vocab::impl::load` 的「判 tokenizer_model → 判 tokenizer_pre → 按 pre 类型置 flag」顺序；缺 tokens 时抛 `cannot find tokenizer vocab in model file`（llama-vocab.cpp:2405） |
| §2 步骤 2 建索引与字节映射 | §7.1 | `token_to_id[word] = i` 建表（llama-vocab.cpp:2439）+ `llama_vocab::byte_to_token`（:3852）的查表通道 |
| §2 步骤 3 文本改写 / §2 步骤 6 拼 BOS / EOS | §7.2 | SPM 分支的 `add_space_prefix && is_prev_special` 补空格（:3348）+ `llama_escape_whitespace`（:3285）转义；`llm_tokenizer_bpe_session::append_bos` / `append_eos`（:570 / :579）与 SPM 分支的 `output.push_back(special_bos_id)` |
| §2 步骤 4 预分词切词 | §7.2 | `unicode_regex_split`（unicode.cpp:1216）先问 `unicode_regex_split_custom`（:1050），命中 llama3/qwen2 走手写扫描器（:333 / :474） |
| §2 步骤 5 按 score 合并 | §7.3 | `llm_tokenizer_spm_session::tokenize`（llama-vocab.cpp:117）的 symbol 链表 + 最高分堆 + 陈旧条目跳过 |
| §2 步骤 6 字节兜底 | §7.4 | `llama_vocab::byte_to_token` 按词表类型分派：SPM 查 `<0xXX>` 再退到单字节，BPE 查 bytes_to_unicode 映射字符 |
| §3 异常 2~5 / §3 异常 6 兜底落空 | §7.1 / §7.4 | 上游 `init_tokenizer` 对未知类型 `GGML_ABORT("unsupported vocab type")`（:3140）、未知 pre 抛 `unknown pre-tokenizer type`（:2370）、数组长度不对抛 `Index out of array bounds for scores` / `for toktypes`（llama-vocab.cpp:2415 / :2425）；缺可选 key 时 `ml.get_key(kid, dst, false)`（llama-model-loader.cpp:423）的第三参数表示「缺失不报错、用默认值」；字节兜底查不到时上游 `byte_to_token` 的 `.at()` 抛 std::out_of_range，本实现静默丢弃（偏离，见 §16.2） |
| §5 观测点五处 | §7.1~§7.4 | 上游 `LLAMA_LOG_INFO` 的 vocab 摘要与 `llama_vocab::print_info()` |

### §15 裁量区与待拍板

| AI 可以自决 | 必须对齐参考实现 | 必须问人 |
|---|---|---|
| 教学便利代码的实现方式：`tools/gen_unicode_tables.py` 的表生成形态、`tests/gguf_writer.h` 的合成词表构造器怎么写；观测点键名（`vocab` / `tokenize` / `byte_fallback` / `byte_fallback_miss` / `detokenize`）的拼写与分组 | 术语（vocab / token / score / merge / BYTE token / SPM / BPE / pre-tokenizer / bytes_to_unicode / `▁`）必须用上游叫法 | 砍掉承重件（§16.3 里判「承重不可砍」的行） |
| | 结构档：Tokenizer / TokenData / 三组枚举的分层与字段职责 | 新增上游没有的概念（本仓目前没有） |
| | 结构档：本单元覆盖到的调用链骨架（load → encode → merge_tokenize → decode） | 改对齐粒度定义，例如把 `tokenizer.ggml.add_bos` 改成上游的 `tokenizer.ggml.add_bos_token` —— 这动的是对外契约名 |
| | 实现档的替换（判据：能力不减 + 以后接得上），见 §16.2 第二张表 | 补 §16.2 里标为「当前未修」的两处缺陷（数组长度校验、add_bos key 名） |

## 还原模式扩展节

### §16 对齐面

> 参考实现：llama.cpp @ 88b47a755
> 本文只对「本单元涉及的部分」负责。

#### 16.1 概念对齐

| 参考实现的术语 | 本实现的对应物 | 对齐到什么粒度 | 备注 |
|---|---|---|---|
| `llama_vocab` / `llama_vocab::impl` / `llama_token` / `token_data`（text / score / attr） | `Tokenizer`（src/tokenizer.h）/ `int`（token id）/ `TokenData`（text / score / type） | 类型分层 + 术语 + 字段职责 | 真实版把「词表」与「编码会话」拆两层，本实现合成一层；`llama_token` 是 `typedef int32_t`（include/llama.h:69）；attr 是位标志集合，本实现用 7 值枚举（偏离见 §16.2） |
| `enum llama_token_type` / `enum llama_vocab_type`（SPM/BPE/WPM/UGM/RWKV/PLAMO2）/ `LLAMA_VOCAB_PRE_TYPE_LLAMA3` / `_QWEN2` | `TokenType` / `VocabType`（SPM/BPE）/ `PreTokenizer::LLAMA3` / `QWEN2` | 术语 + 取值 | include/llama.h:91 ↔ src/tokenizer.h 逐值同；include/llama.h:72 取子集；llama-vocab.cpp:283 / :373 |
| score（合并优先级） / merge（相邻 symbol 按 score 合并） | `TokenData::score` / `Tokenizer::merge_tokenize` | 字段职责 + 调用链骨架 | 上游 `llm_bigram_spm::score` 取自 `token_data.score`（llama-vocab.cpp:222）；上游 SPM / BPE 各一份 session，本实现合一份 |
| BYTE token（`LLAMA_TOKEN_TYPE_BYTE`） / `▁` 转义（`llama_escape_whitespace`） | `TokenType::BYTE` + `byte_to_id_[256]` / `ESCAPED_SPACE` + `replace_all`（src/tokenizer.cpp / :289） | 术语 + 机制 | BYTE token 上游按词表类型分派查表，本实现按文本形状识别；`▁` 转义上游在 llama-vocab.cpp:3285 |
| SPM（`model = "llama"`） / BPE（`model = "gpt2"`）/ pre-tokenizer / bytes_to_unicode（`unicode_byte_to_utf8`） | `VocabType::SPM` / `::BPE` / `pre_tokenize`（src/unicode.cpp）/ `byte_to_unicode_cp` + `unicode_cp_to_byte`（src/unicode.cpp / :118） | 术语 + 机制 | 上游 `llm_tokenizer_spm_session`（llama-vocab.cpp:114）/ `llm_tokenizer_bpe_session`（:561）；`unicode_regex_split` + `unicode_regex_split_custom_llama3/_qwen2`（unicode.cpp:1216 / :333 / :474）；bytes_to_unicode 上游 unicode.cpp:1162，映射表在 :148 |
| `tokenizer_st_partition`（special token 片段提取）/ `ignore_merges` / `get_add_bos` 等 flag | 无 / `add_bos_` / `add_eos_` / `escape_whitespaces_` / `add_space_prefix_` / `byte_encode_` | 不对齐 / 术语 + 子集 | 前者明确砍掉，见 §16.3 第 2 行；上游 flag 更多，见 §16.3 第 5 行，`ignore_merges` 的已知差异见 §16.3 该行「行为测试」列 |

#### 16.2 设计对齐

「结构档的偏离」：

| 参考实现的机制 | 本实现的位置 | 偏离点 | 为什么可以偏离 |
|---|---|---|---|
| `llama_vocab::impl` 与 `llm_tokenizer_*_session` 两层（llama-vocab.cpp:1779 / :114 / :561） | `Tokenizer` 单层，会话概念被 `merge_tokenize` 取代 | 概念偏离：合掉会话层 | 上游自己在 llama-vocab.cpp:246 标了 TODO 说两份 session 该合并复用；合一层后对照更清楚，能力不减 |
| `llama_token_attr` 位标志集合（UNKNOWN / CONTROL / UNUSED / USER_DEFINED…） | `TokenType` 7 值枚举 | 概念偏离：attr 的多标志组合塌成单值 | GGUF 的 `token_type` 数组本来就是单值枚举（include/llama.h:91），塌成单值不丢信息；上游多出的位是运行时按模型名补的（llama-vocab.cpp:2968 起的 per-token attribute 推断），不在本单元范围 |
| `unicode_regex_split` 的通用路径：`std::regex` / `std::wregex`（`unicode_regex_split_stl`，unicode.cpp:739） | `pre_tokenize` 手写 6 分支扫描器 | 机制替换 | 上游对 llama3 / qwen2 这两条模式**自己也不走**通用路径：`unicode_regex_split_custom`（unicode.cpp:1050）先特判到 `unicode_regex_split_custom_llama3`（:333）/ `_qwen2`（:474）。本实现对齐的是上游的手写扫描器路线 |
| `llama_vocab::byte_to_token`（llama-vocab.cpp:3852）按**词表类型**分派：SPM/UGM 查 `<0xXX>` 再退到单字节；WPM/BPE 查 `unicode_byte_to_utf8(ch)`；PLAMO2 查 `<0x%02X>` | `byte_token_text_to_raw`（src/tokenizer.cpp）按**文本形状**识别 + `byte_token`（:155）三级查找链 | 机制替换 | 三种文本格式在真实词表里都出现过（见 §17.2）；按形状识别能同时吃下 llama3 那种「没有显式 BYTE token、只能拿映射字符查表」的词表。代价是比上游多一次文本形状嗅探 |
| 上游 `llama_vocab::impl::load` 校验 scores / token_type 数组长度并抛错（llama-vocab.cpp:2415 / :2425）；解码 `token_to_piece` 的跳过掩码只有 UNKNOWN 与 CONTROL（:3519） | `Tokenizer::load` 直接按 id 下标读（src/tokenizer.cpp / :102）；decode 跳过 CONTROL / UNKNOWN / UNUSED / UNDEFINED，渲染 NORMAL / USER_DEFINED（:326 / :351） | 细节简化（前者**且是缺陷**） | 长度不一致时本实现越界读（UB），**当前未修**，接不可信词表前必须补；解码多跳过 UNUSED / UNDEFINED 两种上游不跳的类型，但真实 GGUF 里 `<s>` 常被标成 USER_DEFINED（4），上游也照渲染 |
| 上游读 `tokenizer.ggml.add_bos_token` / `add_eos_token`（llama-arch.cpp:341 / :342）；上游 SPM 分支用 `is_prev_special` 决定补不补空格（llama-vocab.cpp:3335 / :3348） | 读 `tokenizer.ggml.add_bos` / `add_eos`（src/tokenizer.cpp / :142）；补空格判据是 `add_space_prefix_ && !work.empty()`（src/tokenizer.cpp） | 概念偏离：key 名与上游契约不一致；细节简化：补空格不看 `is_prev_special` | GGUF 里普遍写的是上游那个名字，所以本实现的词表级覆盖实际读不到，目前靠「SPM / llama-bpe 默认加 BOS」的族默认值兜住，**需拍板**（见 §15）；本实现不切 special token 片段，`is_prev_special` 恒为 true，与「全文本当一段」等价，上游那个状态是给 special token 片段之间不补空格用的（见 §17.4） |

「实现档的替换」：

| 参考实现用什么 | 最小版用什么 | 能力是否等价 |
|---|---|---|
| `llm_symbol` 双向链表（llama-vocab.cpp:80）与 `llm_bigram_spm::queue`（std::priority_queue + 自定义 comparator，llama-vocab.cpp:103）/ `llm_bigram_bpe` 的陈旧判定（llama-vocab.cpp:657） | `std::vector<size_t> start/len` + `std::vector<int> prev/next` 下标数组（src/tokenizer.cpp）；本地 `struct Bigram` + `std::priority_queue`（src/tokenizer.cpp），陈旧判定用入队时的字节长度 `size` 与当前 `len` 比对（:236） | **等价**：合并只改下标与长度，物理字符串不移动；同分靠入队序号保序，两者判的都是「这条 bigram 是否已被别的合并吞掉」。**接得上**：换回链表只需改两处寻址方式，比较器字段与上游一一对应，换成文本比较只需替换一个 if |
| `unicode_cpt_flags` 16 位标志位 + 全量区间表（unicode.h:8 / unicode-data.cpp 164.4K、7034 行、5 张表） | `UNICODE_LETTER_RANGES` / `UNICODE_NUMBER_RANGES` 两张区间表（src/unicode-tables.h，205 行，622 + 133 区间） | **能力不减**（只服务字母类与数字类两个分支）。**接得上**：加类别只要在生成器里多给一组前缀，表结构不变 |
| `llama_vocab::impl::load` 里按文本名识别 special token（FIM / EOT 等，llama-vocab.cpp:2600~2948，收进 `cache_special_tokens` 在 :2938） | 只读 GGUF 里的 `bos_token_id` / `eos_token_id`（src/tokenizer.cpp） | **不等价**，本单元明确砍掉（见 §16.3 第 2 行） |

#### 16.3 砍掉清单的承重判断

| 砍掉什么 | 真实版怎么做 | 概念测试 | 行为测试 | 术语测试 | 结论 |
|---|---|---|---|---|---|
| UGM / WPM / RWKV / PLaMo2 四套 tokenizer | `enum llama_vocab_type` 有 6 类（include/llama.h:72），各有一份 session：`_wpm_session`（llama-vocab.cpp:768）、`_ugm_session`（:951）、`_rwkv_session`（:1312）、`_plamo2_session`（:1618），由 `init_tokenizer`（:3117）按类型实例化 | **通过**：这四类的核心概念（Viterbi / WordPiece 词缀表 / greedy / Aho-Corasick DP）都不在 Llama 系主链路上，本单元的「vocab → token 序列」主干由 SPM + BPE 完整覆盖 | **通过（覆盖面内）**：SPM 与 BPE 的编码结果与上游逐 token 一致（§5 三个真实词表的往返）；四套新词表类型属于**新增输入面**，不是已有能力退化 | **通过**：上游术语 WPM / UGM / RWKV / PLaMo2 在本单元只以「未实现」形态出现，没有被自创名词替换 | 可砍 |
| special token 片段提取（`tokenizer_st_partition`） | 上游先把原文切成 raw_text / special token 两种 fragment 再逐段编码（llama-vocab.cpp:3150，调用点 :3324），并支持 LSTRIP / RSTRIP 属性 | **通过**：上游把「文本里嵌了特殊串」当独立概念；Llama 主干 prompt 不含内嵌特殊串，不影响「文本 → token」主链 | **通过（覆盖面内）**：上游 SPM 分支的 `is_prev_special`（:3335）只在片段之间起作用；本实现恒 true，等价于「全文本当一段」 | **通过**：special token / parse_special 这两个上游术语在本仓没有对应的自创概念 | 可砍 |
| 通用正则引擎（`unicode_regex_split_stl` 的 std::regex / wregex 兜底，unicode.cpp:739） | `unicode_regex_split`（:1216）先问 `unicode_regex_split_custom`（:1050）：命中 gpt2 / llama3 / qwen2 / qwen35 / kimi-k2 / afmoe / newlines 七条模式就走手写扫描器，其余才 fallback 到 std::regex | **通过**：「预分词 = 固定 6 分支扫描器」正是上游当前对 llama3 / qwen2 的做法，本实现不是自创路线 | **通过（覆盖面内）**：本实现 6 分支与上游 `unicode_regex_split_custom_llama3`（:333）/ `_qwen2`（:474）的分支顺序、判定条件逐条比对一致（分支 ③ 数字分组：llama3 最多 3 位、qwen2 单个，两边同）。上游另有 gpt2 / qwen35 / kimi-k2 / afmoe 四套扫描器未覆盖，属新增输入面 | **通过**：`LLAMA_VOCAB_PRE_TYPE_LLAMA3` / `_QWEN2` 在本仓是 `PreTokenizer::LLAMA3` / `QWEN2`，术语本身沿用 | 可砍 |
| 全量 Unicode 归一化与全类别表 | 上游 `unicode_cpt_flags`（unicode.h:8）覆盖数字 / 字母 / 分隔 / 组合记号 / 标点 / 符号 / 控制七类，外加空白、大小写、NFD 标志；区间表在 unicode-data.cpp（164.4K、7034 行、5 张表：flags / whitespace / lowercase / uppercase / nfd）；NFD 归一化 `unicode_cpts_normalize_nfd`（unicode.cpp:1117）只被 WPM 用（llama-vocab.cpp:821） | **通过**：归一化在上游是「某个词表类型要」，不是「分词器要」；SPM/BPE 的 6 分支模式里没有任何归一化步骤 | **通过（覆盖面内）**：上游 llama3 / qwen2 扫描器（:333 / :474）只查 is_letter / is_number / is_whitespace 三个标志，不查 NFD / 标点 / 组合记号；本实现的字母类 + 数字类两表够用 | **通过**：上游的类别名 NUMBER / LETTER 等在本仓对应 `is_unicode_letter` / `is_unicode_number`，没有自造类别名 | 可砍 |
| `ignore_merges` / double-bos 告警 / unused token 抑制等 flag | `get_ignore_merges()`（llama-vocab.cpp:3998）在 BPE 编码里「整词已在词表就不再拆」（:618）；上游对 llama3 / llama-bpe 置 `ignore_merges = true`（:2143），对 qwen2 不置（:2213）；double-bos 告警在 `check_double_bos_eos`（:588） | **通过**：`ignore_merges` 是「词已存在就不再拆」的语义开关，上游只对少数 pre 类型开；本单元不引入同名概念，「按 score 贪心合并」这条主干不变 | **不通过（局部，已知差异）**：上游 llama3 / llama-bpe 走 ignore_merges，本实现不实现 —— 这类「整词已在表里」的片段在两边切得可能不同，token id 可能不一致（文本往返仍正确）。差异只落在 llama3 / llama-bpe 词表；qwen2 上游本就不开该 flag，两边同源 | **通过**：flag 名沿用上游叫法，缺失即缺失，不自造 | 可砍 |
| tiktoken 风格的离线批量 token 化接口、token attr 查询 API | 上游对外是 `llama_vocab::tokenize`（llama-vocab.h:170）与 `llama_tokenize`（include/llama.h:1150）两套入口，外加 `token_get_attr` / `is_eog` 等 attr 查询（llama-vocab.h:102-119） | **通过**：批量接口与 attr 查询是「怎么用词表」，不是「词表怎么算」，不进本单元的还原边界 | **通过（覆盖面内）**：上游这两个接口最终仍落到同一个 `llama_vocab::impl::tokenize`（:3313），与本实现同源同结果 | **通过**：上游 attr / eog 术语在本仓只以「未实现」形态出现 | 可砍 |

#### 16.4 教学便利代码清单

| 多出来的 | 在哪 | 为什么需要 | 谁 import 它 |
|---|---|---|---|
| Unicode 类别区间表生成器 | tools/gen_unicode_tables.py（`ranges_for` / `emit` / `main`） | 不提交生成脚本就复现不出 src/unicode-tables.h。上游的 unicode-data.cpp 同样是脚本生成的产物，但生成脚本不在上游仓库里、形态也不同：上游产物是 7034 行、5 张表（区间→标志位 / 空白集 / 小写映射 / 大写映射 / NFD 区间），本仓产物是 205 行、2 张「类别 → 区间」表 | 无代码 import；人工执行 `python3 tools/gen_unicode_tables.py > src/unicode-tables.h` |
| 合成 GGUF 词表构造器 | tests/gguf_writer.h（`minitest::Writer` + `write_temp`） | 单元 2 的 5 组用例要造三份形状不同的词表（llama / gpt2+qwen2 / gpt2+llama-bpe），手写二进制太啰嗦；这个构造器与 src/gguf.cpp 的解析器互为逆运算 | tests/test_tokenizer.cpp（`write_vocab_gguf`、`add_byte_tokens`、`add_byte_tokens_mapped`）、tests/test_gguf.cpp |
| 真实词表冒烟的挂载开关 + 编排侧的 CLI 入口 | CMakeLists.txt:37 的 `TEST_TOKENIZER_REAL_FILE` 环境属性；src/main.cpp / :66 / :78 | 让 ctest 不带模型文件也能跑、人工可挂真实 GGUF 词表复跑；并把词表挂到「加载 → 编码 → 逐 token 解码回显」的最小推理链上，让单元 2 有真实调用方 | ctest / 人工；src/main.cpp 自用，单元 3/4 复用同一份 `Tokenizer` |

### §17 踩坑记录

另有一处「上游契约名抄错」的偏离（`tokenizer.ggml.add_bos` vs 上游的 `add_bos_token`）已登记在 §16.2 偏离表与 §15 待拍板项，不另立坑条目。

#### 17.1 词表格式不能靠猜：qwen2 / llama-bpe 是 bytes_to_unicode 格式 🔴 【坑型：概念层级错位】

**症状**：真实 qwen2 / llama-bpe 词表一跑就丢空格，编码结果与预期系统性错位。
**根因**：第一版按「词表里存的是原始空格」实现。但 GPT-2 的 bytes_to_unicode 把 0~255 的字节映射到 256 个可见 Unicode 码点：可打印字节映射到自身，其余 68 个（控制字符 / 空白）依次映射到 256..323。空格 0x20 不在可打印集，于是变成 U+0120（`Ġ`）。真实词表里 token 1879 的文本是 `"Ġworld"`，`Ġ` 就代表 0x20。
**修法**：加 `byte_encode_` 开关。编码时对每个原始字节做 `byte_to_unicode_cp` 映射再进词表；解码时用 `unicode_cp_to_byte` 逆映射回原始字节（src/tokenizer.cpp / :329-339）。
**真实版**：llama.cpp 的 `unicode_byte_to_utf8`（unicode.cpp:1162，映射表 `unicode_byte_to_utf8_map` :148）对 0x21~0x7E、0xA1~0xAC、0xAE~0xFF 映射到自身，其余 68 个字节映射到 256..323；`llm_tokenizer_bpe` 用 `byte_encode` 标志控制（llama-vocab.cpp:558，默认 true，SPM 式 BPE 才置 false）。本质完全相同。
**教学点**：词表格式是**模型作者的选择**，不是「BPE 就该这样」。少写这层映射，症状是「空格丢了」，但根因在词表编码空间与字节空间不同 —— 不知道这一层就永远定位不到。教训是词表格式不能只猜，必须拿真实词表文件验证 —— 这也正是单元 1 冒烟测试的价值。

#### 17.2 BYTE token 有三种文本格式，混在一起必须逐格式识别 🔴 【坑型：混合】

**症状**：字节兜底在某些词表上整体失效，表现为「词表覆盖不到的字符被无声丢弃」；llama3 词表尤其怪：它根本没有显式 BYTE token。
**根因**：同一句「代表一个字节」在真实词表里有三种写法：`<0x41>` 十六进制（llama-spm / gemma / PLaMo2）、单原始字节（旧 BPE 词表）、bytes_to_unicode 映射字符（qwen2 / gpt2 / llama3）。llama3 没有 BYTE token，字节兜底必须退化成「拿映射字符本身去查词表」。
**修法**：`byte_token_text_to_raw`（src/tokenizer.cpp）按文本形状三分支识别：长度 7 且形如 `<0xXX>` 走十六进制；长度为 1 走原始字节；否则解码成码点再查 bytes_to_unicode 反表。`byte_token`（:155）再叠三级查找：显式 BYTE 映射 → 映射字符查词表 → 单字节原始文本查词表。
**真实版**：`llama_vocab::byte_to_token`（llama-vocab.cpp:3852）按**词表类型**分派而不是按文本形状：SPM / UGM 查 `<0xXX>` 再退到单字节；WPM / BPE 查 `unicode_byte_to_utf8(ch)`；PLaMo2 查 `<0x%02X>`。本质相同（都是「按约定拼出字节的 token 文本再查表」），但分派依据不同 —— 偏离点已登记在 §16.2。
**教学点**：三种格式各自对应一个模型家族。少认一种，那个家族的词表就整片兜底失败；而症状是「字符不见了」，指向不到「格式没认全」。

#### 17.3 GGUF 的 BOOL 取值是 bool，不是 uint8_t 【坑型：平台或类型差异】

**症状**：`std::get<uint8_t>` 直接抛 std::bad_variant_access。
**根因**：GGUF 的 BOOL 类型编号是 7，单元 1 的 `GgufValue` 把这个类型存进 variant 时用的是 `bool` 成员（src/gguf.h 的 variant 里有 `bool`），取的时候类型必须严格对上；用 `uint8_t` 取就等于问 variant 要一个它没存过的类型。
**修法**：读 `tokenizer.ggml.add_bos` 时用 `std::get<bool>(v->v)`（src/tokenizer.cpp / :143）。
**真实版**：上游不用 variant，走 type-punned 原始数组 + 类型编号分派：`gguf_read_emplace_helper<bool>`（ggml/src/gguf.cpp:592）与 `gguf_get_val_bool`（ggml/src/gguf.cpp:1140）都把类型编号 7 当作 bool 处理。本质相同 —— 类型编号与取值类型必须一一对应。
**教学点**：variant 的取值类型是**编译期**契约，写入分支和读取分支各写一次就可能对不上。少这一次对账，编译期不报错、运行期才炸。

#### 17.4 SPM 句首空格与 add_special 无关 🔴 【坑型：不变式被打破】

**症状**：`encode("ab", add_special=false)` 仍然补了句首空格，与「不要特殊 token 就不该有任何附加」的直觉相反。
**根因**：llama.cpp 里 SPM 分支用 `bool is_prev_special = true;`（llama-vocab.cpp:3335）作为「前一段是不是特殊 token」的判据，初值 true 表示前面没有特殊 token；补空格的判据是 `add_space_prefix && is_prev_special`（:3348）。`is_prev_special` 初值与 `add_special` **毫无关系** —— `add_special` 只决定要不要 push BOS。第一版误把 `add_special` 当成补空格的前置条件。
**修法**：本实现改成 `add_space_prefix_ && !work.empty()`（src/tokenizer.cpp），不看 `add_special`；测试断言 `encode("ab", false) == {8}`（tests/test_tokenizer.cpp:186）。
**真实版**：同上游，SPM 分支的补空格只看 `add_space_prefix` 与 `is_prev_special`。`is_prev_special` 会变 false，是因为上游会切 special token 片段（见 §16.3 第 2 行）；本实现不切片段，该状态恒 true，等价。
**教学点**：`add_special`（要不要 BOS/EOS）与 `add_space_prefix`（词边界怎么标）是两个正交开关，名字都带 "add"，语义完全不同。把一个当另一个的子条件，症状是「多了一个空格」，但根因是开关的正交性被误当成包含关系。
