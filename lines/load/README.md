# load 族 —— 载入域

> 把磁盘上的一个文件变成内存里两个可用的对象：`Model`（可前向）和 `Tokenizer`（可编码）。

## 一 链路图

```text
      CLI: --model X.gguf
             │
             ▼
   ┌──────────────────────┐
   │  GgufFile::load      │  src/gguf.cpp
   │  open + mmap         │  ★ 跨层点：文件 → 内存映射
   │  魔数/版本/头        │
   │  元数据 KV（保序）    │
   │  张量表              │
   │  数据区（双重对齐）    │
   └──────────┬───────────┘
              │  GgufFile（只读视图，不拷贝权重）
      ┌───────┴────────┐
      ▼                ▼
 ┌──────────────┐  ┌──────────────────┐
 │ Model::load  │  │ Tokenizer::load  │
 │ 超参映射      │  │ 词表类型分派      │
 │ 权重装载      │  │ 三张平行数组      │
 │ src/model.cpp│  │ 正反向索引        │
 └──────┬───────┘  │ BYTE 双向映射     │
        │          │ src/tokenizer.cpp │
        │          └────────┬─────────┘
        ▼                   ▼
   Model + HParams     Tokenizer
        │                   │
        └─────────┬─────────┘
                  ▼
        前向（infer 族） / 编码（text 族）
```

★ 跨层点只有两处：**文件 → 内存映射**（`GgufFile::load`）、**原始 KV → 强类型对象**
（`Model::load` / `Tokenizer::load`）。

## 二 逐跳对照

### 线 A：`model_load`（8 跳）

| 跳 | 本仓库落点 | 上游对照（llama.cpp @ 88b47a755） | 真机命令 |
|---|---|---|---|
| 1 | `GgufFile::load` → open + mmap（`src/gguf.cpp`） | `src/llama-mmap.cpp:620:llama_mmap::llama_mmap` | `MINILLAMA_MODEL=… ./build/line_model_load` |
| 2 | `GgufFile::parse` 的头部段（魔数 → 版本 → 张量数 → KV 数） | `ggml/src/gguf.cpp:983:gguf_init_from_file` + `:451:gguf_init_from_reader` | 同上（坏魔数用例见 `tests/test_gguf.cpp`） |
| 3 | `Cursor::read_value` 的类型分发 | `ggml/src/gguf.cpp:426:gguf_read_emplace_helper` | 同上 |
| 4 | `GgufFile::parse` 的张量表段（名字 / 维度 / 类型 / 相对偏移） | `ggml/src/gguf.cpp:1163:gguf_find_tensor` + `:1184:gguf_get_tensor_name` | 同上 |
| 5 | `GgufFile::tensor_data` 的对齐修正 | `ggml/src/gguf.cpp:1012:gguf_get_alignment` + `:1016:gguf_get_data_offset` | 同上 |
| 6 | `Model::load` 的超参段（`src/model.cpp`） | `src/llama-arch.cpp:149:LLM_KV_NAMES` + `src/llama-hparams.h:llama_hparams` | 同上 |
| 7 | `Model::load` 的权重段（embedding + 每层 9 张量 + 输出头） | `src/llama-model-loader.cpp:llama_model_loader::load_all_data` | 同上 |
| 8 | `Model::forward_token`（就绪判定） | `src/llama-context.cpp:1693:llama_context::decode` | `./build/mini-llama --model … --prompt "Hi" --n 4` |

### 线 B：`vocab_load`（7 跳）

| 跳 | 本仓库落点 | 上游对照（llama.cpp @ 88b47a755） | 真机命令 |
|---|---|---|---|
| 1 | `GgufFile::load` | `ggml/src/gguf.cpp:983:gguf_init_from_file` | `MINILLAMA_MODEL=… ./build/line_vocab_load` |
| 2 | `Tokenizer::load` 的 ① 段（类型分派 SPM / BPE） | `src/llama-vocab.cpp:1921:llama_vocab::impl::load` | 同上 |
| 3 | ③ 段（tokens / scores / token_type 三张平行数组） | `src/llama-vocab.cpp:1921:llama_vocab::impl::load` 的 add_token 循环 | 同上 |
| 4 | ④ 段（正反向索引） | `src/llama-vocab.cpp:1365:token_to_id`（正向）+ `id_to_token`（反向） | 同上 |
| 5 | ⑤ 段（BYTE 双向映射） | `src/llama-vocab.h:110:token_to_byte` / `:111:byte_to_token` | 同上 |
| 6 | ⑥ 段（BOS / EOS / add_bos / add_eos） | `src/llama-vocab.cpp:2144`（按 `tokenizer.ggml.pre` 定 add_bos） | 同上 |
| 7 | `Tokenizer::encode`（就绪判定） | `src/llama-vocab.cpp:3358:llm_tokenizer_spm_session::tokenize` | `./build/mini-llama --model … --prompt "Hi" --n 4` |

## 三 学习问题的答案

> **GGUF 里存的是「字符串 key + 字符串张量名」，为什么 llama.cpp 要专门做一层架构前缀映射
> （`LLM_KV_NAMES`），而不是把 key 写死成 `llama.*`？**

因为**同一个超参在不同架构下的 key 名不一样，但含义一样**。

| 语义 | LLaMA 系 | Qwen3 系 | Mistral 系 |
|---|---|---|---|
| 层数 | `llama.block_count` | `qwen3.block_count` | `mistral.block_count` |
| 词表大小 | `llama.vocab_size` | `qwen3.vocab_size` | `mistral.vocab_size` |
| RoPE 底数 | `llama.rope.freq_base` | `qwen3.rope.freq_base` | `mistral.rope.freq_base` |

而**张量名不参与这套前缀**——所有 LLaMA 系模型的层权重都叫 `blk.N.attn_q.weight`，
所以「按名字精确取张量」这件事在 LLaMA 系里是通用的（Qwen2/3 架构派生自 LLaMA，
张量布局完全一致，本仓库一个实现通吃）。

所以 llama.cpp 的做法是：**key 的后半段（`block_count` / `embedding_length` …）是常量表，
前半段（架构名）从文件里读 `general.architecture` 得到**，两者拼起来才是一次查找。
`LLM_KV_NAMES` 就是那张「后半段常量表」。

本仓库的 `Model::load` 复刻了这个结构：

```cpp
std::string prefix = "llama.";
if (const GgufValue * v = gguf.find("general.architecture")) {
    prefix = std::get<std::string>(v->v) + ".";
}
auto key = [&](const char * suffix) { return prefix + suffix; };
```

**如果把 key 写死成 `llama.*`，Qwen3 的模型就会因为「超参不完整」而加载失败**——
这正是本仓库单元 3 踩过的坑（见 `docs/03-forward.md` §17.4）。

## 四 认知纠错记录

| # | 曾经的错误假设 | 实际情况 | 判定依据 |
|---|---|---|---|
| 1 | 「GGUF 的张量表第一个张量一定是 `token_embd.weight`」 | **不成立**。TinyLlama 的第一个张量是 `output.weight` | `line_model_load` 第 4 跳的实际输出；GGUF 规范只规定张量表**有序**，不规定顺序 |
| 2 | 「超参的 key 名都叫 `llama.*`」 | **不成立**。Qwen3 用 `qwen3.*` | `docs/03-forward.md` §17.4 |
| 3 | 「GGUF 的数据区只有一层对齐」 | **不成立**。数据区起点绝对对齐 + 张量之间各自相对对齐，两层 | `docs/05-quantization.md` §17.7 |
| 4 | 「词表住在独立的词表文件里」 | **不成立**。词表住在 GGUF 的**元数据**里（`tokenizer.ggml.tokens` 等） | `line_vocab_load` 第 1~3 跳 |

## 五 真机实测

机器：Linux 5.10 / 8 核 / 16GB；模型：`models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf`
（llama.cpp 官方的 Q8_0 转换版，1.1GB）；日期：2026-11-11。

```bash
$ MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_model_load
观测 : GGUF 版本 = 3
观测 : 张量数 = 201
观测 : 对齐字节 = 32
观测 : general.architecture = llama
观测 : 层数 n_layer = 22
观测 : 维度 n_embd = 2048
观测 : 注意力头 n_head = 32
观测 : KV 头 n_head_kv = 4          ← GQA：8 个 query 头共享 1 个 KV 头
观测 : head_dim = 64
观测 : FFN 宽度 n_ff = 5632
观测 : 词表 n_vocab = 32000
观测 : rms_eps = 1e-05
观测 : rope_theta = 10000
观测 : embedding 类型 = Q8_0
观测 : embedding 形状 = 2048x32000
观测 : 输出头 tied = 0
            类型分布 F32 × 45          ← 45 个归一化向量（F32）
            类型分布 Q8_0 × 156
== 线 model_load 走完：8/8 项通过
```

```bash
$ MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_vocab_load
观测 : tokenizer.ggml.model = llama      ← SPM 路径
观测 : token 条数 = 32000
观测 : 词表大小 = 32000
观测 : bos_id = 1        观测 : eos_id = 2
观测 : add_bos = 1
== 线 vocab_load 走完：8/8 项通过
```

**耗时**：模型载入约 4.4 秒（页缓存热），首次冷读约 78 秒（1.1GB 从磁盘读）。

## 六 差异注明

| # | 差异 | 我的实现 | 真机（llama.cpp @ 88b47a755） | 为什么可接受 |
|---|---|---|---|---|
| 1 | 元数据查找 | 线性扫描 `metadata_` | `gguf_context` 的哈希索引 | 元数据只有 24 条，线性查找纳秒级；结构（保序的 key→value）没动 |
| 2 | 写路径 | 只读 | `gguf_write_*` 一族 | 推理只读；写路径是模型转换工具链的职责 |
| 3 | 字节序 | 只支持小端主机 | `gguf` 读路径的 `swap_endian` | 目标平台 x86/ARM 都是小端 |
| 4 | 主机端张量查找 | 按名字精确匹配 | `llama_model_loader::find_tensor` 的通配/正则 | 本仓库的模型都是 LLaMA 系，张量名规范统一 |
| 5 | lazy 加载 / 用量统计 | 装载期一次性全读 | `llama_model_loader` 的 `load_all_data` 可选 lazy | 教学版要的是「一眼看清内存里有什么」 |
| 6 | 超参映射 | `general.architecture` + 后缀常量拼 key | `LLM_KV_NAMES` + `LLM_KV::LLM_KV` | **结构同构**（前半段来自文件、后半端来自常量表） |
| 7 | 分词器 | 只实现 SPM + BPE | 五套（SPM / BPE / UGM / WPM / RWKV） | 掌握两套已覆盖 Llama 全系；其余是同一套词表结构的变体 |
| 8 | 预分词 | 手写 6 分支扫描器 | 通用正则引擎 `unicode_regex_split` | 模式固定，手写更快更透明、零依赖；**切分语义等价** |
| 9 | 词表查找 | `unordered_map` + 双向索引 | `llama_vocab::impl` 的 `token_to_id` / `id_to_token` | 结构一致 |

## 七 可观测点

| 量 | 本实现的接口 | 真机对应（llama.cpp @ 88b47a755） |
|---|---|---|
| GGUF 版本 | `GgufFile::version()` | 文件头字节（`gguf_init_from_file` 校验） |
| 张量数 | `GgufFile::tensor_count()` | `llama_model_meta_count` / `llama_model_desc` |
| 对齐字节 | `GgufFile::alignment()` | `general.alignment` 元数据 |
| 架构名 | `GgufFile::find("general.architecture")` | `llama_model_meta_val_str(model, "general.architecture", …)` |
| 超参 | `Model::hparams()` | `include/llama.h:577:llama_model_n_layer` / `:574:llama_model_n_embd` / `:579:llama_model_n_head` |
| 词表大小 | `Tokenizer::n_tokens()` | `llama_vocab_n_tokens(vocab)` |
| 权重类型 | `TensorBuf::type` | `llama_model_size(model)` + `llama_model_desc` |
| 模型体积 | `GgufFile::tensor_nbytes` 逐张量求和 | `llama_model_size(model)` |

> 真机侧的 `llama_model_desc` / `llama_model_meta_val_str` 是**同一份元数据的另一种读法**——
> 本仓库的 `GgufFile::find` + `get_string` 就是它的教学版。
