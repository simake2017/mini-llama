# 单元 2：分词器 —— 从字节序列到 token 序列

（对应真实 llama.cpp 的 src/llama-vocab.cpp，约 174K 的 llama_vocab + 5 套 tokenizer 实现）

## ① 我实现了什么

- **词表加载**：从 GGUF 元数据读取 token 文本 / score / 类型，构建正反向索引和 BYTE token ↔ 原始字节双向映射
- **两条编码路径**：
  - **SPM**（`model="llama"`，TinyLlama/Llama-2 系）：整段文本切 UTF-8 字符，按 score 贪心两两合并；句首补空格、`▁` 转义、默认加 BOS
  - **BPE**（`model="gpt2"`，Llama-3/Qwen2 系）：先按预分词正则切词，词内合并；支持 `llama-bpe`（数字 1~3 位分组）和 `qwen2`（数字单切）两种预分词器
- **预分词扫描器**：把 llama3/qwen2 的 6 分支正则手写为确定性扫描器（含 `\p{L}`/`\p{N}` Unicode 类别，区间表二分查找）
- **三种字节兜底格式**：`<0xXX>` 十六进制（llama-spm）、单原始字节、GPT-2 `bytes_to_unicode` 映射字符（qwen2/llama-bpe 都是这种！）
- **解码**：特殊/控制 token 跳过，BYTE token 还原字节，▁/bytes_to_unicode 逆变换还原空格和原始字节

关键设计决策：

- **SPM 和 BPE 共享同一个合并核心**（symbol 链表 + 按 score 最大堆 + 陈旧条目校验）。llama.cpp 里这两份代码几乎一样却没合并（历史原因），我们合一份，对照时反而更清楚。
- **正则不引引擎**：模式固定且只有 6 个分支，手写扫描器零依赖、可逐分支注释，还顺带讲清了 `\s+(?!\S)` 的贪心回溯语义。
- **Unicode 类别表用 Python 生成**（tools/gen_unicode_tables.py），对照 llama.cpp 的 unicode-data.cpp 同样是生成的；我们只生成 L/N 两类，205 行 vs 真实版 164K。

## ② 真实 llama.cpp 对应源码

| mini-llama | 真实版 | 说明 |
|---|---|---|
| `src/tokenizer.cpp` load | `llama-vocab.cpp` llama_vocab::impl 加载段（2400 行起） | token_to_id / byte_to_token 构建逻辑一致 |
| `merge_tokenize` | `llm_tokenizer_spm_session::tokenize` + `llm_tokenizer_bpe_session::tokenize` | 两份几乎相同的代码，我们合一份 |
| `src/unicode.cpp` pre_tokenize | `unicode_regex_split`（unicode.cpp）+ `llm_tokenizer_bpe` 的 regex_exprs | 通用正则引擎 vs 手写扫描器 |
| 类别表 `unicode-tables.h` | `unicode-data.cpp`（164K 生成代码） | 同源不同量：L/N 共 755 区间 vs 全类别 |
| `byte_to_unicode_cp` | `unicode_byte_to_utf8` | GPT-2 字节映射，逐字节一致 |
| 空白转义 ▁ | `llama_escape_whitespace` | 仅 SPM/▁ 类词表使用 |
| BPE 编码管线 | `llama_vocab::impl::tokenize` BPE 分支 | fragment → escape → regex split → merge |

## ③ 真实版比我多做了什么、为什么

| 真实版多做 | 为什么可以砍 |
|---|---|
| UGM（Viterbi 一元模型）/WPM/RWKV/PLAMO2 等 5 套 tokenizer | 每套对应特定模型家族，掌握 SPM+BPE 两套已覆盖 Llama 全系 |
| special token 片段提取（tokenizer_st_partition） | 处理文本内嵌 `<|end_of_text|>` 的场景；教学主干可后补 |
| 通用正则引擎 + 全量 Unicode 归一化（NFKC/大小写折叠） | 固定模式手写更快更透明；归一化这些预分词器用不到 |
| ignore_merges、double-bos 告警、unused token 抑制等十余个 flag | 边界行为微调，不影响编码主流程 |
| tiktoken 风格的离线 token 化批量接口、token attr API | 性能/易用性接口 |

## ④ 踩坑记录

- **qwen2/llama-bpe 词表是 bytes_to_unicode 格式**：第一版按"原始空格"实现，真实词表一跑就丢空格。token 1879 = "Ġworld"（Ġ=U+0120 表示 0x20）。教训：词表格式不能只猜，要用真实文件验证——这也印证了单元 1 冒烟测试的价值。
- **BYTE token 三种文本格式**：`<0xXX>`（llama-spm）、原始单字节、映射字符，混在一起需要逐格式识别；llama3 甚至没有显式 BYTE token，字节兜底要退化为"映射字符本身查词表"。
- **GGUF BOOL 解析为 bool 不是 uint8_t**：`std::get<uint8_t>` 直接 bad_variant_access。教训：variant 取值类型必须和写入分支严格一致。
- **SPM 句首空格与 add_special 无关**：llama.cpp 里 `is_prev_special` 初值 true，encode("ab", add_special=false) 依然补空格。第一版加了 add_special 门控，与真实行为不符。

## ⑤ 验收记录

- 合成用例 5 组全绿：预分词扫描器（含切分不变式 fuzz）、SPM 合并优先级/字节兜底、qwen2 映射词表、llama-bpe BOS+映射合并、缺 key/类型错误
- 真实词表 3 个冒烟全过，往返正确：
  - llama-bpe（128256 tokens）：`"Hello world"` → 3 tokens（含 BOS）→ 完整还原
  - llama-spm（32000 tokens）：→ `" Hello world"`（句首空格是 SPM 规范行为）
  - qwen2（151936 tokens）：→ 2 tokens → 完整还原
