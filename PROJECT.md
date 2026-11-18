# PROJECT：mini-llama

> 用 C++17（仅标准库 + POSIX mmap）从零还原 llama.cpp 的推理主干：GGUF 加载 → BPE/SPM 分词 →
> 前向计算 → KV Cache 增量生成 → 采样，量化模型反量化后可直接跑真实 GGUF。

## 基本信息

- **项目类型**：还原项目（restore）
- **参考实现**：llama.cpp b10143（commit `88b47a755`，2026-07-27，master 分支）
- **参考实现位置**：`/root/ai/llama.cpp` @ `88b47a755`（只读，不切分支、不改工作区）
- **使用的还原技能版本**：mini-restore v1.2
- **一句话定位**：我要用最小功能还原 llama.cpp 的**单模型单序列推理主干**——
  一个 GGUF 文件进来，能把一段 prompt 变成续写出来的 token 流。

## 必做范围（Must Have）

按「承重件优先」列出。每条注明它是核心主干 / 承重但复杂。

| 能力 | 承重判断 | 降级还原方式（如有） |
|---|---|---|
| GGUF 文件解析（头 / 元数据 KV / 张量表 / mmap 数据区） | 核心主干 | — |
| 超参映射（GGUF key → hparams）与权重装载 | 核心主干 | — |
| SPM（llama 系）与 BPE（gpt2 系）两条分词路径 | 核心主干 | — |
| 前向计算（RMSNorm / RoPE / GQA / SwiGLU） | 核心主干 | — |
| KV Cache 增量前向 | 核心主干 | — |
| 采样链（greedy / temperature / top-k / top-p） | 核心主干 | — |
| 量化反量化（Q8_0 / Q4_K / Q5_K / Q6_K） | 核心主干 | — |
| **增量前向 ≡ 全量前向**（逐位置 logits 一致） | 承重但复杂 | 不降级——这是「缓存只是避免重算」这一语义的唯一检验，必须真做 |
| 张量级混合量化（同一模型里 F32 与 Q8_0 混用） | 承重但复杂 | 不降级——真实版主流模型（Q4_K_M）就是张量级混合，砍掉就装不下真模型 |
| GGML 计算图抽象 | 承重但复杂 | **降级还原**：换成「阶段顺序固定的直接循环」，数学与阶段划分不变，丢掉的是可复用与并行调度 |

## 明确不做（Out of Scope）

每条砍掉的都要能答出三个测试（概念 / 行为 / 术语）的结论。
同一种形状的边缘加固**批量判**（一条代表性结论 + 列全归入那一批的条目）；
看着像承重件的逐条三测，一个都不省。

| 砍掉什么 | 真实版怎么做 | 概念 | 行为 | 术语 | 结论 |
|---|---|---|---|---|---|
| ① 写 GGUF（`gguf.cpp` 后半 `gguf_write_*`） | ggml 的 `gguf_write_header/kv/data` 一族 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ② 元数据 key 哈希索引 | `gguf_context` 的 `hashmap` | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ③ 大端主机字节序交换 | `gguf` 读路径的 `swap_endian` | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ④ 重叠张量去重（`find_tensor_by_offset`） | ggml 的共享张量查找 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑤ Windows / NUMA / huge pages 适配 | `llama-mmap` 的平台分支 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑥ 张量名通配/正则查找、lazy 加载、用量统计 | `llama_model_loader` 的 `find_tensor` 族 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑦ UGM / WPM / RWKV / PLAMO2 四套分词器 | `llama-vocab.cpp` 的五套 session | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑧ special token 片段提取（`tokenizer_st_partition`） | `llama-vocab.cpp` 的文本内嵌特殊串切分 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑨ 通用正则引擎 + 全量 Unicode 归一化（NFKC / 大小写折叠） | `unicode_regex_split` + `unicode-data.cpp` | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑩ ignore_merges / double-bos 告警 / unused token 抑制等 flag | `llama-vocab.cpp` 的十余个开关 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑪ tiktoken 风格批量 tokenize、token attr API | `llama-vocab.cpp` 的批量接口 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑫ 几十种架构变体（Gemma q-norm、MoE、attention sink、RWKV 循环…） | `llama-arch.cpp` 的架构分派 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑬ 线程池 / SIMD / 缓存分块 | ggml-cpu 的后端调度 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑭ rope_scaling（YaRN / NTK 外推）、rope clip、各种掩码类型 | `llama-model.cpp` 的 rope 参数 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑮ 权重 mlock / NUMA / 张量并行 | `llama-model-loader` 的设备管理 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑯ 4 种 KV cache 实现（DSA / ISWA / dsv4） | `llama-kv-cache*.cpp` | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑰ slot / cell 管理、seq_id 多并发序列、跨 batch 复用 | `llama_kv_cache` 的 cell 分配器 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑱ 采样链 10+ 节（min-p / 重复惩罚 / grammar / mirostat…） | `llama_sampler_chain` | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑲ `llama_batch` / ubatch 并行解码 | `llama-batch.cpp` | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ⑳ 投机解码、层卸载（ngl） | `llama-spec.cpp` / `llama-impl` 的 offload | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ㉑ 约 20 种量化格式（Q2_K~Q8_K 全量、IQ 系列、FP8） | `ggml-quants.c` 的全部 `dequantize_row_*` | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ㉒ SIMD 向量化反量化 + 专用 dot 内核（AVX2 / NEON） | `ggml-cpu` 的量化内核 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ㉓ 原位量化（quantize 侧） | `ggml-quants.c` 的 `quantize_row_*` / `llama-quant.cpp` | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| ㉔ Q3_K 的 super-block scale 再量化（hbits） | `ggml-quants.c` 的 Q3_K 路径 | 通过 | 通过 | 通过 | 边缘加固，可砍 |
| **GGML 计算图抽象**（先建图、后执行、可复用） | `llama-graph.cpp` + ggml 的 `ggml_graph` | 通过 | 通过 | 通过 | **降级还原**：换成阶段顺序固定的直接循环——前向数学、阶段划分（embed → N×block → 输出头）完全保留，只丢掉「图可复用 / 可并行调度 / 可切后端」这三个工程属性。判据：换完之后「一次前向 = 一次 token 进、一次 logits 出」这个概念还在，且以后要加一层只需要在阶段列表里插一项 |
| **KV cache 的 slot / cell 复用与 seq_id** | `llama_kv_cache::find_slot` | **不通过** | 通过 | 通过 | **承重，不可砍 → 降级还原**：降级为「纯顺序追加（pos == size）」的单序列缓存。概念测试不通过的原因：`find_slot` 的存在意义是**在多序列/乱序到达时决定 K/V 落在哪个 cell**；砍掉它之后「cell」这个概念塌了——本实现里 K/V 的位置就是 `pos`，没有独立于位置的 cell 身份。降级后「缓存 = 按位置存放的历史 K/V」这一核心语义完整保留 |
| **RoPE 配对约定**（相邻配对 vs half-split） | `llama-model.cpp` 的 `LLAMA_ROPE_TYPE_NORM` | **不通过** | **不通过** | 通过 | **承重，不可砍**：数学上两种约定不等价（除非加载期对 wq/wk 做行置换），行为测试在「混合 token 的多位置」序列上直接分叉。术语测试通过——业界就叫 RoPE。**必须用与上游当前版本一致的约定**，这是逐数值对拍的前提 |

> ①②③④⑤⑥ 同属「边缘加固 / 工程适配」形状，批量判：概念测试全部通过（GGUF 仍然是
> 「有版本号的头 + 保序的元数据 KV + 张量表 + 数据区」），行为测试全部通过（读路径
> 不受影响），术语测试全部通过（用户不会因为「不能写 GGUF」而改叫别的东西）。
> ⑦⑧⑨⑩⑪ 同属「分词器的旁支机制」形状，批量判；⑫⑬⑭⑮ 同属「架构与平台变体」形状，批量判；
> ⑯⑰⑱⑲⑳ 同属「并发/扩展性」形状，批量判；㉑㉒㉓㉔ 同属「量化格式覆盖面」形状，批量判。

## 模型形状

按家族选（声明式 / 文档型 / KV 型 / 复制共识 / 流式），不要套默认模板。

- **家族**：计算流水线型（pipeline）——**不在 mini-restore 的 canonical 五族之内，这里显式说明理由**
- **形状**：
  1. **静态张量仓库**：一个 GGUF 文件 = 一张「名字 → 形状 + 类型 + 字节」的有序表（`GgufFile`），
     超参（`HParams`）是另一张键值表。模型加载 = 把这两张表**读出来**，不是「构造对象图」。
  2. **分层阶段链**：一次前向 = `embed → [RMSNorm → QKV → RoPE → GQA 注意力 → 投影 → SwiGLU FFN] × n_layer → 输出头`，
     阶段顺序**固定**，每个阶段的输入输出形状由 hparams 决定。
  3. **有状态缓存**：跨 token 的唯一可变状态是 `KVCache`——按 (layer, pos) 存放历史 K/V。
     状态是「追加式」的，不是「可迁移」的。
- **理由**：参考实现 llama.cpp 的核心不是「一个对象的状态机」，而是
  **「一组固定形状的阶段，按序吃进一个 token、吐出一个 token」**。
  - 不是**声明式**：没有「期望 vs 实际」两份状态，没有控制器对账语义。
  - 不是**文档型**：GGUF 的元数据 KV 是**有序**的（保序存储），但它的 schema 是固定的
    （超参 key 由架构决定），不是「字段由用户定」。
  - 不是 **KV 型**：GGUF 里的 "KV" 是 key-value 元数据，不是「键值契约 + 版本号」那个语义；
    真 KV 型项目的承重概念是版本号与 CAS，本项目没有。
  - 不是**复制/共识**：单机单进程，无 quorum、无状态机复制。
  - 不是**流式**（最接近但仍然不同）：流式家族的承重概念是「背压与流控」——
    消费方慢下来时生产者要能顶住；本项目的生成循环是「上一阶段的输出**同步**喂给下一阶段」，
    没有事件循环、没有 handler 注册、没有背压语义。套流式形状会教出错误的直觉。

## 对齐粒度（结构档必守，实现档可换）

- **结构档（不可替换）**：概念术语照抄（`gguf` / `hparams` / `kv cache` / `top-p` / `rope`…）/
  类型分几层 + 有哪些字段 + 职责划分照抄 /
  **本单元覆盖到的那些阶段**的调用链骨架与阶段划分照抄
  （参考实现有、本单元不覆盖的阶段走 §16.3 承重判断，**不算结构偏离**）
- **实现档（可替换）**：字段用什么类型承载（原生链表 → `vector` / `std::list`）/
  内部算法 / 错误处理 / 并发控制 / 日志格式

**判据**：换完之后，**结构还在吗？以后要扩充还接得上吗？**
「接得上」= 以后要加的功能，用新结构能直接写出来，不用推倒重来。

本项目的典型实现档替换（详见各单元 §16.2）：

| 参考实现用什么 | 最小版用什么 | 能力是否等价 |
|---|---|---|
| ggml 计算图 + 后端调度 | 阶段顺序固定的直接循环 | 等价——前向数学与阶段划分不变，少了图复用与并行调度 |
| `ggml_vec_dot` 函数指针类型表 | `matvec` 里的 switch 分发 | 等价——分派集合相同，只是多一层 switch |
| `llama_sampler_chain` 链式组合 | 固定顺序的单个函数 | 等价——**节序已对齐上游默认链**（top-k → top-p → temperature → dist），各节判据相同，少了任意组合的自由度 |
| `llama_kv_cache` cell/slot 分配 | 纯顺序追加（pos == size） | **有损但概念保住**——「历史 K/V 按位置存放」在，多序列与乱序到达不在（§16.3 已按承重不可砍登记） |

## 必须挂载的约束

- `docs/constraints/04-对齐约束.md` —— 承重判据 + 砍掉清单的承重结论
- `docs/constraints/05-注释规范.md` —— **注释规范走 `comment` 技能**
  （四段式 / 禁止清单 / 三层模板 / 图型选法）+ 还原项目叠加层
  （教学标签 / 3 档用例 / 日志分级 / 【教学】标记）

> 01 / 02 / 03 未建及理由见 `docs/constraints/04-对齐约束.md` §0。
> 编号按 mini-restore 规定保留 04 / 05，不重排。

## 使用的 skill

| skill | 版本 | 用途 |
|---|---|---|
| `biz-dev-pipeline` | v1.0 | 开发流程 + 还原模式分支（三步、签字关口、锚点、挂载点、映射表、裁量区） |
| `tech-doc` | v1.1 | 文档骨架（§1~§15），含 §16 对齐面 |
| `comment` | v1.0 | **注释规范**（四段式 + 模板 + 检查清单） |
| `mini-restore` | v1.2 | 还原模式三补丁（对照面 / 承重判据 / 教学化约束） |
| `line` | 无版本号 | 按业务线把主干拆成端到端子流程（`lines/`），与单元划分正交 |

> **版本对不上 = 这个项目是旧形状**，按 `mini-restore/SKILL.md` 的
> 「老项目迁移清单」迁一遍再继续做。

## 验收口径

- **停 1**：概要签字（对齐核对项：结构档——术语 / 分层 / 字段职责 /
  本单元覆盖到的调用链骨架；
  实现档的替换已按 §16.2「实现替换」栏登记）
- **停 2**：单元末验收（清单见 `mini-restore/references/checklist.md`）

本项目的停 2 已于 5 个单元全部走完，记录在各单元文档的「验收记录」节；
数值正确性的最终口径是 §5 观测点里的两条对拍：
与 libllama 的逐 logits 比对、与独立 numpy 仲裁者（`tests/arbiter.py`）的逐位比对。

## 教学便利代码

参考实现没有、还原项目自己加的代码（demo / 调试 CLI / 打印工具 / 断言辅助），
集中在 `src/main.cpp` 与 `tests/`、`tools/`，核心库（`src/*.cpp` 里的
`gguf` / `tokenizer` / `model` / `kvcache` / `sampler` / `quant`）不 import 它们。

| 文件 | 作用 | 对应上游 | 进核心路径？ |
|---|---|---|---|
| `src/main.cpp` | CLI 入口（解析 `--model/--prompt/--temp/--top-k/--top-p/--n/--seed`） | `examples/cli` / `examples/simple-chat` | 否——是入口外壳，库不依赖它 |
| `tests/gguf_writer.h` | 合成 GGUF 构造器（测试用：写头 / KV / 张量表 / 数据区） | 上游无（上游用 `gguf-py` 写文件） | 否——只在 `tests/` 里 |
| `tests/arbiter.py` | 独立 numpy 仲裁者（从 GGUF 原始字节手写解析 + 前向，零复用本仓库代码） | 上游无 | 否——离线对拍脚本 |
| `tools/gen_unicode_tables.py` | 生成 `src/unicode-tables.h`（L / N 两类 Unicode 区间） | 上游 `scripts/` 的 Unicode 表生成流程 | 否——离线生成脚本 |
| `tools/audit_anchors.py` | 核对 `src/` 注释里写的上游锚点（`文件:行:符号`）是否真实存在于 `/root/ai/llama.cpp` | 上游无 | 否——离线审计脚本（落实 `docs/constraints/05-注释规范.md` §9「② 锚点自己 grep 过」这条硬要求） |
| `tools/fix_refs.py` | 注释改造后行号会漂移 —— 用 `git HEAD` 的旧源文件把旧行号还原成符号，再定位新行号 | 上游无 | 否——离线迁移脚本（注释大改后跑一次即可） |
| `tests/test_*.cpp` | 5 个单元的合成用例（自研极简断言宏） | 上游 `tests/` 是 Python 端到端脚本，形态不同 | 否——只在 `tests/` 里 |
| `tests/test_lines.cpp` | 8 条业务线的自检入口（逐线跑一遍并统计校验条数） | 上游无（上游按 `examples/` 拆散程序，无统一自检） | 否——只在 `tests/` 里 |
| `lines/**` | 业务线 harness：一条线一个 `<线名>_line.cpp`，含可点击入口 / 线本体 / 自注册 / 独立 `main`；`lines/_shared/` 提供 `ctx.step_at/value/log/raw/out/check/summary` | 上游无（上游按 C++ 源文件拆，不按业务线拆） | 否——`lines/` 只调用 `src/` 的公开 API，不反向依赖 |

清单同步登记在各单元文档的 §16.4（教学便利代码清单）。

> **`lines/` 的边界规则**：`lines/` 里不出现被研究项目（`src/`）的逻辑实现——
> 只允许调用 `src/` 的既有 API 并组织流程，避免「教学层重写核心」导致两套真相。
