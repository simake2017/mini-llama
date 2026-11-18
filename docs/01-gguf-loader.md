# 技术文档：单元 1 GGUF 模型加载 —— 把「一个文件」变成「可查询的张量仓库」

- 读者：想弄懂 GGUF 容器格式、以及为什么要分「解析」与「取超参」两层的人
- 审法：先过 §16.1 的术语对齐（确认两边说的是同一件事），再按 §17 逐坑自检，最后用 §9 的数据形状核对每个字段
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
| `GgufType`（src/gguf.h） | GGML 张量类型编号（文件里只存 u32 编号） | 文件作者 | `ggml_block_params` / `Model::load` | 编号不在表内 → `ggml_block_params` 抛「未知类型编号」 |
| `GgufValueType`（src/gguf.h） | 元数据 KV 值的类型编号 | 文件作者 | `read_value` | 编号未知 → 抛「未知元数据类型编号」 |
| `GgufValue`（src/gguf.h） | 一个 KV 值：13 种标量 + string + 递归数组 | `read_value` | `get_*` 系列访问器 | 用 `std::variant` 表达，不存在「空值」；类型不符由访问器抛 |
| `TensorInfo`（src/gguf.h） | 张量表一条：名字 / 类型 / 维度 / 相对偏移 | `GgufFile::parse` | `find_tensor` / `tensor_data` | `dims` 为空只在解析期出现（构造后即校验过 n_dims） |
| `GgufFile`（src/gguf.h） | 已 mmap 的模型文件：只读元数据 + 只读张量字节 | `GgufFile::load` | 单元 2/3/5 全部读模型的地方 | 构造失败即抛异常，不存在「半加载的 GgufFile」 |

#### 1.2 枚举值对象

| 枚举 | 取值 | 出现位置 |
|---|---|---|
| `GgufType` | F32 / F16 / BF16 / F64 / I8 / I16 / I32 / I64 + Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q8_1 + Q2_K ~ Q8_K（共 20 种） | 编号即文件格式，**只能追加不能重排**；`ggml_block_params`（src/gguf.cpp）给出每种的块参数 |
| `GgufValueType` | UINT8 / INT8 / UINT16 / INT16 / UINT32 / INT32 / FLOAT32 / BOOL / STRING / ARRAY / UINT64 / INT64 / FLOAT64（13 种） | `read_value`（src/gguf.cpp）按编号分发 |
| 异常类别（`std::runtime_error`） | 「不是 GGUF 文件（魔数不匹配）」「不支持的 GGUF 版本: N」「文件被截断（需要 N 字节，剩余 M 字节）」「张量 X: n_dims 非法: N」「张量 X 越出数据区」「read_value: 未知元数据类型编号 N」「read_value: 不支持嵌套数组」「read_value: 数组元素数异常: N」「GGUF 缺少元数据 key: K」「GGUF 元数据 K 不是 string / 不是 array / 类型不符，期望 u32」「ggml_block_params: 未知类型编号 N」「open 失败 / fstat 失败或文件为空 / mmap 失败」 | 逐条见 §3 与 §8 |

#### 1.3 名词

| 名词 | 一句大白话 |
|---|---|
| `GGUF` | 模型容器格式：头部元数据 KV + 张量表 + 张量原始字节 |
| `magic` | 文件头 4 个原始字节 `"GGUF"`，**不是** GGUF string（无长度前缀） |
| `GGUF string` | `u64 长度前缀 + 原始字节`（无 NUL 结尾），文件名和 KV 的 key 都用它 |
| `KV` | 元数据键值对，`key` 是字符串，`value` 是强类型值；**保序**存储 |
| `tensor info` | 张量表一条：名字 / 维度 / 类型编号 / 相对数据区起点的偏移 |
| `alignment` | 数据区起点与每个张量实际位置的对齐粒度，默认 32 字节 |
| `ne[]` 最快维在前 | GGML 张量维度顺序：`dims[0]` 是行内连续维 |
| `blck` / `type_size` | 一个量化块含多少元素、占多少字节（`ggml_block_params` 的返回值） |

### §2 正常流程

```text
[open + fstat] -> [mmap + madvise] -> [解析] -> [数据区基址]
                                                   |
解析: [魔数] -> [版本] -> [计数] -> [元数据 KV × n] -> [general.alignment] -> [张量表 × n] -> [数据区起点对齐]
```

1. `open` + `fstat` 拿到文件大小；`mmap` 整个文件（`PROT_READ | MAP_SHARED`），`madvise(MADV_WILLNEED)` 提示内核顺序读 —— 【为什么】模型文件动辄 GB 级，整读进内存会撑爆物理内存，mmap 让页按需换入。
2. 解析分五步，**顺序与上游 `gguf_init_from_reader` 完全一致**：魔数 → 版本 → 计数 → 元数据 KV → 张量表；最后把游标位置向上对齐到 `alignment` 得到数据区起点。
3. 元数据 KV **保序**存入 `std::vector<std::pair<std::string, GgufValue>>` —— 【为什么】顺序是文件里写下的顺序，上游也保序（`ctx->kv` 是 vector）；顺序对某些工具链有意义，且保序 + 线性查找已经够快。
4. 张量表逐条读入，维度原样保留（`dims[0]` 仍是最快维），偏移记为**相对数据区起点**的未对齐值。
5. 解析末尾做一次**廉价越界预检**：逐张量算「对齐后的偏移 + 字节数」是否越过数据区末尾，越过即抛 —— 【为什么】越界读会段错误，而段错误的栈指不到「哪个张量坏了」。

### §3 异常处理

| # | 异常 | 如何处理 |
|---|---|---|
| 1 | 文件打不开 / 为空 / mmap 失败 | 不可重试；`GgufFile::load` 抛异常，错误信息带路径与 `strerror(errno)`；不留下半开的 fd |
| 2 | 魔数不匹配（不是 GGUF） | 不可重试；抛「不是 GGUF 文件（魔数不匹配）」；此时刚建立的 mmap 已被 `catch(...)` 块 munmap 回收 |
| 3 | 版本不支持 | 不可重试；抛「不支持的 GGUF 版本: N」（本实现支持 v1/v2/v3，上游已弃 v1，见 §16.2） |
| 4 | 文件被截断（任何一次读取越界） | 不可重试；`Cursor::require` 抛「文件被截断（需要 N 字节，剩余 M 字节）」，**带上剩余字节数**便于判断差多少 |
| 5 | 元数据类型编号未知 | 不可重试；抛「read_value: 未知元数据类型编号 N」 |
| 6 | 数组嵌套数组 / 元素数异常 | 不可重试；分别抛「不支持嵌套数组」「数组元素数异常: N」（N > 2^31 视为恶意长度，先防溢出再循环） |
| 7 | 张量 n_dims 非法（0 或 > 4） | 不可重试；抛「张量 X: n_dims 非法: N」，**带张量名** |
| 8 | 张量越出数据区 | 不可重试；解析期预检抛「张量 X 越出数据区（文件损坏或截断）」；访问期 `tensor_data` 再抛「张量 X 数据越界」 |
| 9 | 元数据 key 不存在 / 类型不符 | 不可重试；`find` 返回 `nullptr`（供「可缺省」的场景用），`get_*` 抛「缺少元数据 key: K」/「类型不符，期望 u32」 |

### §4 Out of Scope

| 砍掉的能力 | 为什么可以砍 | 承重判断（详见 §16.3） |
|---|---|---|
| **写 GGUF** | 模型转换是工具链的事，推理只读 | 可砍 |
| **元数据 key 的哈希索引** | 模型 KV 只有几十条，线性查找是纳秒级 | 可砍 |
| **大端主机字节序交换** | 目标平台 x86/ARM 都是小端 | 可砍（但保留了大端分支作为演示，见 §17.4） |
| **重叠张量去重**（`find_tensor_by_offset`） | 小模型没有共享权重的张量 | 可砍 |
| **超参 → 强类型结构体映射** | 留给单元 3，作为独立分层点 | 可砍（**刻意推迟**） |
| **张量名通配/正则查找、lazy 加载、用量统计** | 单元 3 按名字精确取张量已够用 | 可砍 |

### §5 观测点

| 时机 | 打的键、携带的字段 |
|---|---|
| 解析成功 | 无日志（解析器是库，安静是本分）；调用方（CLI src/main.cpp）打一行模型摘要 |
| 解析失败 | 异常消息带：魔数/版本/截断字节数/张量名/key 名 —— 每条都能定位到具体字段 |
| 单元 1 合成用例 | 7 组：基本解析、量化张量字节数、坏魔数、坏版本、截断、类型不符、真实文件 |
| 真实文件冒烟 | `models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf`（1.2GB，Q8_0）解析成功 |
| ctest | `gguf` 用例全绿 |

## 第二部分：详细设计

### §6 整体流程图

```text
[GgufFile::load] --失败--> [抛异常]
        |
        v
[GgufFile::parse] --失败--> [munmap + close + 抛异常]
        |
        v
[数据区基址就绪] -> [单元 2/3/5 从这里取词表与权重]
```

失败出口的形状：`load` 里 `parse` 被包在 `try { } catch (...) { munmap; close; throw; }` 中 —— 解析失败时**不留半开文件描述符，也不留悬挂的映射**。

### §7 子流程 + 业务说明

#### 7.1 打开与映射

`open(O_RDONLY)` → `fstat` 拿大小 → `mmap(PROT_READ, MAP_SHARED)` → `madvise(MADV_WILLNEED)`。

1. 映射是**只读共享**的 —— 【为什么】模型文件是只读数据，多个进程可以共享同一份物理页；写权限没有任何用途。
2. `MADV_WILLNEED` 提前预取 —— 【为什么】上游 `llama-mmap.cpp` 也这么做，对大模型文件能显著减少缺页中断带来的抖动。

#### 7.2 头部与计数

1. 魔数读成 `u32` 与 `0x46554747` 比较 —— 【为什么】`"GGUF"` 四个字符按小端解释正好是这个数；一次 `u32` 比较比逐字节循环快，代价是失去「哪个字节错了」的信息（上游逐字节比较并打印实际字符，见 §16.2）。
2. 版本必须 `1 <= v <= 3`。**v1 与 v2/v3 的唯一布局差异是计数宽度**：v1 用 `u32` 存张量数与 KV 数，v2/v3 用 `u64`。
3. 两个计数都来自不可信文件，循环前用 `reserve` 预分配 —— 【陷阱】`n_kv` 若被恶意写成 2^40，`reserve` 会直接抛 `bad_alloc`；本实现靠 `read_value` 里对数组元素数的 `1<<31` 上限做同类防护（见 §8）。

#### 7.3 元数据 KV

1. 每条 KV = `string key` + `u32 type` + 按类型读值；`read_value`（src/gguf.cpp）按 `GgufValueType` 编号分发。
2. ARRAY 的元素类型**不允许是 ARRAY** —— 【概念】ggml 的写出器禁止嵌套数组，所以格式本身不支持；这里显式拒绝而不是递归。
3. 保序存入 `metadata_`。

#### 7.4 张量表与数据区

1. 每条 = `string name` + `u32 n_dims` + `u64 dims[n_dims]` + `u32 type` + `u64 offset`。
2. `n_dims` 必须在 1~4 之间 —— 【为什么】GGML 最多 4 维；0 维张量在格式里无意义。
3. 维度**原样保留**，不重排 —— 【为什么】上游也不重排（`ggml_tensor::ne[]` 就是这个顺序），重排会让「读代码 = 读文件」这条链断掉。
4. 张量表结束后把游标向上对齐到 `alignment` 得到 `data_base_` —— 【对照】上游是 `gr.seek(GGML_PAD(gr.tell(), ctx->alignment))`（ggml/src/gguf.cpp:756）。

#### 7.5 访问器

1. `find` 线性查找返回 `nullptr`；`get_u32` / `get_f32` / `get_string` / `get_array` 走 `get_typed` 模板，**类型检查集中在这一处** —— 【概念】上层（读超参时）拿到的是干净的类型错误，而不是段错误。
2. `tensor_nbytes` 按块取整：`row_blocks = ceil(dims[0] / blck)`，其余维直接相乘 —— 【为什么】量化不是逐元素压缩，末维不满一块也要占整块的空间。
3. `tensor_data` 返回**已对齐**的实际指针，并在返回前再查一次是否越过映射末尾。

### §8 异常流转表

| 异常 | 触发点 | 状态/字段变化 | 恢复路径 |
|---|---|---|---|
| open / fstat / mmap 失败 | `GgufFile::load`（src/gguf.cpp） | fd 已关闭；无映射 | 不可恢复：路径或文件本身有问题 |
| 魔数不匹配 | `parse` 第 ① 步（src/gguf.cpp） | 映射已由 `catch(...)` 回收 | 不可恢复：文件不是 GGUF |
| 版本不支持 | `parse` 第 ① 步（src/gguf.cpp） | 同上 | 不可恢复 |
| 读取越界（截断） | `Cursor::require`（src/gguf.cpp），任何一次 `read` | 同上 | 不可恢复：文件被截断 |
| 元数据类型未知 / 嵌套数组 / 元素数异常 | `read_value`（src/gguf.cpp） | 同上 | 不可恢复 |
| n_dims 非法 | `parse` 张量表段（src/gguf.cpp） | 同上；错误带张量名 | 不可恢复 |
| 张量越出数据区 | `parse` 末尾预检（src/gguf.cpp） | 同上 | 不可恢复 |
| key 缺失 / 类型不符 | `get_typed`（src/gguf.cpp） | 映射保持有效（对象仍可用） | 不可恢复：调用方查错了 key 名 |
| 访问越界 | `tensor_data`（src/gguf.cpp） | 映射保持有效 | 不可恢复 |

### §9 数据形状

```jsonc
// GgufFile 私有字段（src/gguf.h）—— 对应上游 ggml/src/gguf.cpp 的 gguf_context
{ "fd": -1, "map": null, "size": 0,          // int / 映射基址 / 文件字节数
  "version": 3, "alignment": 32,              // u32 / u32（general.alignment 可覆盖）
  "data_base": null,                          // 数据区起点（已按 alignment 对齐）
  "metadata": [ {"key": "...", "value": <GgufValue>} ],   // 保序，几十条
  "tensors": [ {"name": "blk.0.attn_q.weight", "type": "Q8_0",
                "dims": [2048, 2048], "offset": 0} ] }     // dims[0] 最快维在前
// GgufValue（src/gguf.h）—— 13 种标量 + string + 递归数组
{ "v": <variant<uint8_t|int8_t|uint16_t|int16_t|uint32_t|int32_t|
                uint64_t|int64_t|float|double|bool|string|Array>> }
// 块参数表（src/gguf.cpp）—— 对应上游 ggml/src/ggml.c:631 的 type_traits[]
{ "F32": [1, 4], "F16": [1, 2], "BF16": [1, 2], "F64": [1, 8],
  "I8": [1, 1], "I16": [1, 2], "I32": [1, 4], "I64": [1, 8],
  "Q4_0": [32, 18], "Q4_1": [32, 20], "Q5_0": [32, 22], "Q5_1": [32, 24],
  "Q8_0": [32, 34], "Q8_1": [32, 40],
  "Q2_K": [256, 84], "Q3_K": [256, 110], "Q4_K": [256, 144],
  "Q5_K": [256, 176], "Q6_K": [256, 210], "Q8_K": [256, 292] }   // [每块元素数, 每块字节数]
```

## 开发流水线扩展节

### §13 编排锚点与挂载点

| 角色 | 路径 | 承载什么 |
|---|---|---|
| 加载期编排 | src/gguf.cpp `GgufFile::load` | open/fstat/mmap/madvise + 异常安全回收，全流程唯一入口 |
| 解析编排 | src/gguf.cpp `GgufFile::parse` | 五步解析 + 对齐 + 越界预检 |
| 值解析 | src/gguf.cpp `read_value` | 13 种类型的分发；ARRAY 递归 |
| 访问层 | src/gguf.cpp | `find` / `get_*` / `find_tensor` / `tensor_nbytes` / `tensor_data` |
| 类型表 | src/gguf.cpp `ggml_block_params` / :70 `ggml_type_name` | 块参数与类型名 |

**删掉锚点流程还跑不跑得起来**：`GgufFile::load` 是单元 2/3/5 唯一的模型入口，删掉它整个仓库编译不过；`read_value` 是 13 种类型的唯一分发点，把类型分发改到调用方会让每个调用方都要处理 13 种类型。

挂载点表：

| 文件:行 | 改什么 | 为什么必须在这儿 |
|---|---|---|
| src/gguf.h `GgmlType` | 类型编号 | 编号是文件格式的一部分，改了就读不了别人的模型 |
| src/gguf.h `GgufValue` | 值类型集合 | 递归结构必须在这里自洽，variant 的替代类型要 complete |
| src/gguf.cpp `ggml_block_params` | 块参数 | 算字节数的唯一依据；单元 5 的反量化内核靠它对齐 |
| src/gguf.cpp `read_le` | 字节序与对齐读法 | 所有读取都过它，改它等于改整个格式层 |
| src/gguf.cpp `Cursor` | 边界检查 | 唯一的越界防线；任何绕过它的读取都失去保护 |
| src/gguf.cpp `GgufFile::parse` | 解析顺序与对齐 | 五步顺序是格式契约，不能重排 |
| src/gguf.cpp `get_typed` | 类型检查 | 强类型访问器的唯一实现点 |

### §14 概要→详细映射表

| 概要条款 | 详细落点 | 参考实现对应 |
|---|---|---|
| §1.1 `GgufType` / `GgufValueType` / `GgufValue` / `TensorInfo` / `GgufFile` | §7.1~§7.5、§9 | `ggml/include/ggml.h` 的 `enum ggml_type`（编号契约）+ `ggml/src/gguf.cpp` 的 `gguf_context` |
| §1.3 `magic` / `GGUF string` / `alignment` / `blck` | §7.2、§7.4 | `GGUF_MAGIC` / `gguf_reader::read(string)` / `GGUF_DEFAULT_ALIGNMENT` / `ggml_type_traits[].blck` |
| §2 步骤 1-2 | §7.1、§7.2 | `gguf_init_from_file`（ggml/src/gguf.cpp:983）→ `gguf_init_from_reader`（:451） |
| §2 步骤 3 | §7.3 | `gguf_read_emplace_helper`（ggml/src/gguf.cpp:426）+ `ctx->kv` 保序 vector |
| §2 步骤 4 | §7.4 | `gguf_init_from_reader` 的 tensor info 段 + `ctx->info` |
| §2 步骤 5 | §7.4 末尾 | `gguf_get_data_offset`（ggml/src/gguf.cpp:1016）+ `gguf_get_alignment`（:1012） |
| §3 异常 1-8 | §8 前 8 行 | 上游同样在解析期报错并 `gguf_free(ctx)` 返回 nullptr；差别在错误通道（C++ 异常 vs 返回 nullptr + `GGML_LOG_ERROR`） |
| §3 异常 9 | §8 第 9 行 | `gguf_find_key` 返回 -1 / `gguf_get_val_*` |
| §5 观测点 | §5 | 上游没有「解析成功」的日志（安静），错误全走 `GGML_LOG_ERROR` |

### §15 裁量区与待拍板

| AI 可以自决 | 必须对齐参考实现 | 必须问人 |
|---|---|---|
| 解析器用 C++ 异常还是返回错误码 | 术语（名字）：magic / KV / tensor info / alignment / blck | 无（本单元不涉及承重件的砍除结论） |
| 错误消息的措辞与是否带张量名 | 结构档：五步解析顺序 + 元数据保序 + 维度不重排 | 无 |
| `Cursor` 抽成类还是用 lambda | 实现档的替换（判据：能力不减 + 以后接得上）：字节序交换→小端分支、哈希索引→线性查找、type-punned 原始数组→`std::variant`、返回错误码→异常 | 无 |
| 越界预检放在解析末尾还是逐张量读时 | 结构档：只读不写、零拷贝暴露张量字节 | 无 |

## 还原模式扩展节

### §16 对齐面

> 参考实现：llama.cpp @ 88b47a755
> 本文只对「本单元涉及的部分」负责。

#### 16.1 概念对齐

| 参考实现的术语 | 本实现的对应物 | 对齐到什么粒度 | 备注 |
|---|---|---|---|
| `gguf_context` | `GgufFile`（src/gguf.h） | 结构档：调用链骨架 | 上游是 `gguf_context` + `gguf_reader` 两层；本实现合到 `GgufFile` 一个类，成员一一对应 |
| `gguf_type` / `enum ggml_type` | `GgmlType`（src/gguf.h） | 字段职责（编号逐个一致） | 本实现只取 20 种，上游 40+ 种；**编号契约不变**（只能追加不能重排） |
| `gguf_value_type` | `GgufValueType`（src/gguf.h） | 字段职责（13 种编号逐个一致） | 与上游 `enum gguf_type` 的元数据子集完全一致 |
| `gguf_kv` / `ctx->kv` | `metadata_`（`vector<pair<string, GgufValue>>`） | 结构档：保序 | 上游是 vector + 哈希索引；本实现只保序 |
| `gguf_tensor_info` | `TensorInfo`（src/gguf.h） | 字段职责 | 上游多 `n_dims`/`data`（`ggml_tensor`）；本实现把维度拆成 `vector<uint64_t> dims` |
| `GGUF_MAGIC` / `GGUF_DEFAULT_ALIGNMENT` | 魔数常量（src/gguf.cpp）/ `alignment_`（默认 32） | 机制 | 上游 `GGUF_DEFAULT_ALIGNMENT` 同为 32 |
| `ggml_type_traits[].blck` / `.type_size` | `ggml_block_params`（src/gguf.cpp） | 字段职责（每种类型的块参数逐个一致） | 上游 `ggml/src/ggml.c:631` 的 `type_traits[]` 表 |
| `gguf_get_data_offset` / `gguf_get_tensor_offset` | `data_base_` + `TensorInfo::offset` | 机制 | 上游 offset 是**未对齐的相对偏移**；本实现读取时再 `align_up`（见 §16.2） |
| `llama_mmap` | `GgufFile::load` 的 mmap 段（src/gguf.cpp） | 机制 | 上游多了 NUMA 亲和、huge page、预取策略 |

#### 16.2 设计对齐

「结构档的偏离」表：

| 参考实现的机制 | 本实现的位置 | 偏离点 | 为什么可以偏离 |
|---|---|---|---|
| 上游**拒绝 GGUF v1**（ggml/src/gguf.cpp:502「GGUFv1 is no longer supported」） | src/gguf.cpp 支持 v1（u32 计数） | **本实现更宽松**：能读上游已弃的 v1 文件 | 教学价值：v1/v2 的唯一差异是计数宽度，一并支持能把「格式版本差异」讲透；读上游产出的文件行为不变 |
| 上游要求张量 offset **严格等于前序张量填充尺寸的累加**（ggml/src/gguf.cpp:769） | src/gguf.cpp / :401 读取时 `align_up(offset, alignment)` | **本实现更宽松**：接受未对齐的 offset 并在读取时补齐 | 对上游写出的文件两者等价（写出时已 pad，见 ggml/src/gguf.cpp:1637）；本实现宽松是为了能吃下第三方工具写出的文件 |
| 上游校验 `alignment` 必须是 2 的幂（ggml/src/gguf.cpp:616） | 未校验 | 校验缺失 | 上游会直接报错；本实现在 `align_up` 里除以 `a`，若 `a` 非 2 的幂也能算出一个确定的（但可能不是 2 的幂的）对齐值 —— 记为**待补**：见 §16.2 尾注 |
| 上游拒绝空 key 与重复 key（ggml/src/gguf.cpp:565） | 未校验 | 校验缺失 | 重复 key 在本实现里 `find` 返回**第一个**匹配项，与上游的「报错」不同；空 key 无害 |
| 上游检查主机/文件字节序不匹配（ggml/src/gguf.cpp:495 `(version & 0xFFFF) == 0`） | 未检查 | 校验缺失 | 本实现的主机字节序分支只处理「本机大端」，不做「文件端序错配」的探测 |

「实现档的替换」表：

| 参考实现用什么 | 最小版用什么 | 能力是否等价 |
|---|---|---|
| 容器：`gguf_context` + `gguf_reader` + `gguf_kv` 原始数组（type-punned） | `GgufFile` + `Cursor` + `std::variant`（src/gguf.h） | 能力不减：13 种类型 + 递归数组都表达得了；以后接得上：variant → 原始数组是纯替换 |
| 算法：key 用哈希表索引（`gguf_find_key`） | 线性查找（src/gguf.cpp） | 能力不减：语义完全相同；性能差在几十条 KV 的纳秒级；以后接得上：换容器即可 |
| 错误处理：返回 `nullptr` + `GGML_LOG_ERROR` | 抛 `std::runtime_error` + `catch(...)` 回收资源 | 能力不减：错误信息同样带定位信息（本实现带张量名/key 名，上游带字段名）；接得上：可加一层 try 包装成上游风格 |
| 字节序：上游 `gguf_reader` 全程显式小端读 | `read_le` 模板 + `#if __BYTE_ORDER__` 分支（src/gguf.cpp） | 能力不减：小端主机走同一条 memcpy 路径；大端主机有正确分支 |
| 内存：上游 `llama_mmap` 支持 NUMA 亲和 / huge page / 分块预取 | 裸 `mmap` + `madvise(MADV_WILLNEED)` | 能力不减（对本项目规模）：映射语义相同；接得上：换成 `llama_mmap` 即可 |

> §16.2 尾注（待补项）：`alignment` 的 2 的幂校验与「重复 key」校验本单元未实现，
> 已在 `docs/constraints/04-对齐约束.md` 的实现档清单里登记为**可替换但需知会**项 ——
> 一旦模型文件来自不可信来源，这两条属于「防御深度」而非「功能缺失」。

#### 16.3 砍掉清单的承重判断

| 砍掉什么 | 真实版怎么做 | 概念测试 | 行为测试 | 术语测试 | 结论 |
|---|---|---|---|---|---|
| 写 GGUF | `gguf_writer_base` / `gguf_writer_buf` / `gguf_writer_file`（ggml/src/gguf.cpp:1424 / :1531 / :1569） | 写路径不参与推理语义 | 单元 1 的合成用例靠 `tests/gguf_writer.h` **手写字节**反证解析器 —— 不依赖写路径就能验证解析 | 沿用上游 `gguf_writer` / `gguf_writer` 词 | 可砍 |
| 元数据哈希索引 | `gguf_find_key`（ggml/src/gguf.cpp:1024）走哈希 | 查找语义（按 key 取值）不变 | 同一批 key 查出来的值必须一致 —— 线性查找给出相同答案 | 用「KV」而不是「字典」 | 可砍 |
| 大端字节序交换 | 上游 `gguf_reader` 显式小端读 | 小端是 GGUF 格式的规定 | 小端主机上逐字节一致 | 保留「小端 / little-endian」词 | 可砍（本仓库目标平台全为小端） |
| 重叠张量去重 | `find_tensor_by_offset` | 同偏移多张量在 LLaMA 系不存在 | 不影响任何张量的字节内容 | 保留上游「重叠张量 / offset 去重」词 | 可砍 |
| 超参 → 强类型结构体映射 | `llama-hparams.cpp` + `LLM_KV_NAMES` | 这是「形状解释」而非「格式解析」 | 单元 3 独立完成并单独测试 | 保留「hparams」词（单元 3 兑现） | 可砍（**刻意推迟到单元 3**） |
| 张量名通配查找 / lazy 加载 / 用量统计 | `llama_model_loader` 的正则匹配与 `load_data` 回调 | 按名精确取张量已覆盖本项目全部需求 | 单元 3 的 9 张量/层都能精确取到 | 保留「lazy load」词 | 可砍 |

#### 16.4 教学便利代码清单

| 多出来的 | 在哪 | 为什么需要 | 谁 import 它 |
|---|---|---|---|
| GGUF 字节流写入器 `minitest::Writer` | tests/gguf_writer.h:13 | 与 src/gguf.cpp 的解析器互为逆运算，把「格式契约」变成可执行文档；单元 1 的全部坏输入用例靠它现场构造 | tests/test_gguf.cpp、tests/test_tokenizer.cpp、tests/test_model.cpp、tests/test_generate.cpp、tests/test_quant.cpp |
| 独立仲裁者（numpy，从 GGUF 原始字节手写解析） | tests/arbiter.py:1 | 零复用本仓库代码，用来裁决 ours-vs-llama.cpp 的发散归属 | 无（命令行人工运行） |

### §17 踩坑记录

### 17.1 魔数被当成 GGUF string 写进了测试文件 🔴 【坑型：概念层级错位】

**症状**：解析合成测试文件时报「张量 token_embd.weight: n_dims 非法」，但那个文件是用测试里的 Writer 现写出来的，字段一个都没写错。
**根因**：GGUF 里有**两种互不相同的字符串编码**：
```text
magic      = 4 个原始字节 "GGUF"        ← 容器层的裸字节，没有长度前缀
GGUF string = u64 长度前缀 + 原始字节    ← KV 层与张量名的编码
```
测试 Writer 的 `str()` 走的是 GGUF string 分支，于是把 `"GGUF"` 写成了「u64 长度 4 + GGUF」共 12 字节。解析器按裸字节读魔数，读到的前 4 字节是 `04 00 00 00`（长度的小端表示），与 `0x46554747` 不匹配 —— 但报错却出在后面的张量表段，因为长度前缀多出来的 8 字节把整个游标推后了，**报错点离出错点隔了整个文件**。
**修法**：魔数绕开 `str()`，直接用 `w.u32(0x46554747u)` 写 4 个裸字节（tests/test_gguf.cpp:52）；`str()` 只用于真正的 GGUF string。两边各锁一条用例：正常合成文件能解析（tests/test_gguf.cpp:46）、坏魔数能报错（tests/test_gguf.cpp:141）。
**真实版**：上游写侧同样分开处理 —— 魔数是**四次单字节写**（`gw.write(GGUF_MAGIC[0..3])`，ggml/src/gguf.cpp:1618-1621），不是走 `write(std::string)`；读侧的魔数检查也是逐字节比较并打印实际字符（ggml/src/gguf.cpp:468）。本质相同：容器层裸字节与 KV 层字符串是两种编码。
**教学点**：**同一个文件格式里往往并存多种编码，读写两侧必须一一对应**。写测试夹具时最容易犯的错，是「用现成的通用写入函数去写一个特例字段」——通用函数的正确性恰恰不覆盖特例。凡是「裸字节 / 长度前缀 / 定长」三种形态混用的格式，都要为每种形态单开一条写入路径。

### 17.2 递归值的三种写法都不通 【坑型：平台/类型差异】

**症状**：想用 `std::variant<..., std::vector<GgufValue>>` 表达「数组的数组也是数组」，编译器报「incomplete type」/「递归 typedef」/「静态断言失败」三种不同的错。
**根因**：三种尝试各撞一条 C++ 规则 ——
```text
① using GgufArray = std::vector<GgufValue>;   // using 别名不能自引用（未定义标识符 GgufValue）
② struct GgufValue : std::vector<GgufValue>    // vector<不完全类型>：libstdc++ 在实例化成员时拒绝
③ variant 里直接放 std::vector<GgufValue>       // variant 的替代类型必须 complete
```
共同点是：`std::vector<T>` 允许 `T` 不完全，但**只限于 `vector` 自身**；一旦 `T` 出现在 `variant` 的替代类型列表里，就必须 complete。
**修法**：把 `variant` 收进结构体成员 —— `struct GgufValue { using Array = std::vector<GgufValue>; std::variant<..., Array> v; };`（src/gguf.h）。类定义体内的 `vector<自身>` 是合法的，因为 `vector` 的成员函数实例化被推迟到类定义完成之后。
**真实版**：上游是 C 风格的 type-punned 原始数组（`gguf_kv` 里一个 union + 指向原始字节的指针），没有这个语言问题 —— 它用「同一块字节按类型重新解释」绕开了递归类型。本质相同：都在表达「值可以是数组，数组元素又是任意值」。
**教学点**：**这不是「写法不够巧妙」，是语言规则的硬边界**。遇到「类型里要装自己」的递归需求，C++ 的标准解法就是「套一层容器」：`vector` 允许包自己，`variant` 不允许 —— 把 variant 放进类里、类里放 vector，递归就成立了。另外注意 libstdc++ 与 libc++ 在 `vector<不完全类型>` 的宽松度上并不一致，写可移植代码时别指望它。

### 17.3 元数据 KV 计数与实际条数不一致 【坑型：不变式被打破】

**症状**：合成文件解析时报「张量 X: n_dims 非法」，而元数据段（几个 key）明明都读对了。
**根因**：测试里声明「8 条 KV」，实际写了 9 条 —— **同一个事实被写了两遍**（头部计数 + 实际内容），两遍不一致。解析器忠实地按头部计数读了 8 条，第 9 条的字节被当成张量表的第一个字段读走，于是 `n_dims` 读出来是个乱七八糟的数。
**修法**：把头部计数改成实际条数（9，tests/test_gguf.cpp:54）；同时解析器的错误信息一律带上张量名 / key 名（`张量 X: n_dims 非法`），让人一眼看出错在哪个字段而不是「整个文件坏了」。**更彻底的做法是让写入器自动维护计数**（本仓库的 `minitest::Writer` 目前仍是手写计数，见 §16.2「校验缺失」那两行）——上游在**读取侧**做的是交叉校验（`GGML_ASSERT(ctx->info.size() == n_tensors)`，ggml/src/gguf.cpp:753），两者都比「两处手写」更稳。
**真实版**：上游同样把计数写在头部，也同样先信任它（ggml/src/gguf.cpp:525 读 `n_kv` 后按它循环）；区别是上游在读完张量表后有一条**交叉校验** —— `GGML_ASSERT((int64_t(ctx->info.size()) == n_tensors)`（ggml/src/gguf.cpp:753），计数与实际条数不符时立刻断言失败。本实现没有这条断言，是一处可补的加固。
**教学点**：**同一个事实被存两遍的地方，就是不变式被打破的地方**。凡是「头里写计数、内容里写实体」的格式，计数要么由写入器自动维护（不可能写错），要么由读取器交叉校验（写错了也会被发现）。两者都没有时，一个手滑的多写一条，会以「某个毫不相干的字段非法」的形式爆出来 —— 报错点离出错点越远，定位越贵。

### 17.4 大端分支写了但没跑过 【坑型：文档纪律】

**症状**：`read_le` 里有一段 `#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__` 的字节交换分支，注释写着「此分支仅为演示」，但没有任何测试覆盖它。
**根因**：目标平台（x86/ARM）全为小端，这段分支在 CI 上永远不会执行 —— 写它的时候没有同步写测试。
**修法**：当前处置是**明确标注它未被覆盖**（src/gguf.cpp 的注释写明「大端主机需要逐字节交换（【简化】本仓库主要在 x86 上跑，此分支仅为演示）」），而不是假装它是对的；`__builtin_bswap16/32/64` 三个重载是同一套位运算的三个宽度，写法一致、比错概率低。
**真实版**：上游 `gguf_reader` 全程显式按小端读，与主机字节序无关（`gr.read` 内部统一处理），并额外探测「文件与主机端序错配」（ggml/src/gguf.cpp:495）—— 本实现没有这一层探测（见 §16.2）。
**教学点**：**没被测试覆盖的代码，要么删掉，要么在注释里明说它没被覆盖**。写「看起来对」的分支比不写更危险：读者会默认它被验证过。本项目的纪律是：凡是未覆盖的分支，都要在注释里标出来，而不是让它安静地待在那里。
