# 技术文档：单元 5 量化推理

- 读者：想搞清"GGUF 里存着 Q4_K 的字节，前向时怎么变回浮点"的人；以及要照着本仓库复刻一条最小可读推理链的人
- 审法：先看 §9 的块布局图，再看 §7.3 的分发链，最后拿 §17 的坑反查自己的偏移常量对不对
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

| 实体 | 业务含义 | 字段（谁写 / 谁读 / 空值语义） |
|---|---|---|
| `GgmlType`（src/gguf.h） | GGUF 文件里每个张量只存一个 u32 类型编号；编号本身不自描述，读者必须内置同一张编号表 | `type`（枚举值）：写 = GGUF 写入方；读 = src/gguf.cpp 解析 + ggml_block_params；无空值，编号非法时 `ggml_block_params` 抛"未知类型编号 N" |
| `TensorBuf`（src/model.h） | 类型化张量缓冲：量化字节原样保存，不展开成浮点 | `type`（GgmlType）、`ne0`（输入维，最快维）、`ne1`（输出行数 = 输出神经元个数）、`bytes`（原始量化字节）：写 = `load_tensor_buf`（src/model.cpp）；读 = `dequant_column` / `matvec` / `tensor_row_to_f32`；`bytes` 为空即 0 元素张量 |
| 量化块（block） | 量化不是逐元素压缩，而是固定数量元素打成一块、共享一组缩放系数 | 每块元素数 `QK8_0 = 32` / `QK_K = 256`（src/quant.h）；每块字节数由类型决定 |
| `ggml_block_params(type)`（src/gguf.cpp） | 类型 → {每块元素数, 每块字节数}；算张量字节数必须按块取整 | 返回 `std::pair<int,int>`；未知类型抛异常，无空值 |
| 反量化行内核 | 把一段连续的量化字节还原成浮点数组 | 签名 `dequant_row_q*(const uint8_t * p, float * y, size_t n)`；`n` 必须是块大小整数倍（加载期已校验） |

#### 1.2 枚举值对象

`GgmlType` 取值（本单元覆盖的用粗体标出，其余只是占编号、加载到会报错）：

| 编号 | 名称 | 块元素数 | 块字节数 | 本单元支持 |
|---|---|---|---|---|
| 0 / 1 | F32 / F16 | 1 | 4 / 2 | 是 |
| 8 | **Q8_0** | 32 | 34 | 是 |
| 12 / 13 / 14 | **Q4_K / Q5_K / Q6_K** | 256 | 144 / 176 / 210 | 是 |
| 2 / 3 / 6 / 7 / 9 | Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_1 | 32 | 18 / 20 / 22 / 24 / 40 | 否（前向抛"未支持的量化类型"） |
| 10 / 11 / 15 | Q2_K / Q3_K / Q8_K | 256 | 84 / 110 / 292 | 否 |
| 24~28 / 30 | I8 / I16 / I32 / I64 / F64 / BF16 | 1 | 1 / 2 / 4 / 8 / 8 / 2 | 否 |

报错类型只有一种载体：`std::runtime_error`，消息前缀即检索键（见 §5）。

#### 1.3 名词

| 名词 | 一句话大白话 |
|---|---|
| super-block | K-quants 的 256 元素大块，块内再切子块共享缩放系数（Q4_K / Q5_K 切 8 个 32 元素的子块，Q6_K 切 16 个 16 元素的子块） |
| 块元素数 / 块字节数 | 一个量化块装多少个元素、占多少字节；两者相除就是"每元素多少 bit" |
| 6bit scale/min 打包 | Q4_K / Q5_K 把 8 个子块的缩放与偏移（共 16 个数）各存 6 bit，压进 12 字节 |
| nibble | 半字节 = 4 bit；一个字节装两个 4bit 量化值 |
| 反量化 | 把量化字节还原成浮点；推理侧唯一需要的量化操作 |
| 张量级混合量化 | 同一个模型里不同张量用不同量化类型（Q4_K_M：多数张量 Q4_K、输出张量 Q6_K） |
| GGUF 类型编号 | 文件里每个张量一个 u32 的类型标签；只能追加、不能重排 |

### §2 正常流程

```text
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 打开模型文件  │ → │ 认张量类型    │ → │ 查块参数      │
└──────────────┘   └──────────────┘   └──────────────┘
                                                   │
        ┌──────────────────────────────────────────┘
        v
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 校验整块      │ → │ 取权重列      │ → │ 反量化到浮点  │
└──────────────┘   └──────────────┘   └──────────────┘
                                                   │
        ┌──────────────────────────────────────────┘
        v
┌──────────────┐   ┌──────────────┐
│ 与激活做点积  │ → │ 出 logits     │
└──────────────┘   └──────────────┘
```

1. 打开 GGUF 文件并解析出张量表：每个张量带一个 u32 类型编号。
2. 用类型编号查块参数表，拿到"每块几个元素、每块几个字节"。【为什么】张量字节数不能按元素数乘元素宽度算 —— 量化类型一个块 32/256 个元素只占 34/144~210 字节。
3. 按类型把张量的原始字节搬进 `TensorBuf`，同时校验形状：量化张量的输入维必须是块元素数的整数倍。【为什么】反量化内核按整块步进访问，不对齐就会读过头。
4. 前向需要第 o 个输出神经元时，取出权重矩阵的第 o 列字节（列内块连续）。
5. 按类型选反量化内核，把这一列还原成浮点数组。
6. 与激活向量做点积，得到该输出神经元的值。
7. 全部输出神经元算完 → logits。

### §3 异常处理

| # | 异常 | 如何处理 |
|---|---|---|
| 1 | GGUF 文件损坏 / 被截断 | 不可重试。解析期抛 `std::runtime_error`（"文件被截断"），模型对象不构造，进程退出（src/main.cpp）。数据侧：mmap 已建立但全部丢弃 |
| 2 | 张量类型编号不在表里 | 不可重试。`ggml_block_params` 抛"未知类型编号 N"（src/gguf.cpp）。数据侧：加载中止，无部分模型 |
| 3 | 量化张量输入维不是块大小整数倍 | 不可重试。`load_tensor_buf` 抛"张量 X 的输入维 N 不是块大小 M 的整数倍"（src/model.cpp）。数据侧：`TensorBuf` 不写入，模型加载中止 |
| 4 | 张量不是 2 维矩阵 | 不可重试。`load_tensor_buf` 抛"张量 X 必须是 2 维矩阵"（src/model.cpp） |
| 5 | norm 类小向量是量化类型 | 不可重试。`load_tensor_f32` 抛"张量 X 是量化类型 T，FP32 前向不支持"（src/model.cpp）—— norm 向量本就不该被量化 |
| 6 | 前向遇到未覆盖的量化类型（Q4_0 / Q2_K / IQ 系列） | 不可重试。`dequant_column` 抛"matvec: 未支持的量化类型 T"（src/model.cpp）。数据侧：本次前向中止，KVCache 里已追加的 K/V 保留，位置游标不前进 |
| 7 | 取权重行越界 | 不可重试。`tensor_row_to_f32` 抛"tensor_row_to_f32: 行越界"（src/model.cpp） |
| 8 | 形状合法但块里存着 NaN / Inf | 本实现**不检测**：无报错，前向产出 NaN logits。上游有可选的块级值域校验（默认关闭），见 §16.3 第 6 行 |

### §4 Out of Scope

| 砍掉的能力 | 理由（三测结论见 §16.3） |
|---|---|
| 除 Q8_0 / Q4_K / Q5_K / Q6_K 外的约 20 种量化格式（IQ 系列、MXFP4 / NVFP4、Q2_K / Q3_K / Q8_K 等） | §16.3 第 1 行：可砍（新增格式是加法不是改法） |
| SIMD / AVX2 / NEON 向量化反量化与专用 dot 内核 | §16.3 第 2 行：可砍（数值等价，吞吐不属教学目标） |
| 量化侧原位量化（`llama-quantize` 工具与 `quantize_row_*_ref`） | §16.3 第 3 行：可砍（真实版也是"离线工具量化、推理只反量化"） |
| Q3_K 的 super-block scale 再量化（hbits / hmins 路径） | §16.3 第 4 行：可砍（格式裁剪的连带项） |
| 量化侧的按张量类别选型策略（Q4_K_M 配方生成） | §16.3 第 5 行：可砍。**注意**：张量级混合量化的运行侧能力是**保留**的，不是砍掉 |
| 加载期块级值域校验（`ggml_validate_row_data` / `check_tensors`） | §16.3 第 6 行：可砍（上游默认也是关的） |
| 融合式"反量化 + 点积"（上游先量化激活再整块 int8 累加） | §16.3 第 7 行：可砍（本实现用分离两趟，数学等价） |

### §5 观测点

| 时机 | 打的键、携带的字段 |
|---|---|
| 加载期（类型编号未知） | 键 `未知类型编号`；字段：u32 编号 |
| 加载期（块对齐校验失败） | 键 `不是块大小`；字段：张量名、输入维 ne0、块元素数 |
| 加载期（维度校验失败） | 键 `必须是 2 维矩阵`；字段：张量名、dims 长度 |
| 加载期（FP32 路径遇量化） | 键 `FP32 前向不支持`；字段：张量名、类型名 |
| 前向（类型未覆盖） | 键 `未支持的量化类型`；字段：类型名 |
| 前向（行越界） | 键 `行越界`；字段：请求行号、行数 |
| 验收（混合量化前后向对照） | 键 `Q8_0 vs F32`；字段：max\|Δlogit\|、argmax 不一致行数 / 总行数 |
| 验收（真实量化模型） | 键 `quant model`；字段：文件路径、vocab、前向是否有限、argmax_token |

验收基线（原文档验收记录，数字原样保留）：

| 验收项 | 基线 | 出处 |
|---|---|---|
| 混合量化前向：同一组逻辑权重分别存 F32 与 Q8_0 | logits max\|Δ\| = **0.04**，argmax 全部一致 | tests/test_quant.cpp:332-335 |
| 与 libllama 逐数值对照（自建工具直调 `llama_decode`，绕过崩溃的 llama-cli） | 单 token 到 8 token 序列：logits max\|Δ\| ≤ **0.27**、mean\|Δ\| ≤ **0.12**；top1 在非近似并列时一致 | 原文档验收记录 |
| 独立 numpy 仲裁者（从 GGUF 原始字节手写解析 + 前向，零复用本仓库代码） | 与本实现**逐位一致** | tests/arbiter.py |
| 端到端 | `mini-llama --model models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf --prompt "..."` 直接生成文本 | src/main.cpp |

## 第二部分：详细设计

### §6 整体流程图

```text
┌────────────────┐
│ ① 解析 GGUF    │  →  张量表（名字 / 维度 / 类型编号 / 偏移）
└────────────────┘
        │
        v
┌────────────────┐
│ ② 查块参数     │  →  {每块元素数, 每块字节数}
└────────────────┘
        │
        v
┌────────────────┐
│ ③ 装张量+校验  │  →  TensorBuf（原始量化字节）
└────────────────┘
        │
        v
┌────────────────┐
│ ④ 矩阵乘向量   │  →  logits
└────────────────┘
```

失败出口：① 抛"文件被截断" / ② 抛"未知类型编号" / ③ 抛"不是块大小整数倍"、"必须是 2 维矩阵" / ④ 抛"未支持的量化类型"、"行越界"。四条出口全部在进程内向上抛 `std::runtime_error`，由 CLI 统一转成 stderr 的一行错误（src/main.cpp）。

### §7 子流程图 + 业务说明

#### 7.1 认类型、查块参数

```text
[张量条目] → [取出 u32 类型编号] → [查块参数表] → [{块元素数, 块字节数}]
```

1. 从张量表读出类型编号（u32）。编号不自描述，读者必须内置同一张表。【为什么】这是 GGUF 格式与实现之间的隐式契约，编号只能追加不能重排。
2. 用编号查块参数表，拿到每块元素数与每块字节数。
3. 编号不在表里就当场报错，不猜、不按 F32 兜底。【为什么】猜错类型 = 按错误的步长读整张张量表，后面全是错的。

#### 7.2 装张量并校验形状

```text
[维度] → [ne0 % 块元素数 == 0 ?] ─是→ [按块算字节数] → [搬原始字节进 TensorBuf]
                    │
                    否 → 抛"不是块大小整数倍"
```

1. 张量必须是 2 维矩阵，否则报错。
2. 输入维 ne0 必须能被块元素数整除，否则报错。【为什么】反量化内核按整块步进（`p += 144` / `p += 176` / `p += 210`），不整除就会读进下一个张量的头几个字节，被当成 f16 缩放系数。
3. 张量总字节数 = ne0 / 块元素数 × 块字节数 × ne1，按块取整而不是按元素数乘宽度。
4. 把 GGUF 里的原始量化字节整段 memcpy 进 `TensorBuf.bytes` —— 不展开成浮点。

#### 7.3 前向：按类型分发、逐列反量化

```text
[matvec] → [对每个输出神经元 o]
              │
              ├→ [算第 o 列的字节起点]
              ├→ [按类型分发]
              │     F32 → memcpy / F16 → 逐元素转 / Q8_0 → dequant_row_q8_0
              │     Q4_K → dequant_row_q4_k / Q5_K → dequant_row_q5_k / Q6_K → dequant_row_q6_k
              └→ [与激活点积 → y[o]]
```

1. 权重矩阵按"最快维在前"存放：第 o 个输出神经元的那一列是连续的，所以列内块连续、缓存友好度可接受。
2. 第 o 列的字节起点 = o × (ne0 / 块元素数 × 块字节数)。【为什么】列长度必须按块算，不能按元素数乘宽度。
3. 按类型分发到对应反量化内核；类型不在支持列表里就抛"未支持的量化类型"。
4. 反量化一列到临时缓冲（栈上 4096 元素，超过转堆），再与激活做浮点点积。
5. 全部列算完 → 输出向量。

【为什么是"一整列"而不是"一个元素"】：反量化粒度太大就退化成 FP32 的成本，太小就没法用块的缩放系数。列是块的整数倍，恰好是块格式能支持的最自然粒度。

#### 7.4 K-quants 的 6bit scale/min 解包

```text
[块首 4 字节后的 12 字节 scales]
        │
        ├→ j < 4  ?  sc = scales[j] & 63 ,      m = scales[j+4] & 63
        └→ j >= 4 ?  sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
                    m = (scales[j+4] >>  4)    | ((scales[j-0] >> 6) << 4)
```

1. 一个 Q4_K / Q5_K 块有 8 个子块（每个 32 元素），每个子块一个缩放和一个偏移。
2. 8 + 8 = 16 个数每个只存 6 bit：16 × 6 = 96 bit = 12 字节，正好是 `K_SCALE_SIZE`。【为什么】若按 8bit 存要 16 字节，块就胖了 33%；压成 6bit 换来的是解包要分两路。
3. 前 4 个子块各占满一字节（`& 63` 取满 6 bit）；第 5 个往后，缩放的低 4 bit 与偏移的低 4 bit 挤在同一字节，高 2 bit 各自从别的字节第 6-7 位借 —— 所以必须分两路解包。【为什么】打包侧正是这样交叉写的（上游 ggml-quants.c:1595-1599：`scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4)`，多出的 2 bit 塞回 `scales[j-4]` 的高位）。
4. 解出来的 sc / m 与块首的 d / dmin 相乘，得到子块的实际缩放与偏移：`value = d * sc * nibble - dmin * m`。

### §8 异常流转表

| 异常 | 触发点 | 状态/字段变化 | 恢复路径 |
|---|---|---|---|
| 文件被截断 | src/gguf.cpp `Cursor::require` | `cur_` 停在越界前的位置，解析抛异常，`GgufFile` 不构造 | 不可恢复；换文件重来 |
| 未知类型编号 N | src/gguf.cpp `ggml_block_params` | 无字段变化，抛异常 | 不可恢复；补块参数表或换模型 |
| 张量 X 不是 2 维矩阵 | src/model.cpp `load_tensor_buf` | `TensorBuf` 尚未填充 | 不可恢复 |
| 张量 X 输入维 N 不是块大小 M 的整数倍 | src/model.cpp `load_tensor_buf` | `TensorBuf.bytes` 未 resize，模型加载中止 | 不可恢复；文件与实现不匹配 |
| 张量 X 是量化类型 T，FP32 前向不支持 | src/model.cpp `load_tensor_f32` | 局部 `out` 丢弃 | 不可恢复；norm 张量不该被量化 |
| matvec: 未支持的量化类型 T | src/model.cpp `dequant_column` | 已追加进 KVCache 的 K/V 保留，`pos` 不前进 | 不可恢复；补反量化内核 |
| tensor_row_to_f32: 行越界 | src/model.cpp `tensor_row_to_f32` | 输出缓冲未写入 | 不可恢复；调用方行号算错 |
| 块里含 NaN / Inf | 无（本实现不检测） | logits 出现 NaN，向后传播全污染 | 无自动恢复路径；上游可选的 `check_tensors` 可拦（默认关） |

### §9 数据形状

```jsonc
// ① 块参数表（src/gguf.cpp 的 ggml_block_params）：类型 → [每块元素数, 每块字节数]
{
  "F32": [1, 4], "F16": [1, 2], "BF16": [1, 2],
  "Q8_0": [32, 34],                          // 32 元素 / 34 字节
  "Q4_K": [256, 144], "Q5_K": [256, 176], "Q6_K": [256, 210]
  // 未知编号 → 抛"未知类型编号 N"
}

// ② TensorBuf（src/model.h）
{
  "type":  "GgmlType",   // 写：load_tensor_buf；读：dequant_column；无空值
  "ne0":   "uint32",     // 输入维（最快维），必须是块元素数的整数倍
  "ne1":   "uint32",     // 输出行数 = 输出神经元个数
  "bytes": "uint8[]"     // 原始量化字节；长度 = ne0/块元素数 × 块字节数 × ne1
}

// ③ GGUF 张量表条目（src/gguf.h）
{
  "name":   "string",
  "type":   "GgmlType（u32 编号，只追加不重排）",
  "dims":   "uint64[]  // 最快维在前：dims[0] = ne0，dims[1] = ne1",
  "offset": "uint64    // 相对数据区起点；取用时按 alignment 向上对齐"
}
```

```jsonc
// ④ 块字节布局（p = 块首字节地址）
// Q8_0：34 字节 / 32 元素（8.5 bit/元素，FP32 的 3.8x）
// 上游 block_q8_0（ggml/src/ggml-common.h:251-256）
{
  "d":  "f16   @ p+0",   // 块缩放
  "qs": "i8[32] @ p+2"    // 每元素 1 个 int8；value = qs[j] * d
}

// Q4_K：144 字节 / 256 元素（4.5 bit/元素，FP32 的 7.1x）
// 上游 block_q4_K（ggml/src/ggml-common.h:327-337）
{
  "d":      "f16     @ p+0",   // super-block 缩放
  "dmin":   "f16     @ p+2",   // super-block 偏移
  "scales": "u8[12]  @ p+4",   // 8 个子块 × (6bit scale + 6bit min)
  "qs":     "u8[128] @ p+16"   // 256 个 4bit：lo nibble 给前 32 元素，hi nibble 给后 32 元素
  // value = d * sc * nibble - dmin * m
}

// Q5_K：176 字节 / 256 元素（5.5 bit/元素）
// 上游 block_q5_K（ggml/src/ggml-common.h:352-354）：scales → qh[32] → qs[128]
// 本实现内核取址（src/quant.cpp）：          scales → qs[128] → qh[32]
// 两段区域互换，详见 §16.2 偏离表与 §17.9
{
  "d":      "f16     @ p+0",
  "dmin":   "f16     @ p+2",
  "scales": "u8[12]  @ p+4",
  "qh":     "u8[32]  @ p+16（上游） / @ p+144（本实现）",  // 每元素第 5 比特
  "qs":     "u8[128] @ p+48（上游） / @ p+16（本实现）"    // 每元素低 4 bit
  // value = d * sc * (nibble + 16 * 第5比特) - dmin * m
}

// Q6_K：210 字节 / 256 元素（6.5625 bit/元素）
// 上游 block_q6_K（ggml/src/ggml-common.h:362-367），本实现逐字节一致
{
  "ql":     "u8[128] @ p+0",    // 低 4 bit：lo nibble → 行 0，hi nibble → 行 2
  "qh":     "u8[64]  @ p+128",  // 高 2 bit：一个字节的 4 组 2bit 分给 4 行
  "scales": "i8[16]  @ p+192",  // 16 个 8bit 子块缩放
  "d":      "f16     @ p+208"   // super-block 缩放（Q6_K 无 dmin）
  // value = d * sc * (6bit 量化 - 32)
}
```

## 开发流水线扩展节

### §13 编排锚点与挂载点

| 角色 | 路径 | 承载什么 |
|---|---|---|
| 反量化内核（4 个） | src/quant.h 声明 / src/quant.cpp 定义（`dequant_row_q8_0` / `dequant_row_q4_k` / `dequant_row_q5_k` / `dequant_row_q6_k`） | Q8_0 / Q4_K / Q5_K / Q6_K 的块 → FP32 |
| 6bit scale/min 解包 | src/quant.cpp `get_scale_min_k4` | K-quants 块内 8 个子块缩放/偏移的还原（两路） |
| 类型 → 块参数 | src/gguf.h 声明 / src/gguf.cpp 定义 | 加载期算张量字节数、校验整块 |
| 类型编号表 | src/gguf.h `GgmlType` | GGUF 文件里的 u32 类型编号 |
| 量化张量加载与校验 | src/model.cpp `load_tensor_buf` | 原始字节进 `TensorBuf` + 整块校验 |
| 前向分发 | src/model.cpp `dequant_column` / 108-123 `matvec` | 按类型选内核 + 列式点积 |
| 验收夹具 | tests/test_quant.cpp | 4 个内核的手算校验 + 混合量化前向对照 |
| 教学便利 | tests/gguf_writer.h、tests/arbiter.py、tests/test_quant.cpp:193-207、src/main.cpp | 合成 GGUF、独立仲裁者、测试侧 Q8_0 量化器、CLI |

**删掉锚点还跑不跑得起来**：

- 删掉 src/quant.cpp 的四个内核 → 加载期校验还在（模型能加载），但一进前向 `dequant_column` 就抛"未支持的量化类型" → **跑不起来**。
- 删掉 `ggml_block_params` 的量化分支 → 加载期直接抛"未知类型编号 8" → **跑不起来**。
- 删掉 `get_scale_min_k4` → 三个 K-quants 内核编译不过（唯一调用点） → **编译不过**。

挂载点表：

| 文件:行 | 改什么 | 为什么必须在这儿 |
|---|---|---|
| src/gguf.h | 校对 / 新增 GGML 类型编号 | 编号是 GGUF 文件与本实现之间的隐式契约，改编号等于改文件格式 |
| src/gguf.cpp | 新增类型的块参数 | 加载期算字节数与整块校验的唯一来源，别处没有第二份 |
| src/quant.h | `QK8_0` / `QK_K` 常量 | 所有内核按块步进，值错了整条链错位 |
| src/quant.cpp | 6bit 解包的位运算 | K-quants 子块缩放的唯一还原点，打包与解包必须同一份约定 |
| src/quant.cpp | 三个 K-quants 内核的偏移常量 | 块字节布局的承重处，§17 的坑全在这里 |
| src/model.cpp | switch 增加新类型分支 | 反量化入口；加载期只校验"形状"，不校验"能不能反量化" |
| src/model.cpp | 整块校验与字节数计算 | 形状错误的唯一拦截点，越早拦越省事 |
| src/gguf.cpp | `Cursor`（`pos` / `base_` / `require`） | 所有 GGUF 读取的边界检查载体，成员初始化必须写全（§17.8） |
| tests/gguf_writer.h:33-41 | `align()` / `align_data()` | GGUF 数据区双重对齐约定的可执行契约 |
| tests/test_quant.cpp:271-281 | 按实际类型字节数累计张量偏移 | 混合量化模型张量表偏移的唯一算法 |

### §14 概要→详细映射表

| 概要条款 | 详细落点 | 参考实现对应 |
|---|---|---|
| §1.1 `GgmlType` 编号契约 | §9 张量表条目、§7.1 | ggml.h 的 `enum ggml_type` 编号只追加不重排，读侧必须内置同一张表 |
| §1.1 `TensorBuf` 字段职责 | §9、§7.2、§7.3 | `ggml_tensor` 的 ne[] / type / data 三元组：形状、类型、字节 |
| §1.2 量化类型枚举与块参数 | §9 块参数表 | `type_traits[].blck_size` 与 `type_size` 决定一个张量占多少字节 |
| §1.3 super-block 与 6bit 打包 | §7.4、§9 各块布局 | `QK_K = 256` 与 `K_SCALE_SIZE = 12` 的来历：16 × 6bit 正好 12 字节 |
| §2 步骤 1 读类型编号 | §7.1 | gguf.cpp 读张量表条目里的 type u32 |
| §2 步骤 2 查块参数 | §7.1、§9 | `ggml_get_type_traits(type)` 取 `blck_size` / `type_size` |
| §2 步骤 3 加载期校验 | §7.2 | `llama_model_loader::load_all_data` 用 `ggml_nbytes` 核对数据区剩余量 |
| §2 步骤 4 取权重列 | §7.3 | `ggml_vec_dot` 把权重的一列与激活做整块点积，列是 src0 的连续视图 |
| §2 步骤 5 反量化一列 | §7.3、§9 | `dequantize_row_q4_K` / `q5_K` / `q6_K` / `q8_0` 的块遍历与 nibble 拆分 |
| §2 步骤 6 点积累加 | §7.3 | `ggml_vec_dot_q4_K_q8_K` 的 int8 累加（本实现换成 FP32 累加） |
| §3 异常 1 文件截断 | §8 | gguf_reader 的 `nbytes_remain` 剩余字节计数 |
| §3 异常 2 未知类型编号 | §8 | `type_traits[]` 里没有该编号 → 上游 ggml 直接报错 |
| §3 异常 3 块不对齐 | §8 | `dequantize_row_q4_K` 的 `assert(k % QK_K == 0)` |
| §3 异常 4 非 2 维张量 | §8 | ggml 的 2D 矩阵断言 |
| §3 异常 5 FP32 路径遇量化 | §8 | 上游量化器明确不量化 norm 张量（src/llama-quant.cpp:301 "do not quantize norm tensors"），所以 norm 张量永远是 F32 |
| §3 异常 6 类型未覆盖 | §8 | `type_traits[].to_float` 为空指针时的报错路径 |
| §3 异常 7 行越界 | §8 | ggml 的 ne 边界检查 |
| §3 异常 8 值域非法不检测 | §8、§16.3 第 6 行 | `ggml_validate_row_data` 的块级 NaN/Inf 扫描（上游 `check_tensors` 默认 false） |
| §4 范围外能力 | §16.3 | 见 §16.3 七行的三测结论 |
| §5 观测点与验收基线 | §5 | 上游无对应物（本实现的验收面） |

### §15 裁量区与待拍板

| AI 可以自决 | 必须对齐参考实现 | 必须问人 |
|---|---|---|
| 教学便利代码的实现方式：tests/arbiter.py 的 numpy 解析/前向写法、tests/gguf_writer.h 构造器的 API 形态、tests/test_quant.cpp 里手写 Q8_0 量化器的算法、src/main.cpp 的 CLI 形态 | 术语：类型名（Q4_K / Q5_K / Q6_K / Q8_0、super-block、block、nibble）与上游一致，不自创；结构档：分层（GGUF 元数据层 / 张量层 / 块层）+ 各层字段职责；结构档：本单元覆盖到的调用链骨架（加载 → 查块参数 → 校验 → 分发 → 反量化 → 点积）；实现档的替换（判据：能力不减 + 以后接得上） | 砍掉承重件：是否补齐 IQ 系列 / SIMD 内核 / quantize 侧；新增参考实现没有的概念：把测试侧手写 Q8_0 量化器放进 src/ 还是留在 tests/；改对齐粒度定义：Q5_K 块的字段顺序以哪边为准（上游 `block_q5_K` 的 qh→qs，还是本实现的 qs→qh，见 §16.2 与 §17.9） |

## 还原模式扩展节

### §16 对齐面

> 参考实现：llama.cpp @ 88b47a755
> 本文只对「本单元涉及的部分」负责。

#### 16.1 概念对齐

| 参考实现的术语 | 本实现的对应物 | 对齐到什么粒度 | 备注 |
|---|---|---|---|
| `ggml_type`（量化类型枚举） | `GgmlType`（src/gguf.h） | 术语 + 编号一一对应 | 编号是文件格式的一部分，只取本单元覆盖的类型 |
| `block_q8_0` / `block_q4_K` / `block_q5_K` / `block_q6_K`（块结构体） | 原始 `uint8_t *` + 偏移常量（src/quant.cpp） | 字段职责（d / dmin / scales / qs / qh 各是什么、各多宽） | 字段顺序：Q8_0 / Q4_K / Q6_K 无偏离，Q5_K 有偏离，见 §16.2 |
| `QK8_0 = 32` / `QK_K = 256` / `K_SCALE_SIZE = 12` | `QK8_0` / `QK_K`（src/quant.h） | 数值 | `K_SCALE_SIZE` 没有独立常量，落成内核里的字面量 12（`p + 4 + 12`） |
| `get_scale_min_k4`（6bit scale/min 解包） | `get_scale_min_k4`（src/quant.cpp） | 函数体逐行一致 | 两路解包（j<4 / j>=4）都要在 |
| `dequantize_row_q8_0` / `q4_K` / `q5_K` / `q6_K` | `dequant_row_q8_0` / `dequant_row_q4_k` / `dequant_row_q5_k` / `dequant_row_q6_k` | 函数体逐行一致 | 变量名沿用上游（d / dmin / sc / m / ql / qh / is）便于对照 |
| `type_traits[].to_float`（类型 → 反量化函数指针） | `dequant_column` 的 switch（src/model.cpp） | 调用链骨架（按类型取反量化入口） | 机制替换，见 §16.2 |
| `type_traits[].blck_size` / `type_size` | `ggml_block_params`（src/gguf.cpp） | 数值 | Q8_0 = 32/34、Q4_K = 256/144、Q5_K = 256/176、Q6_K = 256/210 全部一致 |
| `ggml_vec_dot_q4_K_q8_K`（量化 × 量化点积） | `matvec` 里"反量化到缓冲 + FP32 点积" | 调用链骨架 | 机制替换，见 §16.2 |
| `llama_model_loader::load_all_data`（加载期数据核对） | `load_tensor_buf`（src/model.cpp） | 分层（加载期 vs 计算期） | 机制简化 |
| `ggml_validate_row_data`（块级值域校验） | 无 | 不适用（本单元无对应物） | 砍掉，见 §16.3 第 6 行 |
| `llama_model_quantize` / `llama-quantize`（量化器） | 无 | 不适用（本单元无对应物） | 砍掉，见 §16.3 第 3 行 |
| 张量级类型选择（每张量一个类型编号，Q4_K_M 之类） | `TensorInfo` 各自带 `type`（src/gguf.h） | 机制完整 | **保留**：混合量化模型天然可加载 |
| `gguf`（文件格式） | `GgufFile` / `TensorInfo`（src/gguf.h） | 分层 + 字段职责 | 见单元 1 文档 |

#### 16.2 设计对齐

结构档的偏离：

| 参考实现的机制 | 本实现的位置 | 偏离点 | 为什么可以偏离 |
|---|---|---|---|
| `block_q8_0` / `block_q4_K` / `block_q6_K` 结构体定义块布局 | src/quant.cpp（`dequant_row_q8_0` / `dequant_row_q4_k` / `dequant_row_q6_k`） | 用 `uint8_t *` + 偏移常量，不定义结构体 | **细节简化**：字节序列完全相同。结构体会把"偏移"变成"字段名"，反而掩盖布局本身 —— 本单元要教的就是布局 |
| `block_q5_K` 字段顺序（d / dmin / scales[12] / qh[32] / qs[128]） | src/quant.cpp | 本实现取址是 scales → qs[128]（p+16）→ qh[32]（p+144），与上游相反 | **概念偏离，当前未对齐**：块字段顺序属于文件格式，不属于实现自由度。后果是读不了上游导出的 Q5_K GGUF。修法见 §17.9，拍板项见 §15 |
| `get_scale_min_k4` 两路解包 | src/quant.cpp | **无偏离**：位运算与上游 ggml-quants.c:880-887 逐行一致 | 不适用（无偏离项） |
| `dequantize_row_q8_0` 的公式与遍历顺序 | src/quant.cpp | **无偏离**：`y[i*qk+j] = qs[j]*d` 与上游一致 | 不适用（无偏离项） |
| `dequantize_row_q6_K` 的 ql / qh / scales / d 取址与拼装 | src/quant.cpp | **无偏离**：ql@p+0、qh@p+128、scales@p+192、d@p+208，与上游 block_q6_K 逐字节相同 | 不适用（无偏离项） |
| `type_traits[]` 的 `to_float` 函数指针表 | src/model.cpp | **机制替换**：函数指针表 → switch 分发 | 能力不减：覆盖的类型一一对应；接得上：新增类型 = 加一个 case + 一个 `dequant_row_*`，不改调用方 |
| `ggml_vec_dot_q4_K_q8_K` 的融合点积（先量化激活，再整块 int8 累加） | src/model.cpp | **机制替换**：融合一趟 → "反量化一列到缓冲 + FP32 点积"两趟 | 能力不减：数学等价，误差只来自浮点累加顺序；接得上：内核签名 `(const uint8_t*, float*, size_t)` 与上游同形，日后换 SIMD 版不改编排 |
| `llama_model_loader::load_all_data` 的数据区核对 | src/model.cpp | **细节简化**：只做"整块对齐 + 越界"两项，不做值域扫描 | 值域扫描上游默认也是关的（`check_tensors` 默认 false，llama-model.cpp:2335） |
| `ggml-quantize` 侧的 `quantize_row_*_ref` | 无 | **砍掉** | 推理路径只读不写，量化是离线工具的职责（§16.3 第 3 行） |

实现档的替换：

| 参考实现用什么 | 最小版用什么 | 能力是否等价 |
|---|---|---|
| `type_traits[].to_float` 函数指针表 | `dequant_column` 里的 switch（src/model.cpp） | **等价**（本单元 6 种类型全覆盖）。接得上：加类型是加法 |
| `block_q4_K` 等 block 结构体 | `uint8_t *` + 偏移常量 + 一段注释 | **等价**（同一字节序列）。接得上：日后要写文件时按同样偏移打包即可 |
| AVX2 / NEON 向量化反量化 + 融合 dot 内核 | 标量版 `dequant_row_*` + 分离点积 | **能力等价、数值等价**（吞吐差 10~50 倍，教学版不关心吞吐）。接得上：内核签名与上游 `dequantize_row_*` 同形 |
| 原位量化（`llama_model_quantize` + `quantize_row_*_ref` + `ggml_quantize_chunk`） | 无（只有测试侧一个手写 Q8_0 量化器） | **推理路径能力等价**（推理只需反量化）。接得上：量化器是工具侧，不与推理路径耦合 |

#### 16.3 砍掉清单的承重判断

| 砍掉什么 | 真实版怎么做 | 概念测试 | 行为测试 | 术语测试 | 结论 |
|---|---|---|---|---|---|
| 1. 约 20 种量化格式（Q2_K / Q3_K / Q8_K、IQ 系列、MXFP4 / NVFP4、FP8 等） | 每个格式一套 block 结构体 + 一套 `dequantize_row_*_ref` + `type_traits` 条目（ggml.c:631） | 类型编号是 GGUF 的自描述契约，但**缺编号只会让加载器报"未知类型"，不会静默错读** —— 缺的是覆盖面，不是正确性 | 本实现遇到未覆盖编号在加载期就抛错（src/gguf.cpp），不会按 F32 兜底错读 | 类型名沿用上游叫法（Q4_K 等），未自创"迷你量化"之类 | **可砍**（4 种已覆盖主流发布格式；新增格式是加法不是改法） |
| 2. SIMD / AVX2 / NEON 向量化与专用 dot 内核 | `ggml_vec_dot_q4_K_q8_K`（ggml-cpu/arch/x86/quants.c:2038）与 `ggml_vec_dot_q4_K_q8_K_generic`（ggml-cpu/quants.c:696） | `vec_dot` 的语义是"先量化激活再整块累加"，与"反量化后 FP32 点积"数学等价 | 与 libllama 逐数值对照 logits max\|Δ\| ≤ 0.27 / mean\|Δ\| ≤ 0.12 —— 差值正是浮点累加顺序的噪声量级，不是算法差 | 内核名（`vec_dot` / `from_float` / `to_float`）沿用上游 | **可砍**（吞吐不属教学目标，能力不减） |
| 3. quantize 侧原位量化 | `llama_model_quantize`（src/llama-quant.cpp:1319）、`llama_tensor_quantize_impl`（:718）、`ggml_quantize_chunk`（ggml.c:7913）、`quantize_row_*_ref` | 真实版的量化是**离线工具**（llama-quantize 一次写文件），推理进程只读不写 —— 推理路径不需要量化能力 | 本实现全程只读；GGUF 写入能力本来就不在单元 1 的范围内 | "量化器 / 反量化"两个词继续分开用，边界未模糊 | **可砍**（推理侧确实用不到） |
| 4. Q3_K 的 super-block scale 再量化 | `quantize_row_q3_K_ref` 里的 hmins / hbits 6bit 再量化，以及 `dequantize_row_q3_K` 里 3bit 变体的 `get_scale_min_k4` | 只在 Q3_K 这一种格式内部成立，不影响其它格式的块结构与共享的解包函数 | 本实现没有 Q3_K 的块结构体，也就没有可被污染的路径 | "super-block" 术语照用上游 | **可砍**（格式裁剪的连带项） |
| 5. 量化侧的按张量类别选型策略（Q4_K_M 配方生成） | `llama_tensor_get_type_impl`（src/llama-quant.cpp:420）按张量类别（attn_v / output / ffn_down）决定类型 | 这是**写文件时**的策略；读文件时每个张量自带一个类型编号，读侧不关心配方 | 读侧完全支持：本实现每个 `TensorInfo` 自带 `type`（src/gguf.h），混合模型直接可加载 | 仍称"张量级类型选择"，未自创 | **可砍**（砍的是生成侧策略，不是运行侧能力） |
| 6. 加载期块级值域校验（`ggml_validate_row_data`） | `ggml_validate_row_data`（ggml/src/ggml-quants.c:5409）在 `load_all_data` 里按 `check_tensors` 开关调用（llama-model-loader.cpp:1397） | 值域检查与形状检查是两层；缺值域检查不影响形状正确性 | 形状检查本实现有（ne0 % 块 == 0、越界），值域不查 —— 损坏文件的表现从"加载期报错"退化为"前向期 NaN"（§3 异常 8） | 仍称 validate_row_data / check_tensors | **可砍**（上游默认也是关的：llama-model.cpp:2335 `check_tensors = false`） |
| 7. 融合式"反量化 + 点积"（先量化激活再整块 int8 累加） | `ggml_vec_dot_q4_K_q8_K` 系列内核 | 两个输入都是量化时，激活先量化到 q8_0 再整块 int8 累加；本实现把权重还原成浮点再 FP32 累加 —— 数学上都是同一笔内积 | 数值对照 max\|Δ\| ≤ 0.27 / mean\|Δ\| ≤ 0.12 即等价性证据 | "vec_dot" / "反量化" 两词照用 | **可砍**（能力不减；只是多一趟内存写读） |

**保留项（不是砍掉）**：「每个张量各自带一个类型编号」这条机制在本实现中原样保留 —— src/gguf.h 的 `TensorInfo` 自带 `type` 字段，src/model.cpp 逐张量记录，因此 Q4_K_M 这类混合发布格式可直接加载。实测：混合量化前向 logits max\|Δ\| = 0.04、argmax 全部一致。§16.3 第 5 行砍掉的是**量化器写文件时的选型策略**，不是这条运行侧机制。

#### 16.4 教学便利代码清单

| 多出来的 | 在哪 | 为什么需要 | 谁 import 它 |
|---|---|---|---|
| 合成 GGUF 字节流构造器 `minitest::Writer`（含 `align()` / `align_data()`） | tests/gguf_writer.h:13-50 | 真实版只有 `gguf_write`（整文件写），没有"逐字段拼字节流并自己控制数据区对齐"的测试夹具；单元 5 要造量化模型就必须能控制数据区起点与张量偏移 | tests/test_quant.cpp（`Builder::write`）、tests/test_gguf.cpp、tests/test_tokenizer.cpp |
| 混合量化模型的张量表偏移构造（按 `ggml_block_params` 的实际块字节数累计） | tests/test_quant.cpp:271-281 | 真实版写文件时自动算偏移；测试要造"同一逻辑权重、F32 与 Q8_0 两份"的模型，必须自己算对偏移 | tests/test_quant.cpp |
| 测试侧手写 Q8_0 量化器 | tests/test_quant.cpp:193-207 | 真实版的量化器在 llama-quantize 工具里；测试需要把已知权重变成 Q8_0 字节，才能构造混合量化模型 | tests/test_quant.cpp |
| 独立 numpy 仲裁者（从 GGUF 原始字节手写解析 + 前向，零复用本仓库代码） | tests/arbiter.py:1-185 | 真实版没有第三方独立实现；用来裁决 ours-vs-libllama 分歧归属 —— 同一份 GGUF 由第三方从字节重新解析并前向，结论与本实现逐位一致 | 人工运行（§5 验收基线） |
| CLI 入口（`--model / --prompt / --temp / --top-k / --top-p / --n / --seed`） | src/main.cpp | 真实版对应 examples/cli 与 llama-cli；单元 5 必须能直接跑量化模型出文本 | 用户手动运行 |
| unicode 表生成器 | tools/gen_unicode_tables.py | 真实版用内建表；本仓库把表生成过程显式化（单元 2 用）。**与本单元无关，仅登记** | 单元 2 的构建流程 |

### §17 踩坑记录

#### 17.1 Q4_K 的 qs 不在 p+4 而在 p+16 —— 漏算 scales 的 12 字节 【坑型：数值边界】

**症状**：照抄"缩放系数 + 量化位"的公式写反量化内核，把 4bit 量化位当成紧跟 d/dmin 的第一个字节开始读 —— 读出来的其实是 scales 的字节，手算的 32 个元素值全部对不上，且越往后的元素错得越离谱（scales 那 12 字节被当量化位吃掉）。

**根因**：Q4_K 块头是 d(f16 2B) + dmin(f16 2B) + scales(12B) + qs(128B) = 144 字节。scales 里装的是 8 个子块的 6bit 缩放/偏移，不是量化位。照抄公式时把 scales 的 12 字节漏算，qs 起点就从 p+16 错到 p+4。

**修法**：内核里把两个偏移常量显式命名并成对出现：`const uint8_t * scales = p + 4;` / `const uint8_t * q = p + 4 + 12;`（src/quant.cpp）。测试构造块按同一顺序填：d@0、dmin@2、scales@4、qs@16（tests/test_quant.cpp:100-107）。

**真实版**：上游不写偏移而写结构体 `block_q4_K { ggml_half d; ggml_half dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; }`（llama.cpp @ 88b47a755 的 ggml/src/ggml-common.h:327-337），字段顺序由编译器保证，偏移写错在编译期就露出来；反量化内核 ggml-quants.c:1529 的 `dequantize_row_q4_K` 直接取 `x[i].qs`。

**教学点**：块格式的"字段顺序"是文件格式的一部分，不是实现自由度。用裸指针 + 偏移写内核省掉了结构体样板，却把布局错误从编译期挪到了运行期 —— 所以偏移常量必须成对出现（同一个 12 字节长度既出现在读 scales 处，也出现在算 qs 起点处）。

#### 17.2 K-quants 的 16 个 scale/min（8 个子块 × 2）是 6bit 打包，j >= 4 走另一条解包路径 【坑型：混合】

**症状**：手构造一个 Q4_K 测试块时按"每字节一个 scale"填 scales[12]，只有前 4 个子块的值对，后面的子块全是 0 或乱值；反过来拿真实模型字节跑又"看起来对"（因为前 4 个子块权重最大，误差被掩盖）。

**根因**：8 个子块各有一个缩放、一个偏移，共 16 个数要压进 12 字节 = 96 bit，所以每个数只存 6 bit（16 × 6 = 96）。打包分两路：j < 4 时 `sc = q[j] & 63`、`m = q[j+4] & 63`；j >= 4 时 6 bit 横跨两个字节 —— 低 4 bit 取 `q[j+4] & 0xF`，高 2 bit 从 `q[j-4]` 的第 6-7 位借。

**修法**：解包逻辑两路都要写（src/quant.cpp，与上游 ggml-quants.c:880-887 的 `get_scale_min_k4` 逐行一致）；测试必须两路都测 —— 子块 0..3 走 j<4 分支、子块 4..7 走 j>=4 分支，只测一路等于没测。

**真实版**：上游把两路打包写在量化侧（ggml-quants.c:1595-1599 的 `quantize_row_q4_K_impl`：`scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4)`，多出的 2 bit 塞回 `scales[j-4]` 的高位），解包侧是同一个 `get_scale_min_k4` —— 打包与解包在同一位布局约定上闭环。

**教学点**：6bit 打包是为了把 8bit 的子块缩放再压 25%，代价是"一个字节不是一个数"。手写测试块时按字节直觉填，必然只覆盖一条解包路径。

#### 17.3 Q6_K 的 ql / qh 横跨多个 32 元素行，改一个字节要重算多个元素 【坑型：混合】

**症状**：手算 Q6_K 测试块的期望值时按"ql[l] 只影响元素 l"算，结果某些元素莫名也变了；改一个字节后"只改了预期的那一个元素"，实际好几个元素一起变。

**根因**：Q6_K 块 210 字节 = ql[128] + qh[64] + scales[16] + d(f16)。ql 一个字节的 lo nibble 喂给元素 l（行 0），hi nibble 喂给元素 l+64（行 2）；qh 一个字节的 4 组 2 bit 分别喂给 l / l+32 / l+64 / l+96 四行。一个 ql 字节横跨两行，一个 qh 字节横跨四行。

**修法**：测试预期要按"改一个字节 → 多个元素受影响"全量重算。tests/test_quant.cpp:162-170 就是这么写的：把 ql[0] 改成 0x21（lo = 1、hi = 2）、qh[0] 改成 0x01 后，单独断言 y[0] = -7.5、y[64] = -15.0，并在其余元素的循环里显式跳过被牵连的 l == 64。

**真实版**：上游 `dequantize_row_q6_K`（ggml-quants.c:1939-1968）里 q1..q4 的取法就是这条规则：q1 用 `ql[l] & 0xF` 加 `qh[l]` 的第 0-1 位、q2 用 `ql[l+32] & 0xF` 加第 2-3 位、q3 用 `ql[l] >> 4` 加第 4-5 位、q4 用 `ql[l+32] >> 4` 加第 6-7 位，分别写到 y[l] / y[l+32] / y[l+64] / y[l+96]。本质完全相同。

**教学点**：K-quants 的块里同时存在两种切分 —— Q4_K / Q5_K 的 scales 按 32 元素分组（8 个子块），Q6_K 的 scales 按 16 元素分组（16 个子块）。切分维度不同是这类格式的常态，测试预期必须按切分维度重算，不能按字节顺序线性推。

#### 17.4 Q4_K 一个字节同时供两个 32 元素组用，只验前 32 个等于没验 【坑型：测试自身陷阱】

**症状**：Q4_K 的反量化测试只对前 32 个元素做断言就全绿，但 hi nibble 那一路的解包写错了也照样绿。

**根因**：qs[128] 里每字节两半：lo nibble 给本组的 32 个元素，hi nibble 给紧接着的 32 个元素（每 64 元素一组推进，每组消费 32 字节）。只验前 32 个元素，等于每字节只用到了 lo 半。

**修法**：断言必须覆盖全部 256 个元素。tests/test_quant.cpp:110-117 先验 0..31 的解析式（`2.0 * (l % 16) - 1.0`），再验 32..255 全为 0（其余子块 scale = 0），lo / hi 两路都被覆盖。

**真实版**：上游 `dequantize_row_q4_K`（ggml-quants.c:1544-1551）在同一个 32 次循环里分别写 `q[l] & 0xF` 与 `q[l] >> 4` 两行，lo / hi 两路同时产出；只验一半必然漏。

**教学点**：这类"一个字节两个用途"的格式，测试覆盖率要按**语义单元**（元素）算，不能按字节数算。

#### 17.5 形状错误要在加载期拦下，而不是让它在前向期变成 NaN 【坑型：生命周期】

**症状**：把一个 ne0 不是块大小整数倍的量化张量塞进 GGUF，若不校验，前向跑起来后 logits 全是 NaN，而且看不出是哪一维越界；等到最后一步才发现，排查成本极高。

**根因**：反量化内核按整块步进访问（`for (i = 0; i < n / QK_K; i++) { ... p += 144; }`），n 不是块大小整数倍时最后一块会读过头 —— 读到的是文件里下一个张量的头两个字节，被当成 f16 的 d，缩放系数于是变成天文数字或 NaN。这类错误发生在"计算期"，那时张量表已被消费、KVCache 已写了半截，错误现场和成因隔得很远。

**修法**：加载期就按块参数校验形状：`load_tensor_buf` 里 `buf.ne0 % blck_elems != 0` 直接抛"张量 X 的输入维 N 不是块大小 M 的整数倍"（src/model.cpp），未知类型编号在 `ggml_block_params` 就抛（src/gguf.cpp）。张量字节数也按块取整算（src/gguf.cpp）。

**真实版**：上游有同一道关的三个落点：计算期 `assert(k % QK_K == 0)`（ggml-quants.c:1530，assert 在 NDEBUG 下会被预处理掉，Release 构建默认定义 NDEBUG）；加载期 `llama_model_loader::load_all_data`（llama-model-loader.cpp:1402）用 `ggml_nbytes`（ggml.c:1297，量化类型按 `ne[0]*nb[0]/blck_size` 整除算字节数）逐个核对数据区剩余量，越界即报错。

**教学点**：同一个不变式（n 必须是块大小整数倍）在上游有两个落点 —— 计算期 assert 快但会被 Release 关掉，加载期字节数比对慢但永远在。教学版选了后者：报错要发生在还能说清"哪个张量、哪一维"的时候。

#### 17.6 混合量化模型的张量表偏移必须按实际类型字节数算 【坑型：数值边界】

**症状**：F32 版模型正常，掺进一个 Q8_0 张量后整个模型输出乱码 —— 不只是那个张量错，是**所有**张量都错（词表、norm、每一层权重全乱）。

**根因**：测试构造器最初把张量偏移写死成 `data.size() * 4`（每元素 4 字节）。Q8_0 块 32 个元素只占 34 字节，不是 128 字节；偏移一错，后面每个张量的起点都往前挪，第一个错位之后全错位。

**修法**：偏移必须按张量自己的类型算：`auto [be, bb] = ggml_block_params(t->type); size_t bytes = t->data.size() / be * bb;`（tests/test_quant.cpp:277-279），累加后再按 alignment 对齐（tests/test_quant.cpp:280）。仓库侧的张量字节数同样按块取整（src/gguf.cpp 的 `tensor_nbytes`）。

**真实版**：上游写文件时用 `ggml_nbytes` 算每个张量的字节数，offset 逐个累加并按 alignment 补齐（ggml/src/gguf.cpp:1411：`ctx->info[i].offset = ctx->info[i-1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i-1].t), ctx->alignment)`），读侧用同一个 `ggml_nbytes` 反查，两边同源不会漂。

**教学点**：GGUF 的张量表只存 offset，不存字节数 —— 字节数必须由"类型 + 维度"重新算出来。所以任何自己写 GGUF 的工具都必须内嵌同一张块参数表；写死常量等于把格式契约钉死在一个会过期的数字上。

#### 17.7 GGUF 数据区的对齐有两层，只做一层就整体错位 【坑型：概念层级错位】

**症状**：合成 GGUF 时第一个张量解出来完全正确，第二个张量开始全是垃圾 —— 越往后越离谱。单独把第二个张量的字节 dump 出来看，内容是有意义的，就是起点差了几十字节。

**根因**：GGUF 数据区有**两重**对齐约定叠在一起：① 数据区起点本身要绝对对齐到 alignment（默认 32，gguf/include/gguf.h:46 的 `GGUF_DEFAULT_ALIGNMENT`）；② 每个张量的 offset 是**相对数据区起点**的累计值，张量之间再各自按 alignment 补齐。只做 ① 不做 ②（只把数据区起点对齐、张量紧挨着写），第二个张量起全部错位；只做 ② 不做 ①，则 offset 的基准点本身就不对。

**修法**：构造器分两个对齐动作：`w.align()` 把数据区起点绝对对齐；`w.align_data(data_start)` 在写完每个张量后按"相对数据区起点"补齐（tests/gguf_writer.h:33-41）。仓库侧的对应实现是 `data_base_ = c.base() + align_up(c.pos(), alignment_)`（src/gguf.cpp）与 `align_up(info.offset, alignment_)`（src/gguf.cpp）。

**真实版**：上游写侧用 `gr.seek(GGML_PAD(gr.tell(), ctx->alignment))`（ggml/src/gguf.cpp:756）对齐数据区起点，读侧用同一套 `GGML_PAD` 累加 offset（gguf.cpp:1411）；`GGML_PAD` 就是把 x 向上取整到 a 的倍数。数据区起点与张量 offset 是两个不同基准，写读两侧必须能独立推出同一个起点。

**教学点**：对齐不是"一个地方对齐一下"，而是"每个基准点各有一条规则"。写文件的人和读文件的人必须各自独立推出同一个起点 —— 少想一层，两个人的文件就永远对不上。

#### 17.8 Cursor 构造函数漏初始化 base_：ASan 全过、Release 挂 🔴 【坑型：不变式被打破】

**症状**：真实模型测试随机越界 / NaN；同一个二进制有时能跑出正确结果，有时崩。更迷惑的是：ASan 构建全过，Release 构建反而挂。

**根因**：`Cursor` 的构造函数漏初始化成员 `base_`，而 `pos()` 返回 `cur_ - base_`。base_ 未初始化时读到的是栈上的垃圾值 —— 新进程里那块栈恰好干净（或残留上次运行的值），于是"碰巧正确"；跑过其它代码后那块栈被覆写，`pos()` 返回天文数字，`data_base_ = c.base() + align_up(c.pos(), alignment_)`（src/gguf.cpp）算出的是一个完全离谱的指针，后面的越界检查（src/gguf.cpp）随之全部失效。ASan 全过是因为它把未初始化栈填成 0xBE，0xBE 凑出的整数恰好有限，不会触发它自己的越界检测；Release 反而因为栈内容不同而暴露。

**修法**：构造函数初始化列表逐个写全：`base_(base), cur_(base), end_(base + size), what_(std::move(what))`（src/gguf.cpp）。定位过程值得记：加打印发现 `pos()` 返回天文数字 → 定位到未初始化读。

**真实版**：上游 GGUF 的读侧是 `gguf_reader`（ggml/src/gguf.cpp:230-243）：靠 callback 拉数据、用 `nbytes_remain` 计剩余字节，构造函数成员初始化列表写全，也没有"基准指针相减"这回事 —— 所以上游没有这条坑。它是本实现为了让解析器自带边界检查（`Cursor::require`）而引入的新风险面。

**教学点**：三条教训：① 成员初始化列表逐个 eyeball，漏一个就是随机行为；② "ASan 过、Release 挂"优先怀疑未初始化读（ASan 的 0xBE 填充恰好是有限值，把问题掩盖了）；③ 能跑 ≠ 正确 —— 栈残留的假象必须靠"进程内多次走不同路径"复现才能戳穿。

#### 17.9 Q5_K 的 qh / qs 字段顺序与上游相反 【坑型：混合】

**症状**：Q5_K 的单元测试全绿（手构造的块与本实现内核自洽），但把上游 `block_q5_K` 的字段顺序逐字节对上时发现：本实现把 p+16 当作 qs、p+144 当作 qh；上游是 p+16 = qh[32]、p+48 = qs[128]。两段区域正好互换 —— 按字节序推导，用上游导出的 Q5_K GGUF 加载时，第 5 比特与低 4 bit 会互相串位。

**根因**：上游 `block_q5_K` 的字段声明顺序是 d / dmin / scales[12] / qh[32] / qs[128]（llama.cpp @ 88b47a755 的 ggml/src/ggml-common.h:352-354），qh（每元素第 5 比特）排在 qs（低 4 bit）前面。本实现按 Q4_K 的"scales 后面紧跟 qs"直觉排布，把 qs 放在 p+16、qh 放在 p+144（src/quant.cpp）。Q4_K 恰好没有 qh，所以同一个直觉在 Q4_K 上是对的、在 Q5_K 上就错了 —— 手写单元测试又是拿本实现自己的布局构造块，于是自洽地全绿。

**修法**：把两个指针的取址对调回上游顺序：`qh = p + 4 + 12`（32 字节）、`qs = p + 4 + 12 + 32`（128 字节），并同步改测试构造块的填充偏移（tests/test_quant.cpp:130-132）。该改动列为待拍板项（§15）。

**真实版**：上游 `dequantize_row_q5_K`（ggml-quants.c:1731-1755）取的是 `x[i].qs` 与 `x[i].qh` 两个成员指针，字节位置由结构体声明顺序唯一确定；写侧 `quantize_row_q5_K_impl` 用同一顺序落盘（ggml-quants.c:1825-1826：`uint8_t * qh = y[i].qh; uint8_t * ql = y[i].qs;`）—— 写读两侧共用一份布局定义，本质完全相同。本实现没有这份共享定义，同一偏移在两处各写一遍，就可能写反。

**教学点**：自洽的单元测试只能证明"实现与自己一致"，证明不了"实现与文件格式一致"。块格式的字段顺序最好有**唯一**定义（上游用结构体），一旦拆成两处裸偏移，就失去了"编译器替你检查"这层保险。
