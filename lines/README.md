# lines/ —— 按业务线看 mini-llama

> **一条线 = 一个端到端子流程。** 不是目录导览，是真的跑一遍。

## 为什么要有线

mini-llama 的代码按**机制**分层：`gguf` / `tokenizer` / `unicode` / `model` /
`kvcache` / `sampler` / `quant`。但一次真实的推理是**竖着穿**这些层的：

```text
--model X.gguf ──▶ 解析 GGUF ──▶ 映射超参 ──▶ 装载权重 ──▶ 前向 ──▶ 采样 ──▶ 输出
```

这条竖线**不属于任何一个目录**——每一层都不知道自己是从哪个业务入口被调进来的。
所以研究和沉淀都按**线**组织：一条线 = 一个端到端子流程 = 一份线文档 + 一个可运行的线代码。

线与单元（`docs/01~05`）是**正交**的两套切法：

| 切法 | 回答的问题 | 产物 |
|---|---|---|
| **单元**（docs/） | 「llama.cpp 的哪个能力被还原了，对照面是什么」 | 对照文档 + 单元测试 |
| **线**（lines/） | 「一次推理从头到尾是怎么跑起来的」 | 线代码（可运行 + 逐跳观测） |

## 目录约定

```text
lines/
├── README.md              全局地图：每族有哪些线、每线一句话、校验条数
├── _shared/               公共 harness（LineCtx + 注册表 + main），不含任何线的逻辑
│   ├── line.hpp             日志 / 观测 / 轻量自检 + 注册宏 + 入口宏
│   ├── line.cpp             注册表实现 + 两个入口壳
│   ├── lines.hpp            8 条线的声明 + 模型文件定位
│   ├── lines.cpp            模型文件定位的实现
│   └── main.cpp             合集入口（--list / <线名> / all / --step）
├── load/                  载入域
│   ├── README.md
│   ├── model_load_line.cpp
│   └── vocab_load_line.cpp
├── text/                  文本域
│   ├── README.md
│   ├── tokenize_line.cpp
│   └── detokenize_line.cpp
├── infer/                 推理工域
│   ├── README.md
│   ├── prefill_line.cpp
│   └── decode_line.cpp
└── numeric/               数值域
    ├── README.md
    ├── dequant_line.cpp
    └── sample_line.cpp
```

四条边界规则（**违反 = 线白串了**）：

| 东西 | 放哪 |
|---|---|
| 被研究项目的真实能力 | `src/`（各机制文件） |
| **串线时发现缺的能力** | **补进 `src/`**，再回来串线 |
| 线本体（串线代码 + 逐跳日志 + 轻量自检） | `lines/<族>/<线名>_line.cpp` |
| 公共 harness | `lines/_shared/` |

**`lines/` 里不出现被研究项目的逻辑。** 写线时如果需要「顺手实现一下」，
说明 `src/` 缺东西——补到 `src/`，再回来串。

## 每族的线清单

| 族 | 线 | 一句话 | 校验条数 |
|---|---|---|---|
| **load** 载入域 | `model_load` | GGUF 文件 → 可前向的 Model（头/元数据/张量表 → 超参映射 → 权重装载） | 8 |
| | `vocab_load` | GGUF 元数据 → 可用的 Tokenizer（词表 + 索引 + BYTE 映射 + 特殊 token） | 8 |
| **text** 文本域 | `tokenize` | prompt 文本 → token id 序列（归一化 → 切词 → 按 score 合并） | 3 |
| | `detokenize` | token id 序列 → 文本（按类型决定渲不渲染 → BYTE 还原 → ▁ 逆变换） | 4 |
| **infer** 推理工域 | `prefill` | token 序列 → 写满 KV Cache 的 logits（增量前向 ≡ 全量前向） | 9 |
| | `decode` | 末位 logits → 采样 → 新 token → 写回 KV Cache → 循环到 EOS | 5 |
| **numeric** 数值域 | `dequant` | GGUF 量化字节 → 逐块还原成 F32 权重 → 进入 matvec | 10 |
| | `sample` | logits → softmax → top-k → top-p → 温度 → 按概率抽样 | 11 |

合计 **58 条校验**。

## 推荐阅读顺序

| 阶段 | 顺序 | 线名 | 优先级 | 重点读哪些内容 | 核心价值 |
|---|---|---|---|---|---|
| 1 进来 | 1 | `model_load` | ★★★ | 第 3 跳（架构前缀映射）、第 5 跳（双重对齐） | 文件格式 → 内存模型的映射，**整条链的地基** |
| 1 进来 | 2 | `vocab_load` | ★★☆ | 第 2 跳（SPM/BPE 分派）、第 5 跳（BYTE 三级查找链） | 词表是「数据 + 算法」的合体 |
| 2 文本 | 3 | `tokenize` | ★★★ | 第 3 跳（归一化顺序）、第 4 跳（预分词切分） | 「同一段文本为什么能拆成 15 个 token」 |
| 2 文本 | 4 | `detokenize` | ★☆☆ | 第 3 跳（token 类型决定渲不渲染） | 逆变换比正变换更容易被忽略 |
| 3 推理 | 5 | `prefill` | ★★★ | 第 6 跳（核心不变式）、第 7 跳（因果性） | **KV Cache 的正确性论证** |
| 3 推理 | 6 | `decode` | ★★★ | 第 4~6 跳（采样 → 前向 → 写回） | 生成循环的骨架 |
| 4 数值 | 7 | `dequant` | ★★☆ | 第 2 跳（块布局）、第 3~5 跳（三种 K-quant 的公式） | 量化格式 = 数据结构 + 位运算 |
| 4 数值 | 8 | `sample` | ★★☆ | 第 6 跳（top-p 含越界那一个） | 分布裁剪的语义 |

## 走线五步法

每条线都是这五步走出来的：

| 步 | 做什么 | 产出 | 在本仓库的落点 |
|---|---|---|---|
| 1 跑通 | 让这条流程真实发生一次 | 可复现的入口 | `line_<线名>` 二进制 + `MINILLAMA_MODEL` 环境变量 |
| 2 画图 | 粗粒度链路总图，圈出跨层点 | 链路总图 | 各族 README 的「链路图」节 |
| 3 逐跳 | 每跳一张三段卡片（是什么 → 源码锚点 → 观测证据） | 逐跳卡片 | 线代码里的 `ctx.step_at(n, 标题, 源码, 验证)` |
| 4 找动机 | 只在跨层点问「为什么不用更简单的做法」 | 每点一句话 | 各族 README 的「学习问题的答案」节 |
| 5 写疑问 | 疑点、疑似 bug、与文档不符处 | 反哺清单 | 各族 README 的「认知纠错记录」节 |

**每一步都不许跳**：没有第 1 步的「真实发生过」，第 3 步的锚点可能是死代码；
没有第 3 步的锚点，第 4 步的动机是空谈。

## 怎么跑

```bash
# 0) 先构建
cmake -S . -B build -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6

# 1) 看有哪些线
./build/mini-llama-lines --list

# 2) 跑一条（需要模型文件的线要指到 GGUF）
MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_model_load

# 3) 逐跳单步（每跳停一下等回车）
MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_prefill --step

# 4) 全跑一遍（退出码 = 不符条数，0 = 全过）
MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/mini-llama-lines all

# 5) 单跑某条也可以用合集入口
./build/mini-llama-lines sample
```

模型文件不在默认位置时用 `MINILLAMA_MODEL` 指过去；**没有模型时相关线会「跳过」**
（不是失败——缺外部条件不是线写错了）。

## 自动化

```bash
ctest --test-dir build --output-on-failure          # 全部（含 lines）
ctest --test-dir build -R lines --output-on-failure # 只跑线（秒级，模型线跳过）
ctest --test-dir build -R lines_model                # 只跑带模型的重测试（约 1~2 分钟）
```

| ctest 用例 | 跑什么 | 耗时 |
|---|---|---|
| `lines` | 8 条线全跑，需要模型的线自动跳过 | ~0.03 秒 |
| `lines_model` | 8 条线全跑，挂真模型（**仅当 `models/` 下有 GGUF 时才注册**） | ~95 秒 |

## 加一条新线的步骤

1. 在 `lines/<族>/` 下新建 `<线名>_line.cpp`，头部注释写全五要素
   （一句话主线 / 对照源码清单 / 跨模块清单 / 学习问题 / 跑法）
2. 线本体写 `int minillama::lines::run_<线名>(LineCtx&)`——
   **辅助函数收在匿名 ns 里**，线本体要在 `minillama::lines` 命名空间里可达
   （测试的绿箭头要能落在它身上）
3. 在 `lines/_shared/lines.hpp` 声明它，在 `lines/_shared/lines.cpp` 旁边没有额外依赖
4. 文件末尾加自注册 + 独立 main：

   ```cpp
   MINILLAMA_LINE_REGISTER(reg_<线名>, "<线名>", "<一句话>", ::minillama::lines::run_<线名>);
   #ifdef MINILLAMA_LINE_STANDALONE
   MINILLAMA_LINE_STANDALONE_MAIN("<线名>", ::minillama::lines::run_<线名>)
   #endif
   ```

5. `CMakeLists.txt` 的 `MINILLAMA_LINE_NAMES` / `MINILLAMA_LINE_FILES` 两个平行列表里各加一行
   （CMake 的 `set()` 引号不保护分号，所以用两个列表 + 下标配对）
6. `tests/test_lines.cpp` 里加一行 `run_line("<线名>", ...)`
7. 在本文件「每族的线清单」和对应族 README 里补上这条线

## 反哺清单

走线时发现的疑点记进**对应族 README 的「认知纠错记录」节**，写清文件位置。
**就地去改 `src/` = 这条线白串了**——线是研究产物，发现的问题要回写到
`docs/constraints/` 或单元文档，不是就地偷偷改。
