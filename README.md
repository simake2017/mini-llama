# mini-llama：llama.cpp 核心能力最小还原

用 C++17（仅标准库 + POSIX mmap）从零还原 llama.cpp 的推理主干：

第一次接触本项目，请先读[产品与技术指南](docs/guide/README.md)：从“它是什么、能做什么”开始，分为产品篇 4 章、技术篇 7 章，结合输入、输出和中间过程实例说明。

| 单元 | 能力 | 对照真实版 |
|---|---|---|
| 1 | GGUF 模型加载 | ggml/src/gguf.cpp + llama-model-loader.cpp |
| 2 | BPE 分词器 | src/llama-vocab.cpp |
| 3 | 前向计算（RMSNorm/RoPE/GQA/SwiGLU） | src/llama-model.cpp + ggml |
| 4 | KV Cache + 生成循环 + 采样 | src/llama-kv-cache*.cpp + llama-sampler.cpp |
| 5 | 量化推理（Q8_0 / Q4_K / Q5_K / Q6_K） | ggml/src/ggml-quants.c 反量化内核 |

## 构建与测试

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
ctest --test-dir build --output-on-failure
```

## 端到端运行

下载任意 LLaMA 系 GGUF（如 TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf，约 1.2GB）放入
`models/`，然后：

```bash
./build/mini-llama --model models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf \
    --prompt "The capital of France is" --n 64 --temp 0.8
```

数值正确性已与 llama.cpp（libllama 逐 logits 对照 + 独立 numpy 仲裁者）验证，
见 docs/03、docs/05 的验收记录；`tests/arbiter.py` 是从 GGUF 原始字节独立实现的
前向参考，可用于复现对照。

## 明确不做

训练、多架构、多后端（CUDA/Metal 等）、GGML 计算图、投机解码、多模态、LoRA、语法采样。
**每一条都有承重判据结论**（能不能砍 / 为什么没塌），见 `PROJECT.md` 的 Out of Scope 表
与 `docs/constraints/04-对齐约束.md`。

## 文档

| 位置 | 内容 |
|---|---|
| `PROJECT.md` | **还原项目声明**：参考实现版本、Must Have / Out of Scope 表、对齐粒度、教学便利代码清单 |
| `docs/01`~`docs/05` | 5 篇单元技术文档（tech-doc v1.1 骨架 §1~§15 + 还原模式扩展节 §16 对齐面 / §17 踩坑记录） |
| `docs/constraints/04-对齐约束.md` | **权威约束**：承重判据（能不能砍）+ 24 条砍掉结论 + 对齐粒度落地清单 |
| `docs/constraints/05-注释规范.md` | **权威约束**：四段式注释 + 还原项目教学化叠加层（教学标签 / 3 档用例 / 日志分级） |
| `lines/` | **业务线**：把推理主干拆成 8 条端到端子流程，每条线可单跑、可自动校验 |

### 走业务线（推荐入口）

`lines/` 把「一个文件 → 一段文本」拆成 4 族 8 条线，每条线一个可执行程序：

```bash
./build/mini-llama-lines --list                      # 列出全部 8 条线
MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf \
    ./build/mini-llama-lines prefill --step           # 走一条线，逐步打印每跳的观测
```

| 族 | 线 | 一句话 |
|---|---|---|
| **load** 载入域 | `model_load` / `vocab_load` | GGUF 文件 / 元数据 → 可前向的 Model / 可用的 Tokenizer |
| **text** 文本域 | `tokenize` / `detokenize` | 文本 ↔ token id 序列 |
| **infer** 推理工域 | `prefill` / `decode` | 推进 KV Cache → 采样 → 写回 → 自回归循环 |
| **numeric** 数值域 | `dequant` / `sample` | 量化字节 → F32 权重；logits → token |

族 README（`lines/<族>/README.md`）各含七节：链路图 / 逐跳对照 / 学习问题的答案 /
认知纠错记录 / 真机实测 / 差异注明 / 可观测点。合计 **58 条自动校验**。

### 引用约定

- **上游锚点**带行号，因为它们钉死在某个 commit 上：`llama.cpp @ 88b47a755  src/llama-vocab.cpp:1921:llama_vocab::impl::load`
- **本仓库引用**只写文件 + 符号，不写行号 —— 注释一改行号就漂，符号名才稳：
  `src/tokenizer.cpp`（`Tokenizer::encode`）
