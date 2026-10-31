# mini-llama：llama.cpp 核心能力最小还原

用 C++17（仅标准库 + POSIX mmap）从零还原 llama.cpp 的推理主干：

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

训练、多架构、多后端（CUDA/Metal 等）、GGML 计算图、投机解码、多模态、LoRA、语法采样。理由见 docs/ 各单元对照文档。

## 文档

每单元完成后产出三段式对照文档（我做了啥 / 真实源码在哪 / 真实版多做啥），见 `docs/`。
