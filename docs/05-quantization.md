# 单元 5：量化推理 —— 反量化加载 Q8_0 / Q4_K / Q5_K / Q6_K 真实模型

（对应真实 llama.cpp 的 ggml/src/ggml-quants.c + src/llama-quant.cpp）

## ① 我实现了什么

- **类型系统扩展**：`GgmlType` 枚举在 F32/F16 之外增加 Q8_0=8、Q4_K=12、Q5_K=13、Q6_K=14，每种类型配块参数表（每块元素数 / 每块字节数），与 ggml 完全一致
- **TensorBuf**：张量不再强制转 FP32，量化字节原样保存；`matvec` 按类型分发到对应反量化例程
- **反量化内核**（逐行移植 ggml-quants.c）：
  - Q8_0：32 元素一块，{f16 d; int8 qs[32]}，value = d * qs
  - Q4_K/Q5_K：256 元素一块，value = d*sc*nibble - dmin*m；16 个 6bit scale/min 压缩在 12 字节里（get_scale_min_k4 位打包）
  - Q6_K：256 元素一块，有符号 6bit 量化 + 16 个 int8 scale，value = d * sc * (q6 - 32)
- **加载校验**：量化张量要求 ne0 % 块元素数 == 0，越界数据区在加载期报错（不等前向才炸）
- **混合量化前向**：同一组逻辑权重分别存 F32 和 Q8_0，前向 logits max|Δ| = 0.04、argmax 全部一致——验证"反量化接入前向"这条链路正确
- **端到端**：`mini-llama --model TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf --prompt ...` 直接生成文本
- **与 libllama 逐数值对照**（自定义工具直接调 `llama_decode` API，绕过崩溃的 llama-cli）：单 token 到 8 token 序列的 logits max|Δ| ≤ 0.27、mean|Δ| ≤ 0.12（FP32 累加顺序噪声量级），top1 在非近似并列时一致；独立 numpy 仲裁者（从 GGUF 原始字节手写解析+前向，零复用本仓库代码）与本实现逐位一致

关键设计决策：

- **只反量化、不量化**：llama.cpp 的真实用法也是"转换工具（llama-quantize）一次性量化 → 推理只反量化"。我们只还原推理侧，量化器砍掉。
- **反量化粒度是"一整列"**：matvec 每次取 W 的一列（一个输出神经元）反量化到临时缓冲，再与激活做点积。代码简单、缓存友好度可接受（列内块连续）。
- **不引入 ggml 的 block 结构体**：用原始 uint8_t 指针 + 偏移注释表达块布局，保持"能不用结构体就不用"的教学风格。

## ② 真实 llama.cpp 对应源码

| mini-llama | 真实版 | 说明 |
|---|---|---|
| `src/quant.h/.cpp` dequant_row_* | `ggml/src/ggml-quants.c` 同名函数 | 逐行移植，块偏移逐一对照 |
| get_scale_min_k4 | ggml-quants.c 同名 inline | K-quants 的 scale/min 位解压 |
| ggml_block_params | `ggml_type_traits`（ggml.c） | 类型 → 块大小映射 |
| TensorBuf + matvec 分发 | `ggml_vec_dot` 系列 + `ggml_get_type_traits` | 真实版是函数指针表，我们是 switch |
| 加载期校验 | `llama_model_loader::load_all_data` + `ggml_nbytes` | 等价 |

## ③ 真实版比我多做了什么、为什么

| 真实版多做 | 为什么可以砍 |
|---|---|
| 约 20 种量化格式（Q2_K~Q8_K、IQ 系列、FP8/FP16 混合） | 4 种已覆盖主流推理模型的全部结构（Q4_K_M 等混合量化 = 张量级类型选择，加载器天然支持） |
| SIMD 向量化反量化 + 专用 dot 内核（AVX2/NEON） | 纯标量版正确性等价，速度差 10-50 倍但教学版不关心吞吐 |
| 原位量化（quantize 侧） | 推理路径用不到，llama-quantize 工具的职责 |
| super-block 的 scale 再量化（Q3_K 的 hbits） | 格式数量裁剪的连带项 |
| 张量级混合量化元数据（tensor_type 每张量独立） | 我们天然支持——每个 TensorInfo 各自带类型 |

## ④ 踩坑记录

1. **Q4_K/Q5_K 的 qs 偏移**：块布局是 d(2B) dmin(2B) scales(12B) qs(128B)，qs 在 p+16 不是 p+4——照抄公式时把 scales 长度算进去。
2. **K-quants 的 scale/min 是 6bit 打包**：16 个值 × 6bit = 96bit = 12 字节，j<4 与 j>=4 走不同解包路径，手构测试块时必须两路都测。
3. **Q6_K 的 ql/qh 是"两行共享"**：改 ql[0] 的 lo nibble 影响元素 0、hi nibble 影响元素 64——测试预期要算全。
4. **nibble 重叠**：Q4_K 的 32 个字节 lo/hi 分别对应两个 32 元素组，测试不能只验证前 32 个。
5. **加载期校验 vs 前向期爆炸**：量化张量 ne0 不是块大小整数倍属于"文件损坏"，加载时报错比前向算到一半 NaN 好调试得多。
6. **混合量化模型的张量表偏移必须按实际类型字节数算**：测试构造器最初写死 `data.size()*4`，量化张量一进来整个张量表偏移全错。
7. **GGUF 数据区对齐的双重约定**：数据区起点本身绝对对齐到 alignment（默认 32），张量 offset 相对该起点累计、张量之间再各自对齐（gguf.cpp: `gr.seek(GGML_PAD(gr.tell(), alignment))`）。测试构造器只做了前者时，第二个张量起全部错位。
8. **Cursor 构造函数漏初始化 base_ 成员**：`pos()` 返回 `cur_ - base_`，base_ 未初始化时读的是栈垃圾——新进程靠栈残留"碰巧正确"，跑过其它代码后必崩。症状链：真实模型测试随机越界/NaN → 误判为对齐语义问题两次改错方向 → ASan 全过 vs Release 挂（未初始化读特征；ASan 的 0xBE 填充恰好是有限浮点数）→ 加打印发现 pos() 返回天文数字 → 定位。教训：①成员初始化列表逐个 eyeball，②"ASan 过 Release 挂"优先怀疑未初始化读，③能跑≠正确，栈残留假象必须靠进程内多次不同路径复现戳穿。
