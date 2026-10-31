# 单元 1：GGUF 模型加载 —— 把"一个文件"变成"可查询的张量仓库"

（对应真实 llama.cpp 的 GGUF 格式解析层）

## ① 我实现了什么

- **文件头解析**：魔数 `GGUF`、版本号（支持 v1/v2/v3）、张量数、元数据 KV 数
- **元数据 KV 解析**：13 种标量类型 + string + array（含嵌套值），保序存储，提供强类型访问器（`get_u32` / `get_string` / `get_array` ...），类型不符直接抛异常
- **张量表解析**：名字 / 维度（最快维在前，与 ggml `ne[]` 同序）/ 类型编号 / 相对偏移
- **零拷贝数据访问**：mmap 整个文件，张量数据以原始指针暴露；按 `general.alignment`（默认 32）对齐修正每个张量的实际位置
- **量化类型字节数计算**：内置 GGML 类型表的块参数（如 Q4_K = 256 元素/144 字节），按块取整计算张量字节数
- **不可信输入防御**：所有读取经边界检查游标；坏魔数、坏版本、截断、越界都抛异常而不是段错误

关键设计决策：

- **解析器不做超参解释**。`llama.context_length` 这类 key 到结构体字段的映射留给单元 2/3，解析器只忠实暴露原始 KV —— 分层清晰，对照真实版时一眼能看出 `gguf_context` 和 `model_loader` 的分工。
- **`GgufValue` 用 struct 包 variant** 表达 JSON 式递归值。踩过的坑：`using` 别名不能自引用，`vector<不完全类型>` 在 libstdc++ 里不能 `push_back`，最终用"类内 `vector<自身>` 成员"这一标准手法解决。

## ② 真实 llama.cpp 对应源码

| mini-llama | 真实版 | 说明 |
|---|---|---|
| `src/gguf.h` GgmlType 枚举 | `ggml/include/ggml.h` `enum ggml_type` | 编号契约，真实版 40+ 种，我们只取 20 种 |
| `src/gguf.cpp` ggml_block_params | `ggml.c` `type_traits[]` 表 | 块参数来源一致 |
| `src/gguf.cpp` Cursor / read_le | `gguf.cpp` `gguf_bufread` | 真实版叫 bufreader，同样做边界检查 |
| `src/gguf.cpp` read_value | `gguf.cpp` `gguf_read_value` | 类型分发逻辑相同 |
| `src/gguf.cpp` GgufFile::parse | `gguf.cpp` `gguf_init_from_file` | 五步顺序一致：魔数→头→KV→张量表→数据区 |
| `src/gguf.cpp` load (mmap) | `src/llama-mmap.cpp` `llama_mmap` | 同样 MADV_WILLNEED |
| `GgufFile::find/get_*` | `gguf.cpp` `gguf_find_key/get_val_*` | 真实版有哈希索引 |
| `tensor_nbytes/tensor_data` | `gguf.cpp` `gguf_get_tensor_nbytes/offset` | 对齐修正逻辑一致 |

## ③ 真实版比我多做了什么、为什么

| 真实版多做 | 为什么可以砍 |
|---|---|
| **写 GGUF**（`gguf.cpp` 后半） | 模型转换工具链的事，推理只读 |
| **哈希表索引元数据 key** | 模型 KV 只有几十条，线性查找纳秒级 |
| **大端主机字节序交换** | 目标平台 x86/ARM 都是小端 |
| **重叠张量去重**（`gguf.cpp` 的 `find_tensor_by_offset`） | 小模型没有共享权重的张量，且不影响格式理解 |
| **超参 → 强类型结构体映射**（`llama-hparams.cpp`） | 留给单元 3，作为独立分层点 |
| **`llama_model_loader` 的张量名通配/正则查找、lazy 加载、用量统计** | 单元 3 按名字精确取张量已够用 |
| **Windows/NUMA/ huge pages 适配** | 单机 Linux 教学项目 |

## ④ 踩坑记录

- **魔数写测试时被当成 string**：GGUF 头的 magic 是 4 个原始字节 `"GGUF"`，不是 GGUF string（u64 长度前缀 + 内容）。测试 Writer 的 `str()` 会多写 8 字节，导致解析错位。教训：构造合成文件时，每个字段都要对照格式文档确认是"原始字节"还是"长度前缀串"。
- **递归 variant 三种写法都不通**：`using` 自引用别名（非法）→ `vector<不完全类型>` + 继承（libstdc++ 拒绝实例化）→ 最终 `struct { variant<..., vector<自身>> }` 才合法。这是 C++ 表达树形/JSON 数据的经典考点。
- **元数据 KV 计数错位**：测试里声明 8 条实际写了 9 条，解析器把多余字节当张量表读，报出"n_dims 非法"。好在错误信息带了上下文（张量名），一眼定位。这印证了"解析器报错要带 key/张量名"的设计。

## ⑤ 验收记录

- 合成用例 6 组全绿：基本解析（标量/string/array/张量表/数据抽查）、量化张量字节数、坏魔数、坏版本、截断、类型不符
- 真实文件冒烟 3 个全过：`ggml-vocab-llama-bpe.gguf`（纯词表）、`qmd-query-expansion-1.7B-q4_k_m.gguf`（311 个张量，Q6_K/Q4_K 混合）、`embeddinggemma-300M-Q8_0.gguf`（316 个张量）
- 运行方式：`cmake --build build -j6 && TEST_GGUF_REAL_FILE=<路径> ./build/test_gguf`
