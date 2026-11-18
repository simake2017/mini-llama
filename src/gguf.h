// ============================================================================
// ① 定位：GGUF 容器格式的只读解析器 —— 把「一个模型文件」变成「可查询的张量仓库」
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:983  gguf_init_from_file
//    llama.cpp @ 88b47a755  ggml/include/gguf.h:41  GGUF_MAGIC（:46 GGUF_DEFAULT_ALIGNMENT）
//    llama.cpp @ 88b47a755  src/llama-model-loader.h:31  struct llama_model_loader
//
// ③ 动机：
//    【概念】GGUF 的骨架固定为「版本号头 + 保序元数据 KV + 张量表 + 数据区」四段，
//    解析顺序与上游 gguf_init_from_reader 逐项对齐 —— 顺序本身就是格式契约。
//    【简化】三处简化都不动这个骨架：只读不写（写侧是模型转换工具链的事）；
//    端序统一由 read_le 处理（大端分支本仓库没覆盖过，只在 x86 上跑过）；
//    值用 std::variant 表达（上游是 type-punned 原始数组，承载的信息集相同）。
//    【对照】上游在 gguf_context 与 llama_model_loader 之间还有一层 llama_model
//    超参提取（llama-hparams.h:40），本仓库把它留给单元 3 —— 解析器只忠实暴露
//    原始 KV，分层更清楚，对照时能逐项对上。
//
// ④ demo + 多方样例：
//    🟢 合成小文件（4B 魔数 + 版本 3 + 0 张量 0 KV）→ 解析成功，tensor_count()=0
//    🟡 models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf → alignment=32，元数据几十条
//    🔴 坏魔数 → 抛「不是 GGUF 文件（魔数不匹配）」；截断文件 → 抛「文件被截断
//       （需要 N 字节，剩余 M 字节）」，M 直接告诉你差多少
//    调用方：src/main.cpp 是全仓唯一的构造点
// ============================================================================

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace minillama {

// ============================================================================
// ① 定位：GGML 张量类型编号表 —— 文件里只存一个 u32 编号，读者必须自带同一张表
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/include/ggml.h:389  enum ggml_type
//    llama.cpp @ 88b47a755  ggml/src/ggml.c:1351  ggml_type_name
//
// ③ 动机：
//    【概念】编号本身没有自描述能力 —— 读 GGUF 的人必须内置同一张编号表，
//    这是 GGUF 文件格式与 GGML 库之间的隐式契约。
//    【陷阱】编号只能追加、不能重排：文件里存的就是这个数字，重排会让所有
//    历史模型文件读出错误的类型（上游 ggml.h 在枚举上方有这句 NOTE）。
//    【简化】本实现只取 20 种（上游 40+），但编号逐个一致 —— 取子集不影响契约。
//
// ④ demo + 多方样例：
//    🟢 F32=0 / F16=1 / Q4_0=2 / Q4_1=3 / Q5_0=6 / Q8_0=8 / Q8_1=9
//    🟡 Q2_K=10 ~ Q8_K=15；非量化段 I8=24 / I16=25 / I32=26 / I64=27 / F64=28 / BF16=30
//    🔴 编号 4、5 是被删掉的旧 Q4_2/Q4_3，上游同样空着 —— 不能拿来编码新类型
//    调用方：src/gguf.cpp ggml_block_params 按编号查块参数
// ============================================================================
enum class GgmlType : uint32_t {
    F32   = 0,
    F16   = 1,
    Q4_0  = 2,
    Q4_1  = 3,
    Q5_0  = 6,
    Q5_1  = 7,
    Q8_0  = 8,
    Q8_1  = 9,
    Q2_K  = 10,
    Q3_K  = 11,
    Q4_K  = 12,
    Q5_K  = 13,
    Q6_K  = 14,
    Q8_K  = 15,
    I8    = 24,
    I16   = 25,
    I32   = 26,
    I64   = 27,
    F64   = 28,
    BF16  = 30,
};

// 块参数：量化不是逐元素压缩，而是把固定数量的元素打成一块、共享一组缩放系数，
// 计算张量字节数必须按块取整。返回 {每块元素数, 每块字节数}，非量化类型块 = 1 个元素。
// 四段式说明与对照见 src/gguf.cpp 的定义处。
std::pair<int, int> ggml_block_params(GgmlType type);

// 类型名，用于日志和报错；未知编号返回 "UNKNOWN" 而不是抛异常。
const char * ggml_type_name(GgmlType type);

// ============================================================================
// ① 定位：GGUF 元数据 KV 的值类型编号 —— 决定一个值在文件里怎么编码、占多少字节
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/include/gguf.h:53  enum gguf_type
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:426  gguf_read_emplace_helper
//
// ③ 动机：
//    【概念】这 13 个编号是「KV 层」的格式契约，与张量类型编号是两套独立编号，
//    同一个数字在两处含义不同（张量 Q4_0=2 vs 元数据 INT16=3）。
//    【陷阱】编号未知必须报错而不是按默认值猜：元数据被误读会让上层拿到
//    完全错误、但「看起来合法」的超参。
//
// ④ demo + 多方样例：
//    🟢 UINT8=0 / INT8=1 / UINT16=2 / INT16=3 / UINT32=4 / INT32=5
//    🟡 FLOAT32=6 / BOOL=7 / STRING=8 / ARRAY=9
//    🔴 UINT64=10 / INT64=11 / FLOAT64=12；编号 >= 13 上游没有 → 抛「未知元数据类型编号」
//    调用方：src/gguf.cpp read_value 按编号分发到 13 个读法
// ============================================================================
enum class GgufValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// ============================================================================
// ① 定位：一个元数据 KV 值 —— 13 种标量 + string + 可嵌套的数组，JSON 风格的任意值
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:131  struct gguf_kv
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:426  gguf_read_emplace_helper
//
// ③ 动机：
//    【概念】ARRAY 的元素本身是任意值，所以「值」这个类型是递归的。
//    【陷阱】递归类型撞 C++ 的语言硬边界：using 别名不能自引用，std::variant 的
//    替代类型又必须 complete，std::vector<自身> 只在 vector 自身里合法。标准解法
//    是把 variant 收进结构体成员 —— 类定义体内的 vector<自身> 合法，因为 vector
//    的成员函数实例化推迟到类定义完成之后。
//    【对照】上游用 union + 原始字节指针（type-punned），靠「同一块字节按类型
//    重新解释」绕开递归，没有这个语言问题；两边承载的信息集相同。
//
// ④ demo + 多方样例：
//    🟢 GgufValue v(uint32_t(3)) → v.v 持有 uint32_t
//    🟡 GgufValue(Array{...}) → v.v 持有 vector<GgufValue>，元素又各是一个 GgufValue
//    🔴 std::get<bool> 去取一个存着 uint8_t 的值 → 抛 std::bad_variant_access
//       （BOOL 的类型编号是 7，但存进 variant 时必须存 bool 成员，两侧要对上账）
//    调用方：src/gguf.cpp get_u32 等走 std::get<T>；src/tokenizer.cpp 读 BOOL
// ============================================================================
struct GgufValue {
    using Array = std::vector<GgufValue>;

    std::variant<
        uint8_t, int8_t, uint16_t, int16_t,
        uint32_t, int32_t, uint64_t, int64_t,
        float, double, bool, std::string, Array> v;

    // 各标量类型的隐式构造，让 read_value 里可以 return 原始值
    GgufValue(uint8_t  x) : v(x) {}
    GgufValue(int8_t   x) : v(x) {}
    GgufValue(uint16_t x) : v(x) {}
    GgufValue(int16_t  x) : v(x) {}
    GgufValue(uint32_t x) : v(x) {}
    GgufValue(int32_t  x) : v(x) {}
    GgufValue(uint64_t x) : v(x) {}
    GgufValue(int64_t  x) : v(x) {}
    GgufValue(float    x) : v(x) {}
    GgufValue(double   x) : v(x) {}
    GgufValue(bool     x) : v(x) {}
    GgufValue(std::string x) : v(std::move(x)) {}
    GgufValue(Array    x) : v(std::move(x)) {}
};

// 元数据里的数组类型（语义别名，读代码时更直观）
using GgufArray = GgufValue::Array;

// ============================================================================
// ① 定位：张量表里的一条 —— 名字 / 类型 / 维度 / 相对数据区起点的偏移
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:212  struct gguf_tensor_info
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:1179  gguf_get_tensor_offset
//
// ③ 动机：
//    【概念】dims 按「最快维在前」存（ggml 的 ne[] 顺序），本实现不重排 ——
//    重排会让「读代码 = 读文件」这条链断掉，前向计算也必须按同一约定解释。
//    【陷阱】offset 是相对数据区起点的未对齐偏移，不是文件绝对偏移；真正的落盘
//    位置 = 数据区起点 + align_up(offset, alignment)（见 tensor_data 的四段式）。
//    【对照】上游这条里还塞了一份 ggml_tensor；本实现把维度摊平成 vector<uint64_t>。
//
// ④ demo + 多方样例：
//    🟢 name="blk.0.attn_q.weight", type=Q8_0, dims={2048, 2048}, offset=0
//    🟡 dims={4096, 11008} → 首维 4096 是行内连续维（ne[0]）
//    🔴 offset=18 且 alignment=32 → 实际落盘在「数据区起点 + 32」，不是 +18
//    调用方：src/model.cpp 的 load_tensor_f32 拿到后接 tensor_nbytes / tensor_data
// ============================================================================
struct TensorInfo {
    std::string        name;    // 如 "blk.0.attn_q.weight"
    GgmlType           type;
    std::vector<uint64_t> dims; //  fastest-moving dim first，如 {4096, 4096}
    uint64_t           offset;  // 相对数据区起点的（未对齐）偏移
};

// ============================================================================
// ① 定位：已 mmap 的 GGUF 文件视图 —— 只读元数据 + 只读张量字节，零拷贝暴露
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:217  struct gguf_context
//    llama.cpp @ 88b47a755  src/llama-model-loader.h:31  struct llama_model_loader
//    llama.cpp @ 88b47a755  src/llama-hparams.h:40  struct llama_hparams
//
// ③ 动机：
//    【概念】模型文件动辄 GB 级，整读进内存会撑爆物理内存；mmap 让页按需换入，
//    张量数据因此可以零拷贝地暴露成指针交给反量化内核。
//    【对照】真实版在 gguf_context 和 model_loader 之间还有一层 llama_model 超参
//    提取（llama-hparams.cpp），把 architecture / context_length 这类字符串 key
//    映射到强类型字段。本版本把这一步留给单元 2/3，解析器只忠实暴露原始 KV。
//    【简化】不做「超参 → 强类型结构体」映射：承重判据是「GGUF 这个概念塌了
//    吗」—— 没塌，GGUF 仍然是「版本头 + KV + 张量表 + 数据区」，只是取超参的
//    活儿挪到了单元 3（结论见 docs/constraints/04-对齐约束.md §2.1 组 A）。
//
// ④ demo + 多方样例：
//    🟢 GgufFile g = GgufFile::load(path); g.get_u32("general.alignment")
//    🟡 单元 2 用 get_string/get_array 取 tokenizer.ggml.*；单元 3 用 find_tensor 取权重
//    🔴 构造失败即抛异常 —— 不存在「半加载的 GgufFile」：load 里 parse 抛异常时
//       会先 munmap + close 再 rethrow
//    调用方：src/main.cpp 是全仓唯一的构造点
// ============================================================================
class GgufFile {
public:
    // =========================================================================
    // ① 定位：打开并解析 GGUF 文件 —— 全仓唯一的模型入口
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:983  gguf_init_from_file
    //    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:451  gguf_init_from_reader
    //
    // ③ 动机：
    //    【陷阱】模型文件来自网络/磁盘，必须假设它是坏的：所有读取都做边界检查，
    //    失败要在「还能报出是哪一步坏」的位置抛，而不是让它变成段错误。
    //    【对照】上游用「返回 nullptr + GGML_LOG_ERROR」表达失败；本实现抛
    //    std::runtime_error —— 信息量不减（都带定位信息），调用方少写一层判空。
    //
    // ④ demo + 多方样例：
    //    🟢 合成小文件（4B 魔数 + 版本 3 + 0 张量 0 KV）→ 解析成功，tensor_count()=0
    //    🟡 models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf（1.2GB，Q8_0）→ 解析成功
    //    🔴 坏魔数 → 抛「不是 GGUF 文件（魔数不匹配）」；坏版本 → 抛「不支持的
    //       GGUF 版本: N」（本实现收 v1/v2/v3，上游已弃 v1）
    //    调用方：src/main.cpp
    // =========================================================================
    static GgufFile load(const std::string & path);

    ~GgufFile();
    GgufFile(GgufFile &&) noexcept;
    GgufFile & operator=(GgufFile &&) noexcept;
    GgufFile(const GgufFile &) = delete;

    // 已解析结果的只读视图：文件版本号、张量条数、数据区对齐粒度。
    // 都是 O(1) 取值，不做任何计算，也不抛异常。
    uint32_t version()      const { return version_; }
    uint64_t tensor_count() const { return tensors_.size(); }
    uint32_t alignment()    const { return alignment_; }

    // =========================================================================
    // ① 定位：按 key 找元数据 —— 找不到返回 nullptr，供「可缺省」的场景使用
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:1024  gguf_find_key
    //
    // ③ 动机：
    //    【简化】线性查找而不是哈希索引：模型 KV 只有几十条，线性查找是纳秒级；
    //    承重判据是「KV 查找这个概念塌了吗」—— 没塌，语义与上游完全相同，
    //    换容器即可接回去（结论见 docs/constraints/04-对齐约束.md §3.2）。
    //    【陷阱】返回 nullptr 与「值本身为空」要分清：元数据没有「空值」，
    //    值不存在只能是「找不到」，调用方必须判 nullptr 而不是解引用。
    //
    // ④ demo + 多方样例：
    //    🟢 find("general.architecture") → 命中时返回指向 GgufValue 的指针
    //    🟡 find("tokenizer.ggml.scores") → 该 key 缺失时返回 nullptr（词表可能没有分数）
    //    🔴 find("不存在的 key") → nullptr；上游对应 gguf_find_key 返回 -1
    //    调用方：src/model.cpp 的 Model::load 判 general.architecture；src/tokenizer.cpp 判可选数组
    // =========================================================================
    const GgufValue * find(const std::string & key) const;

    // =========================================================================
    // ① 定位：强类型元数据访问器 —— 找不到或类型不符一律抛异常
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:1104  gguf_get_val_u32
    //    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:1068  gguf_get_arr_n
    //
    // ③ 动机：
    //    【概念】类型检查集中在一处（get_typed），上层读超参时拿到的是干净的类型
    //    错误，而不是「把 uint32 当 float 用」这种静默错误。
    //    【陷阱】类型编号与 variant 成员必须一一对应：BOOL 的编号是 7，但存进
    //    variant 时要存 bool 成员，取的时候也得按 bool 取，两侧要对上账。
    //
    // ④ demo + 多方样例：
    //    🟢 get_u32("general.alignment") → 32
    //    🟡 get_string("tokenizer.ggml.model") → "llama"（SPM）或 "gpt2"（BPE）
    //    🔴 get_u32("tokenizer.ggml.tokens") → 抛「类型不符，期望 u32」（它是 array）；
    //       key 不存在 → 抛「缺少元数据 key: K」
    //    调用方：src/tokenizer.cpp / :71 / :86；src/model.cpp 的 Model::load 读超参
    // =========================================================================
    const std::string & get_string(const std::string & key) const;
    uint32_t            get_u32(const std::string & key) const;
    uint64_t            get_u64(const std::string & key) const;
    int64_t             get_i64(const std::string & key) const;
    float               get_f32(const std::string & key) const;
    // 数组访问：返回 variant 数组
    const GgufArray & get_array(const std::string & key) const;

    // =========================================================================
    // ① 定位：按名字精确查张量 —— 找不到返回 nullptr
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:1163  gguf_find_tensor
    //    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:1184  gguf_get_tensor_name
    //
    // ③ 动机：
    //    【简化】只做精确匹配，不做通配/前缀查找：本项目单元 3 需要的 9 张量/层
    //    都是全名，按名精确取已覆盖全部需求（承重结论见 docs/01-gguf-loader.md §16.3）。
    //    【对照】上游 llama_model_loader 用正则匹配张量名（支持「blk.N.」这类
    //    按层号拼出来的模式），并把张量数据 load 进 ggml_context；本实现只暴露
    //    指针，不复制字节。
    //
    // ④ demo + 多方样例：
    //    🟢 find_tensor("token_embd.weight") → 命中
    //    🟡 find_tensor("blk.0.attn_q.weight") → 命中，dims 给出该层形状
    //    🔴 find_tensor("不存在的名字") → nullptr；上游同场景 gguf_find_tensor 返回 -1
    //    调用方：src/model.cpp 的 load_tensor_f32 / load_tensor_buf / Model::load
    // =========================================================================
    const TensorInfo * find_tensor(const std::string & name) const;
    const std::vector<TensorInfo> & tensors() const { return tensors_; }

    // =========================================================================
    // ① 定位：张量占多少字节 —— 量化按块取整，首维不满一块也占整块
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  ggml/src/ggml.c:1297  ggml_nbytes
    //    llama.cpp @ 88b47a755  ggml/src/ggml.c:1326  ggml_blck_size
    //
    // ③ 动机：
    //    【概念】量化不是逐元素压缩：32 个（或 256 个）元素打成一块共享缩放系数，
    //    所以字节数是「块数 × 块字节数」，不能拿元素数乘元素大小。
    //    【陷阱】首维是行内连续维，块只能跨首维切 —— 末维不满一块也要占整块，
    //    于是字节数要向上取整到块的整数倍。
    //    【对照】上游 ggml_row_size 断言 ne0 % blck == 0（要求首维本身就是块整数倍）；
    //    本实现改成向上取整，语义更宽松，能吃下第三方写出的非整块张量。
    //
    // ④ demo + 多方样例：
    //    🟢 F32 dims={4096, 4096} → 4096 * 4096 * 4 = 67108864 字节
    //    🟡 Q8_0 dims={4096, 4096} → 块 32 元素 / 34 字节 → 128 * 4096 * 34 = 17825792
    //    🔴 Q8_0 dims={100, 1} → 首维 100 不是 32 的倍数 → 4 块 * 34 = 136 字节
    //       （不是 100 字节）
    //    调用方：src/model.cpp 的 load_tensor_f32 取权重字节数；src/gguf.cpp 解析末尾做越界预检
    // =========================================================================
    size_t tensor_nbytes(const TensorInfo & info) const;

    // =========================================================================
    // ① 定位：取张量数据的实际指针 —— 已按对齐修正，越界即抛
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:1016  gguf_get_data_offset
    //    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:1179  gguf_get_tensor_offset
    //
    // ③ 动机：
    //    【陷阱】GGUF 有**双重对齐**，两处都要做，且参照系不同：
    //      ① 数据区起点相对**文件**对齐（parse 里把游标位置向上取整）；
    //      ② 每个张量 offset 相对**数据区起点**再对齐一次（这里）。
    //    少任何一处，指针都会偏掉几字节 —— 读出来是「形状对、值全错」的权重。
    //    【陷阱】返回的是裸指针，文件映射必须保持存活；越过映射末尾即抛异常，
    //    因为越界读是段错误，而段错误的栈指不到「哪个张量坏了」。
    //
    // ④ demo + 多方样例：
    //    🟢 offset=0 的张量 → 指针 = data_base_（数据区起点）
    //    🟡 offset=18、alignment=32 → 指针 = data_base_ + 32（补齐到对齐边界）
    //    🔴 文件被截断、张量尾部越过映射末尾 → 抛「张量 X 数据越界」
    //    调用方：src/model.cpp 的 load_tensor_f32 / load_tensor_buf 直接把指针交给反量化内核
    // =========================================================================
    const void * tensor_data(const TensorInfo & info) const;

    // 整个数据区起点（调试用）
    const void * data_section() const { return data_base_; }

private:
    GgufFile() = default;

    // 五步解析（魔数 → 版本与计数 → 元数据 KV → 张量表 → 对齐到数据区），
    // 只在 mmap 建立之后调用；四段式说明见 src/gguf.cpp 的定义处。
    void parse();

    // --- mmap 资源管理（RAII） ---
    int    fd_    = -1;
    void * map_   = nullptr;  // 文件映射基址
    size_t size_  = 0;

    uint32_t version_   = 0;
    uint32_t alignment_ = 32; // GGUF 默认 32 字节对齐
    const uint8_t * data_base_ = nullptr;  // 数据区起点（头 + 张量表之后，已对齐）

    std::vector<std::pair<std::string, GgufValue>> metadata_;  // 保序存储
    std::vector<TensorInfo> tensors_;
};

} // namespace minillama
