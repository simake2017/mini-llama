// GGUF 模型文件格式解析器（单元 1）。
//
// 对照真实 llama.cpp:
//   ggml/src/gguf.cpp            —— GGUF 格式的读写实现
//   src/llama-model-loader.cpp   —— 在 GGUF 之上提取模型超参 / 张量
//
// 本最小版简化点:
//   - 只读不写（真实版还能写 GGUF，用于模型转换）
//   - 只支持 little-endian 主机（真实版处理字节序交换）
//   - 元数据值用 std::variant 表示（真实版用 type-punned 原始数组）
//
// 为什么可以简化:
//   解析器的主干就是"魔数 -> 头 -> 元数据 KV -> 张量表 -> 数据区"五步，
//   字节序和写路径不改变这个结构。

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace minillama {

// ---------------------------------------------------------------------------
// 【对照】ggml/include/ggml.h enum ggml_type
//
// GGUF 文件里每个张量只存一个 u32 类型编号，编号本身没有自描述能力，
// 所以读者必须内置同一张编号表 —— 这是 GGUF 格式和 GGML 库之间的隐式契约，
// 编号永远只能追加、不能重排（ggml.h 里有注释强调这一点）。
// ---------------------------------------------------------------------------
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

// 每种量化类型的"块"参数：量化不是逐元素压缩，而是把固定数量的元素
// 打成一块、共享一组缩放系数。计算张量字节数必须按块取整。
// 返回 {每块元素数, 每块字节数}；非量化类型块 = 1 个元素。
std::pair<int, int> ggml_block_params(GgmlType type);

// 类型名，用于日志和报错
const char * ggml_type_name(GgmlType type);

// ---------------------------------------------------------------------------
// GGUF 元数据 KV 的值类型（gguf.cpp 里叫 gguf_value_type，编号即文件格式）
// ---------------------------------------------------------------------------
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

// 【概念】JSON 风格的递归任意值：ARRAY 元素本身是任意值。
// 【陷阱】using 别名不能自引用，std::variant 的替代类型又必须 complete。
//         标准手法是把 variant 收进结构体成员 —— 类定义体内的 vector<自身>
//         是合法的（vector 的成员函数实例化推迟到类定义完成之后）。
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

// 张量表条目。文件里的 dims 按"最快维在前"存储（即 ggml 的 ne[] 顺序），
// 我们不重排，原样保留，前向计算时按同一约定解释。
struct TensorInfo {
    std::string        name;    // 如 "blk.0.attn_q.weight"
    GgmlType           type;
    std::vector<uint64_t> dims; //  fastest-moving dim first，如 {4096, 4096}
    uint64_t           offset;  // 相对数据区起点的（未对齐）偏移
};

// 解析后的 GGUF 文件。内存映射整个文件，张量数据零拷贝直接暴露指针。
//
// 真实 llama.cpp 对应:
//   gguf.cpp 的 struct gguf_context + llama-model-loader.cpp 的 llama_model_loader
//
// 关键区别:
//   真实版在 gguf_context 和 model_loader 之间还有一层 llama_model 超参提取
//   （llama-hparams.cpp），把 architecture/llama.context_length 这类
//   字符串 key 映射到强类型字段。本版本把这一步留给单元 2/3，
//   解析器只忠实暴露原始 KV —— 分层更清晰，也更容易对照。
class GgufFile {
public:
    // 打开并解析 GGUF 文件。失败抛 std::runtime_error（坏魔数、截断、版本过高等）。
    // 【陷阱】模型文件来自网络/磁盘，必须假设它是坏的：所有读取都做边界检查。
    static GgufFile load(const std::string & path);

    ~GgufFile();
    GgufFile(GgufFile &&) noexcept;
    GgufFile & operator=(GgufFile &&) noexcept;
    GgufFile(const GgufFile &) = delete;

    uint32_t version()      const { return version_; }
    uint64_t tensor_count() const { return tensors_.size(); }
    uint32_t alignment()    const { return alignment_; }

    // 元数据访问：找不到返回 nullptr（真实版 gguf_find_key 返回 -1）
    const GgufValue * find(const std::string & key) const;
    // 便捷强类型访问：找不到或类型不符抛异常
    const std::string & get_string(const std::string & key) const;
    uint32_t            get_u32(const std::string & key) const;
    uint64_t            get_u64(const std::string & key) const;
    int64_t             get_i64(const std::string & key) const;
    float               get_f32(const std::string & key) const;
    // 数组访问：返回 variant 数组
    const GgufArray & get_array(const std::string & key) const;

    // 张量查找：按名字精确匹配（真实版还有通配符/前缀查找，单元 3 需要时再补）
    const TensorInfo * find_tensor(const std::string & name) const;
    const std::vector<TensorInfo> & tensors() const { return tensors_; }

    // 张量数据的字节数（按块取整）
    size_t tensor_nbytes(const TensorInfo & info) const;

    // 张量数据指针（已按对齐修正）。文件必须保持 mmap 存活，禁止越过数据区末尾。
    const void * tensor_data(const TensorInfo & info) const;

    // 整个数据区起点（调试用）
    const void * data_section() const { return data_base_; }

private:
    GgufFile() = default;

    void parse();  // 在 mmap 建立后解析头、元数据、张量表

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
