// GGUF 解析器实现。格式规范见 https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//
// 文件布局（v2/v3，小端）:
//   ┌────────────────────────────────────────────┐
//   │ magic "GGUF"                          4B   │
//   │ version                             u32    │
//   │ tensor_count   (v1: u32, v2/v3: u64)        │
//   │ metadata_kv_count (同上)                    │
//   │ ── 元数据 KV（保序）──                      │
//   │   key:   u64 len + bytes                   │
//   │   type:  u32 (GgufValueType)               │
//   │   value: 按类型定长 / string / array        │
//   │ ── 张量表（tensor_count 条）──              │
//   │   name:  string, n_dims: u32               │
//   │   dims:  u64 × n_dims（最快维在前）          │
//   │   type:  u32, offset: u64（相对数据区）      │
//   │ ── 数据区（起点对齐到 alignment，默认 32）── │
//   │   张量实际位置 = 数据区起点 + align(offset)  │
//   └────────────────────────────────────────────┘

#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

namespace minillama {

// ---------------------------------------------------------------------------
// 类型表（对照 ggml.c 的 type_traits[GGML_TYPE_F32] = {..., .blck = 1, ...}）
// ---------------------------------------------------------------------------

std::pair<int, int> ggml_block_params(GgmlType type) {
    switch (type) {
        // 非量化：1 个元素 = type_size 字节
        case GgmlType::F32:  return {1, 4};
        case GgmlType::F16:  return {1, 2};
        case GgmlType::BF16: return {1, 2};
        case GgmlType::F64:  return {1, 8};
        case GgmlType::I8:   return {1, 1};
        case GgmlType::I16:  return {1, 2};
        case GgmlType::I32:  return {1, 4};
        case GgmlType::I64:  return {1, 8};
        // 传统量化：32 元素一块
        case GgmlType::Q4_0: return {32, 18}; // 32×4bit + 2B f16 scale
        case GgmlType::Q4_1: return {32, 20}; // 再加 2B f16 bias
        case GgmlType::Q5_0: return {32, 22};
        case GgmlType::Q5_1: return {32, 24};
        case GgmlType::Q8_0: return {32, 34}; // 32×int8 + 2B f16 scale
        case GgmlType::Q8_1: return {32, 40};
        // K-quants：256 元素一块，块内再分子块（单元 5 细讲）
        case GgmlType::Q2_K: return {256, 84};
        case GgmlType::Q3_K: return {256, 110};
        case GgmlType::Q4_K: return {256, 144};
        case GgmlType::Q5_K: return {256, 176};
        case GgmlType::Q6_K: return {256, 210};
        case GgmlType::Q8_K: return {256, 292};
        default:
            throw std::runtime_error(
                std::string("ggml_block_params: 未知类型编号 ") +
                std::to_string(static_cast<uint32_t>(type)));
    }
}

const char * ggml_type_name(GgmlType type) {
    switch (type) {
        case GgmlType::F32:  return "F32";
        case GgmlType::F16:  return "F16";
        case GgmlType::BF16: return "BF16";
        case GgmlType::F64:  return "F64";
        case GgmlType::I8:   return "I8";
        case GgmlType::I16:  return "I16";
        case GgmlType::I32:  return "I32";
        case GgmlType::I64:  return "I64";
        case GgmlType::Q4_0: return "Q4_0";
        case GgmlType::Q4_1: return "Q4_1";
        case GgmlType::Q5_0: return "Q5_0";
        case GgmlType::Q5_1: return "Q5_1";
        case GgmlType::Q8_0: return "Q8_0";
        case GgmlType::Q8_1: return "Q8_1";
        case GgmlType::Q2_K: return "Q2_K";
        case GgmlType::Q3_K: return "Q3_K";
        case GgmlType::Q4_K: return "Q4_K";
        case GgmlType::Q5_K: return "Q5_K";
        case GgmlType::Q6_K: return "Q6_K";
        case GgmlType::Q8_K: return "Q8_K";
        default:             return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// 小端读取游标：所有解析都经过它，天然带边界检查
// 【概念】GGUF 文件格式是小端的，与主机字节序无关。x86 上 memcpy 即可，
//        但显式写成 from_le 形式，语义和可移植性都更清楚。
// ---------------------------------------------------------------------------

// 【陷阱】reinterpret_cast<uint32_t*>(p) 可能违反严格别名规则且未对齐访问会崩，
//         memcpy 是现代编译器会优化成单条 mov 的安全写法。
template <typename T>
static T read_le(const uint8_t * p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    // 大端主机需要逐字节交换（【简化】本仓库主要在 x86 上跑，此分支仅为演示）
    if constexpr (sizeof(T) == 2) { v = __builtin_bswap16(v); }
    if constexpr (sizeof(T) == 4) { v = __builtin_bswap32(v); }
    if constexpr (sizeof(T) == 8) { v = __builtin_bswap64(v); }
#endif
    return v;
}

class Cursor {
public:
    Cursor(const uint8_t * base, size_t size, std::string what)
        : base_(base), cur_(base), end_(base + size), what_(std::move(what)) {}

    size_t pos() const { return static_cast<size_t>(cur_ - base_); }
    const uint8_t * base() const { return base_; }

    void require(size_t n) const {
        if (static_cast<size_t>(end_ - cur_) < n) {
            throw std::runtime_error(
                what_ + ": 文件被截断（需要 " + std::to_string(n) +
                " 字节，剩余 " + std::to_string(static_cast<size_t>(end_ - cur_)) + "）");
        }
    }

    template <typename T> T read() {
        require(sizeof(T));
        T v = read_le<T>(cur_);
        cur_ += sizeof(T);
        return v;
    }

    // GGUF string: u64 长度前缀 + 原始字节（无 NUL 结尾）
    std::string read_string() {
        uint64_t len = read<uint64_t>();
        require(len);
        std::string s(reinterpret_cast<const char *>(cur_), len);
        cur_ += len;
        return s;
    }

private:
    const uint8_t * base_;
    const uint8_t * cur_;
    const uint8_t * end_;
    std::string what_;
};

// ---------------------------------------------------------------------------
// 元数据值解析
// ---------------------------------------------------------------------------

static GgufValue read_value(Cursor & c, GgufValueType type) {
    switch (type) {
        case GgufValueType::UINT8:   return c.read<uint8_t>();
        case GgufValueType::INT8:    return c.read<int8_t>();
        case GgufValueType::UINT16:  return c.read<uint16_t>();
        case GgufValueType::INT16:   return c.read<int16_t>();
        case GgufValueType::UINT32:  return c.read<uint32_t>();
        case GgufValueType::INT32:   return c.read<int32_t>();
        case GgufValueType::FLOAT32: return c.read<float>();
        case GgufValueType::BOOL:    return c.read<uint8_t>() != 0;
        case GgufValueType::UINT64:  return c.read<uint64_t>();
        case GgufValueType::INT64:   return c.read<int64_t>();
        case GgufValueType::FLOAT64: return c.read<double>();
        case GgufValueType::STRING:  return c.read_string();
        case GgufValueType::ARRAY: {
            // 【概念】数组 = 元素类型 + 元素个数 + 连续 N 个该类型值。
            // 元素类型不能是 ARRAY（ggml 的写出器禁止嵌套数组）。
            GgufValueType elem = static_cast<GgufValueType>(c.read<uint32_t>());
            if (elem == GgufValueType::ARRAY) {
                throw std::runtime_error("read_value: 不支持嵌套数组");
            }
            uint64_t n = c.read<uint64_t>();
            // 【陷阱】count 来自不可信文件，先防溢出再循环读
            if (n > (1ull << 31)) {
                throw std::runtime_error("read_value: 数组元素数异常: " + std::to_string(n));
            }
            GgufArray v;
            v.reserve(static_cast<size_t>(n));
            for (uint64_t i = 0; i < n; i++) {
                v.push_back(read_value(c, elem));
            }
            return v;
        }
        default:
            throw std::runtime_error(
                "read_value: 未知元数据类型编号 " + std::to_string(static_cast<uint32_t>(type)));
    }
}

// ---------------------------------------------------------------------------
// GgufFile：mmap + 解析
// ---------------------------------------------------------------------------

// 【概念】mmap 把文件直接映射进进程地址空间：
//   - 读取页缺失时才由内核从磁盘换入，大模型文件不会撑爆物理内存
//   - 张量数据以零拷贝方式暴露为指针，后续计算直接读
// 真实版对照: src/llama-mmap.cpp 的 llama_mmap（功能相同，多了 NUMA/Windows 适配）
GgufFile GgufFile::load(const std::string & path) {
    GgufFile f;

    f.fd_ = open(path.c_str(), O_RDONLY);
    if (f.fd_ < 0) {
        throw std::runtime_error("open 失败: " + path + " (" + std::strerror(errno) + ")");
    }
    struct stat st {};
    if (fstat(f.fd_, &st) != 0 || st.st_size <= 0) {
        close(f.fd_);
        throw std::runtime_error("fstat 失败或文件为空: " + path);
    }
    f.size_ = static_cast<size_t>(st.st_size);

    f.map_ = mmap(nullptr, f.size_, PROT_READ, MAP_SHARED, f.fd_, 0);
    if (f.map_ == MAP_FAILED) {
        close(f.fd_);
        throw std::runtime_error("mmap 失败: " + path);
    }
    // 【性能】madvise 告诉内核"顺序读"：真实版 llama-mmap.cpp 也这样做，
    //         对大模型文件能显著减少缺页中断带来的抖动。
    madvise(f.map_, f.size_, MADV_WILLNEED);

    try {
        f.parse();
    } catch (...) {
        munmap(f.map_, f.size_);
        close(f.fd_);
        throw;
    }
    return f;
}

GgufFile::~GgufFile() {
    if (map_)  { munmap(map_, size_); }
    if (fd_ >= 0) { close(fd_); }
}

GgufFile::GgufFile(GgufFile && o) noexcept { *this = std::move(o); }

GgufFile & GgufFile::operator=(GgufFile && o) noexcept {
    if (this != &o) {
        if (map_)  { munmap(map_, size_); }
        if (fd_ >= 0) { close(fd_); }
        fd_ = o.fd_;       o.fd_ = -1;
        map_ = o.map_;     o.map_ = nullptr;
        size_ = o.size_;   o.size_ = 0;
        version_ = o.version_;
        alignment_ = o.alignment_;
        data_base_ = o.data_base_;
        metadata_ = std::move(o.metadata_);
        tensors_ = std::move(o.tensors_);
    }
    return *this;
}

static size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

void GgufFile::parse() {
    Cursor c(static_cast<const uint8_t *>(map_), size_, "gguf");

    // ① 魔数：GGUF 就是 "GGUF" 四字节。
    //   【历史】前身 GGML 格式用魔数 'ggml'，v3 起改名 GGUF（Unified Format）。
    uint32_t magic = c.read<uint32_t>();
    if (magic != 0x46554747u) { // "GGUF" 的 little-endian u32 表示
        throw std::runtime_error("不是 GGUF 文件（魔数不匹配）");
    }

    version_ = c.read<uint32_t>();
    if (version_ == 0 || version_ > 3) {
        throw std::runtime_error("不支持的 GGUF 版本: " + std::to_string(version_));
    }

    // v1 用 u32 存计数，v2/v3 用 u64 —— 这是三个版本间唯一的布局差异
    uint64_t n_tensors, n_kv;
    if (version_ == 1) {
        n_tensors = c.read<uint32_t>();
        n_kv      = c.read<uint32_t>();
    } else {
        n_tensors = c.read<uint64_t>();
        n_kv      = c.read<uint64_t>();
    }

    // ② 元数据 KV（保序存入，后续 get 时按 key 线性查找 —
    //    【简化】真实版 gguf.cpp 用哈希表索引 key，模型 KV 只有几十条，线性查找足够）
    metadata_.reserve(static_cast<size_t>(n_kv));
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = c.read_string();
        auto type = static_cast<GgufValueType>(c.read<uint32_t>());
        metadata_.emplace_back(std::move(key), read_value(c, type));
    }

    // general.alignment 可覆盖默认 32 字节对齐（llama.cpp 转换器一般不写，用默认）
    if (const GgufValue * v = find("general.alignment")) {
        alignment_ = std::get<uint32_t>(v->v);
    }

    // ③ 张量表
    tensors_.reserve(static_cast<size_t>(n_tensors));
    for (uint64_t i = 0; i < n_tensors; i++) {
        TensorInfo t;
        t.name = c.read_string();
        uint32_t n_dims = c.read<uint32_t>();
        if (n_dims == 0 || n_dims > 4) {
            throw std::runtime_error(
                "张量 " + t.name + ": n_dims 非法: " + std::to_string(n_dims));
        }
        t.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; d++) {
            t.dims[d] = c.read<uint64_t>();
        }
        t.type   = static_cast<GgmlType>(c.read<uint32_t>());
        t.offset = c.read<uint64_t>();
        tensors_.push_back(std::move(t));
    }

    // ④ 数据区起点：张量表结束后填充到 alignment
    // 【对照】gguf.cpp: gr.seek(GGML_PAD(gr.tell(), ctx->alignment))，张量 offset 相对此起点
    data_base_ = c.base() + align_up(c.pos(), alignment_);

    // 数据区至少要覆盖最后一个张量（便宜检查，提前报错好过段错误）
    for (const auto & t : tensors_) {
        size_t nb = tensor_nbytes(t);
        size_t off = align_up(t.offset, alignment_);
        if (off + nb > size_ - static_cast<size_t>(data_base_ - c.base())) {
            throw std::runtime_error("张量 " + t.name + " 越出数据区（文件损坏或截断）");
        }
    }
}

// ---------------------------------------------------------------------------
// 访问器
// ---------------------------------------------------------------------------

const GgufValue * GgufFile::find(const std::string & key) const {
    for (const auto & [k, v] : metadata_) {
        if (k == key) { return &v; }
    }
    return nullptr;
}

const std::string & GgufFile::get_string(const std::string & key) const {
    const GgufValue * v = find(key);
    if (!v) { throw std::runtime_error("GGUF 缺少元数据 key: " + key); }
    if (!std::holds_alternative<std::string>(v->v)) {
        throw std::runtime_error("GGUF 元数据 " + key + " 不是 string");
    }
    return std::get<std::string>(v->v);
}

// 【概念】强类型访问器模板：把 variant 的类型检查集中在一处，
//         上层（单元 2/3 读超参时）拿到的是干净的类型错误而不是段错误。
template <typename T>
static T get_typed(const GgufFile & f, const std::string & key, const char * what) {
    const GgufValue * v = f.find(key);
    if (!v) { throw std::runtime_error("GGUF 缺少元数据 key: " + key); }
    if (!std::holds_alternative<T>(v->v)) {
        throw std::runtime_error("GGUF 元数据 " + key + " 类型不符，期望 " + what);
    }
    return std::get<T>(v->v);
}

uint32_t GgufFile::get_u32(const std::string & key) const { return get_typed<uint32_t>(*this, key, "u32"); }
uint64_t GgufFile::get_u64(const std::string & key) const { return get_typed<uint64_t>(*this, key, "u64"); }
int64_t  GgufFile::get_i64(const std::string & key) const { return get_typed<int64_t>(*this, key, "i64"); }
float    GgufFile::get_f32(const std::string & key) const { return get_typed<float>(*this, key, "f32"); }

const GgufArray & GgufFile::get_array(const std::string & key) const {
    const GgufValue * v = find(key);
    if (!v) { throw std::runtime_error("GGUF 缺少元数据 key: " + key); }
    if (!std::holds_alternative<GgufArray>(v->v)) {
        throw std::runtime_error("GGUF 元数据 " + key + " 不是 array");
    }
    return std::get<GgufArray>(v->v);
}

const TensorInfo * GgufFile::find_tensor(const std::string & name) const {
    for (const auto & t : tensors_) {
        if (t.name == name) { return &t; }
    }
    return nullptr;
}

size_t GgufFile::tensor_nbytes(const TensorInfo & info) const {
    auto [blck, blck_size] = ggml_block_params(info.type);
    if (info.dims.empty()) { return 0; }
    // 行内按块取整（首维最快移动），其余维直接相乘
    uint64_t row_blocks = (info.dims[0] + blck - 1) / blck;
    uint64_t n = row_blocks;
    for (size_t i = 1; i < info.dims.size(); i++) { n *= info.dims[i]; }
    return static_cast<size_t>(n) * blck_size;
}

const void * GgufFile::tensor_data(const TensorInfo & info) const {
    size_t off = align_up(info.offset, alignment_);
    if (data_base_ + off + tensor_nbytes(info) >
        static_cast<const uint8_t *>(map_) + size_) {
        throw std::runtime_error("张量 " + info.name + " 数据越界");
    }
    return data_base_ + off;
}

} // namespace minillama
