// ============================================================================
// ① 定位：GGUF 解析器实现 —— 把映射好的文件字节流走成「元数据 + 张量表 + 数据区」
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:451  gguf_init_from_reader
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:426  gguf_read_emplace_helper
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:230  struct gguf_reader
//
// ③ 动机：
//    【概念】解析主干是五步：魔数 → 版本与计数 → 元数据 KV → 张量表 → 对齐到
//    数据区。五步顺序与上游 gguf_init_from_reader 逐项一致 —— 顺序就是格式契约。
//    【陷阱】文件里所有数字都不可信：长度、计数、维度、数组元素数全要先过边界
//    检查与上限检查，再进循环。
//    【对照】上游读完张量表后有一条交叉校验 GGML_ASSERT(ctx->info.size() ==
//    n_tensors)（ggml/src/gguf.cpp:753）；本实现没有这条断言，是已登记的加固
//    缺口（见 docs/01-gguf-loader.md §17.3）。
//
// ④ demo + 多方样例：
//    🟢 合成文件 = 4B 魔数 + 版本 3 + 0 张量 0 KV → 解析成功，无 KV 无张量
//    🟡 真实 Q8_0 模型 → 张量表几百条，每条按「名字 + n_dims + dims + 类型 + 偏移」读出
//    🔴 文件尾部被截断 → Cursor::require 抛「文件被截断（需要 N 字节，剩余 M 字节）」
//    调用方：src/gguf.h 的 GgufFile 各接口；上游对照见 ggml/src/gguf.cpp 同一批函数
//
// 格式规范：https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
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
// ============================================================================

#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

namespace minillama {

// ============================================================================
// ① 定位：张量类型的块参数表 —— 「一块多少元素 / 一块多少字节」
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/ggml.c:631  struct ggml_type_traits type_traits[]
//    llama.cpp @ 88b47a755  ggml/src/ggml.c:1326  ggml_blck_size
//
// ③ 动机：
//    【概念】量化不是逐元素压缩，而是把固定数量的元素打成一块、共享一组缩放
//    系数；张量字节数因此是「块数 × 块字节数」，不是「元素数 × 元素大小」。
//    【陷阱】这张表是**编号契约的一部分**：改一个数字，全仓读出来的张量长度
//    全错，而且不会报错 —— 只会在反量化时读到错位的字节。
//    【对照】上游把这张表放在 ggml_type_traits 里，blck_size / type_size 一一
//    对应；本实现只取两者，is_quantized 与转换函数指针用不到。
//
// ④ demo + 多方样例：
//    🟢 F32 → {1, 4}：1 个元素 4 字节（无量化，块 = 元素）
//    🟡 Q8_0 → {32, 34}：32 个 int8 + 2 字节 f16 scale；Q4_K → {256, 144}
//    🔴 编号不在表内（如 4、5）→ 抛「ggml_block_params: 未知类型编号 N」
//       —— 不猜默认值，猜错的后果是静默的长度错误
//    调用方：src/gguf.cpp GgufFile::tensor_nbytes；src/quant.cpp 的反量化内核
// ============================================================================
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

// ============================================================================
// ① 定位：类型编号 → 类型名字符串，用于日志与报错信息
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:1003  gguf_type_name
//    llama.cpp @ 88b47a755  ggml/src/ggml.c:1351  ggml_type_name
//
// ③ 动机：
//    【陷阱】未知编号返回 "UNKNOWN" 而不是抛异常：它是**格式化错误信息**用的，
//    抛异常会把「报告错误」变成「制造新错误」。真正的合法性校验在
//    ggml_block_params 与解析期，那里对未知编号是抛的。
//    【对照】上游 gguf_type_name 对未知编号返回 nullptr（调用方得自己判空）；
//    本实现返回 "UNKNOWN" 少一次判空，代价是丢掉了「未知」与「名字恰好叫
//    UNKNOWN 的类型」的区别 —— GGUF 里没有后者。
//
// ④ demo + 多方样例：
//    🟢 ggml_type_name(GgmlType::Q4_0) → "Q4_0"
//    🟡 ggml_type_name(GgmlType::BF16) → "BF16"
//    🔴 ggml_type_name(static_cast<GgmlType>(4)) → "UNKNOWN"（4 是被删掉的旧编号）
//    调用方：src/model.cpp 的 matvec / load_tensor_buf 拼错误信息；
//       lines/load/model_load_line.cpp 统计张量类型分布
// ============================================================================
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
// 【对照】上游 gguf_reader 内部同样按小端读，不看主机字节序。
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

// 游标把「当前位置 + 剩余长度 + 出错时的前缀说明」三件事捆在一起：
// 所有读取都必须先过 require，于是越界只有一个出口，报错信息也不会各写各的。
class Cursor {
public:
    Cursor(const uint8_t * base, size_t size, std::string what)
        : base_(base), cur_(base), end_(base + size), what_(std::move(what)) {}

    size_t pos() const { return static_cast<size_t>(cur_ - base_); }
    const uint8_t * base() const { return base_; }

    // =========================================================================
    // ① 定位：读取前的边界检查 —— 整个解析器唯一的越界防线
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:230  struct gguf_reader
    //    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:331  gguf_reader::read
    //
    // ③ 动机：
    //    【陷阱】语义是「**剩余**字节够不够」，不是「总长够不够」：越界必须在
    //    **读之前**判，否则读操作本身就已经踩到映射外。
    //    【概念】报错必须带「需要 N 字节、剩余 M 字节」：M 直接告诉人差多少，
    //    省掉一次二分查找损坏点。段错误做不到这一点 —— 它的栈指不到「哪个张量
    //    坏了」。
    //
    // ④ demo + 多方样例：
    //    🟢 读 u32 时剩余 >= 4 → 正常返回
    //    🟡 读 GGUF string 的长度前缀后只剩 2 字节、内容长度 5 → 抛「需要 5 字节，剩余 2 字节」
    //    🔴 文件被截断 → 抛「文件被截断（需要 N 字节，剩余 M 字节）」，M 指出差多少
    //    调用方：Cursor::read / Cursor::read_string；上游对应 gguf_reader 的 read 重载
    // =========================================================================
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
    // 【概念】长度是 u64 而不是 u32：文件名和词表文本的长度上限不该被 32 位卡住；
    //         「无 NUL 结尾」是格式约定，容器层的裸字节魔数也是同理（见 docs/01 §17.1）。
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

// ============================================================================
// ① 定位：按类型编号读出一个元数据值 —— 13 种类型的唯一分发点
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:426  gguf_read_emplace_helper
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:585  12 路类型分派（ARRAY 落到 default 被拒）
//
// ③ 动机：
//    【陷阱】ARRAY 的元素类型**不能是 ARRAY**：GGUF 的数组编码只有「元素类型 +
//    元素个数 + 连续 N 个值」三段，没有嵌套数组的表示法（上游写出侧同样没有
//    这条路径）。所以这里显式拒绝而不是递归 —— 递归会读出一个格式里根本不
//    存在的结构。
//    【陷阱】数组元素数来自不可信文件：先卡上限（1<<31）再 reserve，否则一个
//    2^40 的计数会直接让 reserve 抛 bad_alloc，错误信息还指不到真正的原因。
//    【概念】BOOL 的类型编号是 7，但存进 variant 时要存 bool 成员 —— 编号与
//    取值类型必须一一对应，写入侧与读取侧要对上账。
//
// ④ demo + 多方样例：
//    🟢 类型编号 4（UINT32）→ 读 4 字节小端 → GgufValue 持有 uint32_t
//    🟡 类型编号 9（ARRAY），元素类型 8（STRING），个数 3 → 读 3 个 GGUF string
//    🔴 元素类型是 9（ARRAY）→ 抛「read_value: 不支持嵌套数组」
//       元素数 2^40 → 抛「read_value: 数组元素数异常: 1099511627776」
//    调用方：src/gguf.cpp 的 GgufFile::parse 解析元数据 KV 段时调用
// ============================================================================
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
// 【对照】上游 src/llama-mmap.cpp:43 的 llama_mmap（功能相同，多了 NUMA/Windows 适配）
// 【陷阱】解析失败时必须回收映射与 fd —— load 里用 try/catch(...) 包住 parse，
//         失败路径先 munmap + close 再 rethrow，不留半开 fd 与悬挂映射。
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

// RAII：映射与 fd 的所有权跟对象走，析构即回收。
GgufFile::~GgufFile() {
    if (map_)  { munmap(map_, size_); }
    if (fd_ >= 0) { close(fd_); }
}

GgufFile::GgufFile(GgufFile && o) noexcept { *this = std::move(o); }

// 移动赋值要先释放自己手上的旧资源再接管对方的 —— 否则会泄漏一份映射与 fd。
// 接管后把对方置成「空」状态（fd=-1 / map=nullptr），它的析构就什么都不做。
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

// 向上对齐到 a 的整数倍。GGUF 的对齐值默认 32（上游 GGUF_DEFAULT_ALIGNMENT），
// 上游用 GGML_PAD 做同一件事；本实现不校验 a 是否 2 的幂（上游会校验，见 docs/01-gguf-loader.md §16.2）。
static size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

// ============================================================================
// ① 定位：五步解析 —— 魔数 → 版本与计数 → 元数据 KV → 张量表 → 对齐到数据区
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:451  gguf_init_from_reader
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:756  GGML_PAD(gr.tell(), ctx->alignment)
//    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:770  张量 offset 严格累加校验
//
// ③ 动机：
//    【陷阱】GGUF 的数据区有**双重对齐**，两处对齐的参照系不同：
//      ① 数据区起点相对**文件**对齐（把游标位置向上取整到 alignment 的倍数）；
//      ② 每个张量的 offset 相对**数据区起点**再对齐一次（在 tensor_data 里）。
//    只做①不做②，张量指针会偏掉几字节 —— 读出来是「形状对、值全错」的权重。
//    【对照】上游要求张量 offset **严格等于**前序张量填充尺寸的累加
//    （ggml/src/gguf.cpp:770），本实现放宽为「读取时再 align_up」，能吃下
//    第三方工具写出的文件（偏离已登记在 docs/01-gguf-loader.md §16.2）。
//    【陷阱】解析末尾做一次廉价越界预检：越界读是段错误，而段错误的栈指不到
//    「哪个张量坏了」。
//
// ④ demo + 多方样例：
//    🟢 版本 3 + 0 张量 0 KV → 五步走完，data_base_ = align_up(24, 32) = 32
//    🟡 v1 用 u32 存计数、v2/v3 用 u64 —— 这是三个版本间唯一的布局差异
//    🔴 张量尾部越出文件末尾 → 抛「张量 X 越出数据区（文件损坏或截断）」
//    调用方：src/gguf.cpp 的 GgufFile::load
// ============================================================================
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
    // 【对照】上游 gguf.cpp:756 是 gr.seek(GGML_PAD(gr.tell(), ctx->alignment))，
    //          张量 offset 相对此起点
    // 【陷阱】这是双重对齐的**第一处**（相对文件）；第二处（张量 offset 相对数据区
    //          起点）在 tensor_data 里补齐 —— 两处都要做，缺一处指针就偏几字节。
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
// 访问器（对外契约与四段式说明见 src/gguf.h 的声明处）
// ---------------------------------------------------------------------------

// 线性查找：模型 KV 只有几十条，纳秒级；找不到返回 nullptr 让调用方自己判「可缺省」。
const GgufValue * GgufFile::find(const std::string & key) const {
    for (const auto & [k, v] : metadata_) {
        if (k == key) { return &v; }
    }
    return nullptr;
}

// get_string 单独一份而不走 get_typed：它要返回**引用**（避免拷贝整段文本），
// 而 get_typed 按值返回；语义上两者一致，都是「类型不符就抛」。
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

// get_array 与 get_string 同理要返回**引用**：词表的 tokens / scores 动辄几万条，
// 按值返回会把整个数组拷一遍。
const GgufArray & GgufFile::get_array(const std::string & key) const {
    const GgufValue * v = find(key);
    if (!v) { throw std::runtime_error("GGUF 缺少元数据 key: " + key); }
    if (!std::holds_alternative<GgufArray>(v->v)) {
        throw std::runtime_error("GGUF 元数据 " + key + " 不是 array");
    }
    return std::get<GgufArray>(v->v);
}

// 精确匹配张量名；上游 gguf_find_tensor 找不到时返回 -1，本实现返回 nullptr。
const TensorInfo * GgufFile::find_tensor(const std::string & name) const {
    for (const auto & t : tensors_) {
        if (t.name == name) { return &t; }
    }
    return nullptr;
}

// 字节数 = 块数 × 块字节数；块只能跨首维切，所以首维先向上取整到块的整数倍。
size_t GgufFile::tensor_nbytes(const TensorInfo & info) const {
    auto [blck, blck_size] = ggml_block_params(info.type);
    if (info.dims.empty()) { return 0; }
    // 行内按块取整（首维最快移动），其余维直接相乘
    uint64_t row_blocks = (info.dims[0] + blck - 1) / blck;
    uint64_t n = row_blocks;
    for (size_t i = 1; i < info.dims.size(); i++) { n *= info.dims[i]; }
    return static_cast<size_t>(n) * blck_size;
}

// 双重对齐的第二处：offset 相对数据区起点再对齐一次；返回前再查一次是否越过
// 映射末尾（解析期的预检只覆盖文件损坏，越界读仍然是段错误）。
const void * GgufFile::tensor_data(const TensorInfo & info) const {
    size_t off = align_up(info.offset, alignment_);
    if (data_base_ + off + tensor_nbytes(info) >
        static_cast<const uint8_t *>(map_) + size_) {
        throw std::runtime_error("张量 " + info.name + " 数据越界");
    }
    return data_base_ + off;
}

} // namespace minillama
