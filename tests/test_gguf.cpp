// 单元 1 测试：GGUF 解析器
//
// 策略：测试自己构造合法的 GGUF 字节流（minitest::Writer），写到临时文件再解析回来，
//       不需要依赖任何外部模型文件就能覆盖全部代码路径；
//       末尾可选对真实文件（环境变量 TEST_GGUF_REAL_FILE）做冒烟测试。

#include "gguf.h"
#include "gguf_writer.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace minillama;
using minitest::Writer;
using minitest::write_temp;

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__            \
                      << "  CHECK(" #cond ")\n";                           \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

#define CHECK_THROW(stmt)                                                  \
    do {                                                                   \
        bool thrown = false;                                               \
        try { stmt; } catch (const std::exception &) { thrown = true; }    \
        if (!thrown) {                                                     \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__            \
                      << "  期望抛异常但未抛: " #stmt "\n";                \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// TestGguf_ParseBasic：最小合法文件 —— 头 + 若干元数据 + 2 个张量 + 数据区
// ---------------------------------------------------------------------------
static void test_parse_basic() {
    Writer w;
    w.u32(0x46554747u);     // magic "GGUF"
    w.u32(3);               // version
    w.u64(2);               // tensor_count
    w.u64(9);               // metadata_kv_count

    w.kv_header("general.architecture", 8); w.str("llama");
    w.kv_header("general.name", 8);         w.str("mini-test");
    w.kv_header("llama.context_length", 4); w.u32(2048);
    w.kv_header("llama.embedding_length", 4); w.u32(64);
    w.kv_header("test.negative", 11);       w.i64(-42);
    w.kv_header("test.pi", 6);              w.f32(3.1415926f);
    w.kv_header("test.flag", 7);            w.u8(1);
    w.kv_header("tokenizer.ggml.tokens", 9); // array of string
        w.u32(8); w.u64(3);
        w.str("<unk>"); w.str("<s>"); w.str("</s>");
    w.kv_header("test.scores", 9);           // array of f32
        w.u32(6); w.u64(2);
        w.f32(0.5f); w.f32(-1.25f);

    // 张量表
    w.str("token_embd.weight"); w.u32(2); w.u64(64); w.u64(100);
        w.u32(0);  // F32
        w.u64(0);  // offset（数据区内）
    w.str("output.weight");     w.u32(2); w.u64(100); w.u64(64);
        w.u32(0);
        w.u64(64 * 100 * 4);  // 紧跟上一个张量（读时按对齐修正）

    // 数据区：起点对齐到 32 字节（与读取方 align_up(header_end) 约定一致）
    w.align();
    for (int i = 0; i < 64 * 100; i++) { w.f32(static_cast<float>(i) * 0.001f); }
    w.align();
    for (int i = 0; i < 100 * 64; i++) { w.f32(1.0f); }

    std::string path = write_temp(w.buf(), "basic");
    GgufFile f = GgufFile::load(path);

    CHECK(f.version() == 3);
    CHECK(f.tensor_count() == 2);
    CHECK(f.get_string("general.architecture") == "llama");
    CHECK(f.get_u32("llama.context_length") == 2048);
    CHECK(f.get_i64("test.negative") == -42);
    CHECK(f.get_f32("test.pi") > 3.14 && f.get_f32("test.pi") < 3.15);

    const auto & tokens = f.get_array("tokenizer.ggml.tokens");
    CHECK(tokens.size() == 3);
    CHECK(std::get<std::string>(tokens[0].v) == "<unk>");

    const TensorInfo * embd = f.find_tensor("token_embd.weight");
    CHECK(embd != nullptr);
    CHECK(embd->dims.size() == 2 && embd->dims[0] == 64 && embd->dims[1] == 100);
    CHECK(f.tensor_nbytes(*embd) == 64 * 100 * 4);

    // 数据内容抽查：embd[5][7]（首维最快移动）= 5 + 7*64 号元素
    const float * data = static_cast<const float *>(f.tensor_data(*embd));
    CHECK(data[0] == 0.0f);
    CHECK(data[5 + 7 * 64] == (5 + 7 * 64) * 0.001f);

    CHECK(f.find("no.such.key") == nullptr);
    CHECK(f.find_tensor("no.such.tensor") == nullptr);

    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// TestGguf_QuantTensorSize：量化张量按块计算字节数
// ---------------------------------------------------------------------------
static void test_quant_tensor_size() {
    CHECK(ggml_block_params(GgmlType::Q4_K) == std::make_pair(256, 144));
    CHECK(ggml_block_params(GgmlType::Q8_0) == std::make_pair(32, 34));
    CHECK(ggml_block_params(GgmlType::F32) == std::make_pair(1, 4));

    Writer w;
    w.u32(0x46554747u); w.u32(3); w.u64(1); w.u64(0);
    w.str("blk.0.ffn_gate.weight"); w.u32(2); w.u64(256); w.u64(64);
        w.u32(12);  // Q4_K
        w.u64(0);
    w.align();
    for (int i = 0; i < 9216 / 4; i++) { w.f32(0.0f); }

    std::string path = write_temp(w.buf(), "quant");
    GgufFile f = GgufFile::load(path);
    const TensorInfo * t = f.find_tensor("blk.0.ffn_gate.weight");
    CHECK(t != nullptr && t->type == GgmlType::Q4_K);
    CHECK(f.tensor_nbytes(*t) == 9216);

    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// 错误场景（不可信输入防御）
// ---------------------------------------------------------------------------
static void test_bad_magic() {
    Writer w;
    w.u32(0x464c4747u); w.u32(3); w.u64(0); w.u64(0);
    std::string path = write_temp(w.buf(), "magic");
    CHECK_THROW(GgufFile::load(path));
    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

static void test_bad_version() {
    Writer w;
    w.u32(0x46554747u); w.u32(4); w.u64(0); w.u64(0);
    std::string path = write_temp(w.buf(), "version");
    CHECK_THROW(GgufFile::load(path));
    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

static void test_truncated() {
    Writer w;
    w.u32(0x46554747u); w.u32(3); w.u64(0); w.u64(1);
    w.kv_header("general.architecture", 8);
    w.u64(100);              // 声明 100 字节字符串…
    w.raw_bytes("lla", 3);   // …但实际只有 3 字节，文件就此结束 —— 截断
    std::string path = write_temp(w.buf(), "trunc");
    CHECK_THROW(GgufFile::load(path));
    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

static void test_wrong_type_access() {
    Writer w;
    w.u32(0x46554747u); w.u32(3); w.u64(0); w.u64(1);
    w.kv_header("some.key", 4); w.u32(7);
    std::string path = write_temp(w.buf(), "wrongtype");
    GgufFile f = GgufFile::load(path);
    CHECK_THROW(f.get_string("some.key"));   // 类型不符
    CHECK_THROW(f.get_u32("missing.key"));   // key 不存在
    std::remove(path.c_str());
    if (g_fail == 0) { ++g_pass; }
}

// ---------------------------------------------------------------------------
// TestGguf_RealFile：对真实文件冒烟（需要环境变量指定路径）
// ---------------------------------------------------------------------------
static void test_real_file() {
    const char * path = std::getenv("TEST_GGUF_REAL_FILE");
    if (!path || !*path) {
        std::cout << "  [skip] 未设置 TEST_GGUF_REAL_FILE，跳过真实文件冒烟\n";
        return;
    }
    GgufFile f = GgufFile::load(path);
    CHECK(f.get_string("general.architecture").size() > 0);
    std::cout << "  real file: " << path
              << "  version=" << f.version()
              << "  tensors=" << f.tensor_count() << "\n";
    if (f.tensor_count() > 0) {
        const TensorInfo & t = f.tensors()[0];
        CHECK(f.tensor_data(t) != nullptr);
        std::cout << "  tensor[0]: " << t.name << " type=" << ggml_type_name(t.type) << "\n";
    }
    if (g_fail == 0) { ++g_pass; }
}

int main() {
    test_parse_basic();
    test_quant_tensor_size();
    test_bad_magic();
    test_bad_version();
    test_truncated();
    test_wrong_type_access();
    test_real_file();

    std::cout << "passed=" << g_pass << " failed=" << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
