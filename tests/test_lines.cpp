// 业务线测试：quiet 跑一遍全部 8 条线
//
// 策略：每条线都是「端到端子流程 + 轻量自检」，自检不中断、只标 ✗。
//       这里把每条线当一个用例跑，退出码 = 不符条数（0 = 全过）。
//
// 运行：
//   ./build/test_lines                    （无模型文件时相关线自动「跳过」）
//   MINILLAMA_MODEL=<路径> ./build/test_lines
//
// 注意：带真模型时 prefill / decode 线会跑真实的朴素前向，耗时较长
//       （TinyLlama 约 1~2 秒/token）。

#include "line.hpp"
#include "lines.hpp"

#include <cstdio>
#include <iostream>
#include <string>

using namespace minillama::lines;

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

// 跑一条线：quiet 模式，过程日志全静默，只有 ✗ 冒出来
static void run_line(const char * name, int (*fn)(LineCtx &)) {
    LineCtx ctx;
    ctx.quiet = true;
    ctx.name  = name;
    const int bad = fn(ctx);
    if (bad != 0) {
        std::cerr << "FAIL line " << name << ": " << bad << " 项不符\n";
        ++g_fail;
    } else {
        ++g_pass;
    }
}

int main() {
    // 载入域
    run_line("model_load", ::minillama::lines::run_model_load);
    run_line("vocab_load", ::minillama::lines::run_vocab_load);
    // 文本域
    run_line("tokenize",    ::minillama::lines::run_tokenize);
    run_line("detokenize",  ::minillama::lines::run_detokenize);
    // 推理工域
    run_line("prefill",     ::minillama::lines::run_prefill);
    run_line("decode",      ::minillama::lines::run_decode);
    // 数值域
    run_line("dequant",     ::minillama::lines::run_dequant);
    run_line("sample",      ::minillama::lines::run_sample);

    std::cout << "passed=" << g_pass << " failed=" << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
