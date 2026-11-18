// ============================================================================
// 线代码公共 harness：LineCtx（日志 / 观测 / 轻量自检）+ 注册表 + 入口宏
//
// ① 定位
//   所有业务线共用的最小骨架：每跳打日志、记录观测值、做不会中断的轻量自检。
//   本文件**不含任何一条线的逻辑**——线的逻辑一律在 lines/<族>/<线名>_line.cpp。
//
// ② 源码锚点
//   本仓库没有上游对应物（llama.cpp 没有「线」这种组织形态）。
//   对照的是 line 技能的结构约定：
//     ~/wiki/common/skills/line/references/线结构规范.md §四 harness API
//     ~/wiki/common/skills/line/examples/2-线代码样例/line_sample.cpp
//
// ③ 动机
//   【概念】一条线 = 一个端到端业务子流程，必须**跑得出、看得见**。
//   所以 harness 只提供三件事：跳（把「现在走到哪一步」显式化）、
//   观测（把「这一步看到了什么数据」显式化）、自检（把「看到的是不是预期」显式化）。
//
//   【概念】自检**绝不中断**——✗ 只标记不抛异常，整条流每次都能完整走完。
//   理由：线的用途是**对照**（本实现 vs 真实版）。中途抛异常会让后半条线看不到，
//   而「后半段也跑一遍」恰恰是发现问题的关键。
//
//   【性能】quiet 模式下过程日志全部静默，只有 ✗ 冒出来；out() 换成空 streambuf，
//   线里的手绘表格代码照常执行、一个字符都不落屏——不用为「要不要打印」分叉。
//
// ④ demo + 多方样例
//   🟢 最小（测试入口）：quiet 跑一遍，全过一声不吭，返回 0
//   🟡 典型（单跑一条）：./build/line_model_load            逐跳打日志
//   🔴 极限（逐跳单步）：./build/line_model_load --step     每跳停一下等回车
//   另一个视角：./build/mini-llama-lines --list            看有哪些线
//                ./build/mini-llama-lines all              全跑一遍，退出码 = 不符条数
//
//   调用方：lines/<族>/<线名>_line.cpp（写线）、tests/test_lines.cpp（跑线）
// ============================================================================

#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

namespace minillama {
namespace lines {

// ---------------------------------------------------------------------------
// LineCtx：一跳一记的观测上下文
// ---------------------------------------------------------------------------
class LineCtx {
public:
    bool quiet = false;   // true = 过程日志全静默（测试入口用），只有 ✗ 打印
    bool step  = false;   // true = 每跳停一下等回车（--step）
    int  failed = 0;      // 不符条数
    int  total  = 0;      // 检查总条数
    bool skipped = false; // true = 这条线因为缺少外部条件没能跑（不是失败）
    std::string skip_reason;
    std::string name;     // 线名，用于 ✗ 行首标识

    // ------------------------------------------------------------------
    // 第 N 跳：标题 + 源码锚点 + 验证方式
    // ------------------------------------------------------------------
    void step_at(int n, const std::string & what,
                 const std::string & src = "", const std::string & verify = "") {
        if (quiet) return;
        std::cout << "\n▸ 第 " << n << " 跳：" << what << '\n';
        if (!src.empty())    std::cout << "    源码 : " << src << '\n';
        if (!verify.empty()) std::cout << "    验证 : " << verify << '\n';
        if (step) { std::cout << "    [回车继续]"; std::cin.get(); }
    }

    // ------------------------------------------------------------------
    // 观测值：这一步「看到了什么数据」
    // ------------------------------------------------------------------
    template <class T>
    void value(const std::string & label, const T & v) {
        if (quiet) return;
        std::cout << "    观测 : " << label << " = " << v << '\n';
    }

    // 十六进制观测值（地址 / 魔数）
    void hex(const std::string & label, uint64_t v) {
        if (quiet) return;
        std::cout << "    观测 : " << label << " = 0x" << std::hex << v << std::dec << '\n';
    }

    // 一行说明（缩进在观测之下，不占跳的层级）
    void log(const std::string & text) {
        if (quiet) return;
        std::cout << "            " << text << '\n';
    }

    // 多行图 / 表原样输出
    void raw(const std::string & text) {
        if (quiet) return;
        std::cout << text << '\n';
    }

    // 自绘图表用的流：quiet 模式下写进去就丢（代码照跑，不落屏）
    std::ostream & out() { return quiet ? null_stream_ : std::cout; }

    // ------------------------------------------------------------------
    // 轻量自检：不符合预期打 ✗ 但**绝不中断**
    // ------------------------------------------------------------------
    void check(bool ok, const std::string & what) {
        ++total;
        if (!ok) {
            ++failed;
            std::cout << "    ✗ [" << name << "] " << what << '\n';
        }
    }

    void check(bool ok, const std::string & what,
               const std::string & actual, const std::string & expected) {
        ++total;
        if (!ok) {
            ++failed;
            std::cout << "    ✗ [" << name << "] " << what
                      << "  实际=" << actual << "  预期=" << expected << '\n';
        }
    }

    // ------------------------------------------------------------------
    // 收尾汇总：返回不符条数（0 = 全过）
    // ------------------------------------------------------------------
    int summary() const {
        if (skipped) {
            if (!quiet) std::cout << "\n== 线 " << name << " 跳过：" << skip_reason << '\n';
            return 0;
        }
        if (quiet) {
            if (failed == 0) return 0;           // 全过一声不吭
            std::cout << "线 " << name << "：" << failed << " 项不符\n";
            return failed;
        }
        std::cout << "\n== 线 " << name << " 走完："
                  << (total - failed) << "/" << total << " 项通过";
        if (failed) std::cout << "（" << failed << " 项不符）";
        std::cout << '\n';
        return failed;
    }

    // 这条线因为缺少外部条件没跑起来（不是失败）
    void skip(const std::string & reason) {
        skipped = true;
        skip_reason = reason;
    }

private:
    class NullBuf : public std::streambuf {
    public:
        int overflow(int c) override { return c; }
    };
    NullBuf      null_sink_;
    std::ostream null_stream_{&null_sink_};
};

// ---------------------------------------------------------------------------
// 线注册表：编译进去就出现在 --list，不改任何公共文件
// ---------------------------------------------------------------------------
using LineFn = std::function<int(LineCtx &)>;

struct LineEntry {
    std::string name;
    std::string desc;
    LineFn       fn;
};

class LineRegistry {
public:
    void add(const std::string & name, const std::string & desc, LineFn fn);
    const std::vector<LineEntry> & entries() const { return entries_; }
    const LineEntry * find(const std::string & name) const;

private:
    std::vector<LineEntry> entries_;
};

LineRegistry & registry();
void register_line(const char * name, const char * desc, LineFn fn);

// 测试入口的公共壳：quiet 跑一遍，返回不符条数（0 = 全过）
int run_entry_checked(const std::string & name, LineFn fn);

// 独立 main 的公共壳：解析 --step，然后跑这一条线
int run_standalone(const std::string & name, LineFn fn, int argc, char ** argv);

// ---------------------------------------------------------------------------
// 宏
// ---------------------------------------------------------------------------

// 自注册：把一条线挂进注册表
struct LineRegistrar {
    LineRegistrar(const char * n, const char * d, LineFn f) { register_line(n, d, f); }
};
#define MINILLAMA_LINE_REGISTER(var, name_str, desc_str, fn) \
    static ::minillama::lines::LineRegistrar var(name_str, desc_str, fn)

// 独立 main：只在编译单跑目标时打开（不是 IDE 绿箭头，套条件编译没问题）
#define MINILLAMA_LINE_STANDALONE_MAIN(name_str, fn)                             \
    int main(int argc, char ** argv) {                                           \
        return ::minillama::lines::run_standalone(name_str, fn, argc, argv);      \
    }

} // namespace lines
} // namespace minillama
