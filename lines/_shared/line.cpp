// ============================================================================
// 线注册表实现 + 入口公共壳
//
// ① 定位
//   实现 lines/_shared/line.hpp 声明的注册表，并提供「测试入口」「独立 main」两个壳。
//   本文件不含任何一条线的逻辑。
//
// ② 源码锚点
//   本仓库无上游对应物。对照 line 技能的结构约定：
//     ~/wiki/common/skills/line/references/线结构规范.md §三「四件事」
//     ~/wiki/common/skills/line/examples/2-线代码样例/line_sample.cpp
//
// ③ 动机
//   【概念】三个入口（测试用例 / 单跑 / 合集）必须调的是**同一份** run_<线名>，
//   否则就会出现「测试过了但命令行跑不对」这种最难查的分叉。所以这里只提供壳，
//   壳里只做「参数解析 + quiet 开关」，线本体一行都不碰。
//
// ④ demo + 多方样例
//   🟢 ./build/mini-llama-lines --list     列出全部线
//   🟡 ./build/mini-llama-lines model_load 跑一条
//   🔴 ./build/mini-llama-lines all --step  全跑 + 逐跳停
// ============================================================================

#include "line.hpp"

#include <cstring>

namespace minillama {
namespace lines {

// ---------------------------------------------------------------------------
// 注册表
// ---------------------------------------------------------------------------
void LineRegistry::add(const std::string & name, const std::string & desc, LineFn fn) {
    for (auto & e : entries_) {
        if (e.name == name) {          // 同名覆盖：重编译时避免重复登记
            e.desc = desc;
            e.fn = fn;
            return;
        }
    }
    entries_.push_back(LineEntry{name, desc, fn});
}

const LineEntry * LineRegistry::find(const std::string & name) const {
    for (const auto & e : entries_) {
        if (e.name == name) return &e;
    }
    return nullptr;
}

LineRegistry & registry() {
    static LineRegistry r;
    return r;
}

void register_line(const char * name, const char * desc, LineFn fn) {
    registry().add(name, desc, fn);
}

// ---------------------------------------------------------------------------
// 入口壳
// ---------------------------------------------------------------------------

// 测试入口：quiet 跑一遍，只让 ✗ 冒出来。返回值 = 不符条数（0 = 全过）。
int run_entry_checked(const std::string & name, LineFn fn) {
    LineCtx ctx;
    ctx.quiet = true;
    ctx.name  = name;
    if (!fn) return 0;
    return fn(ctx);
}

// 独立 main：解析 --step，跑这一条线。
int run_standalone(const std::string & name, LineFn fn, int argc, char ** argv) {
    LineCtx ctx;
    ctx.name = name;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--step") == 0) ctx.step = true;
    }
    if (!fn) {
        std::cerr << "线 " << name << " 没有注册\n";
        return 1;
    }
    return fn(ctx);
}

} // namespace lines
} // namespace minillama
