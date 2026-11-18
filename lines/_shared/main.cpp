// ============================================================================
// 线合集入口：./mini-llama-lines [--list | <线名> | all] [--step]
//
// ① 定位
//   把所有注册进来的线汇成一个可执行文件：列出、跑一条、跑全部。
//
// ② 源码锚点
//   本仓库无上游对应物。对照 line 技能的结构约定：
//     ~/wiki/common/skills/line/references/线结构规范.md §一「目录形状」
//     （`_shared/line.cpp` = 注册表实现 + main（--list / <线名> / all / --step））
//
// ③ 动机
//   【概念】线是**研究产物**，读者需要「一次看全有哪些线」和「一条一条单跑」两种视角。
//   拆成 N 个可执行文件会让「有哪些线」这件事变得不可见；合成一个又不能单跑。
//   所以：合集程序负责「选哪几条跑」，单跑程序负责「只跑一条」——两个入口共用同一份
//   run_<线名>，不存在两套实现。
//
// ④ demo + 多方样例
//   🟢 ./build/mini-llama-lines --list          看有哪些线
//   🟡 ./build/mini-llama-lines all             全跑一遍（退出码 = 不符条数）
//   🔴 ./build/mini-llama-lines all --step      全跑 + 逐跳停
// ============================================================================

#include "line.hpp"

#include <cstring>
#include <iostream>
#include <string>

using namespace minillama::lines;

static void print_list() {
    const auto & entries = registry().entries();
    std::cout << "已注册 " << entries.size() << " 条线：\n";
    for (const auto & e : entries) {
        std::cout << "  " << e.name << "  ——  " << e.desc << '\n';
    }
    std::cout << "\n用法: mini-llama-lines [--list | <线名> | all] [--step]\n";
}

int main(int argc, char ** argv) {
    std::string which;
    bool step = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--list" || a == "-l") {
            print_list();
            return 0;
        } else if (a == "--step") {
            step = true;
        } else if (a == "all") {
            which = "all";
        } else if (a.rfind("--", 0) == 0) {
            std::cerr << "未知参数: " << a << "\n";
            return 2;
        } else if (which.empty()) {
            which = a;
        } else {
            std::cerr << "只能指定一条线（或 all），收到: " << a << "\n";
            return 2;
        }
    }

    if (which.empty()) {
        print_list();
        return 0;
    }

    const auto & entries = registry().entries();
    int bad = 0;

    if (which == "all") {
        for (const auto & e : entries) {
            LineCtx ctx;
            ctx.name = e.name;
            ctx.step = step;
            bad += e.fn(ctx);
        }
        std::cout << "\n== 合集走完：共 " << entries.size() << " 条线，不符 " << bad << " 项\n";
        return bad;
    }

    const LineEntry * e = registry().find(which);
    if (!e) {
        std::cerr << "没有这条线: " << which << "\n";
        print_list();
        return 2;
    }
    LineCtx ctx;
    ctx.name = e->name;
    ctx.step = step;
    return e->fn(ctx);
}
