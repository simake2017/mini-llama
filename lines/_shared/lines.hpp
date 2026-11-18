// ============================================================================
// 全部业务线的声明（供 tests/test_lines.cpp 与合集入口调用）
//
// ① 定位
//   一份头文件声明 8 条线的线本体入口，外加「模型文件在哪」的统一解析。
//
// ② 源码锚点
//   本仓库无上游对应物。对照 line 技能的结构约定：
//     ~/wiki/common/skills/line/references/线结构规范.md §三「四件事」
//     （线本体 = 一个 run_<线名>(LineCtx&)，辅助函数收在匿名 ns）
//
// ③ 动机
//   【概念】线本体必须在**全局命名空间可达**——测试用例的绿箭头要能落在它身上，
//   单步时才能从调用点直接进线本体。收进匿名 ns 的话 IDE 只能看到壳。
//
// ④ demo + 多方样例
//   🟢 tests/test_lines.cpp：quiet 跑一遍，8 条线全过返回 0
//   🟡 ./build/mini-llama-lines prefill
//   🔴 ./build/mini-llama-lines prefill --step
// ============================================================================

#pragma once

#include "line.hpp"

#include <string>

namespace minillama {
namespace lines {

// ---------------------------------------------------------------------------
// 模型文件定位：环境变量 > 仓库默认路径
// ---------------------------------------------------------------------------
// 【概念】线要「真实发生过」才算数，所以加载类线必须拿到真的 GGUF 文件。
//   仓库里的 models/*.gguf 是 gitignore 的（1GB+），所以先看环境变量
//   MINILLAMA_MODEL，没有再退回仓库默认路径。两个都没有 → 这条线标记为「跳过」
//   （不是失败——缺外部条件不是线写错了）。
std::string resolve_model_path();

// ---------------------------------------------------------------------------
// 8 条线的线本体
// ---------------------------------------------------------------------------
// 载入域：一个 GGUF 文件 → 可前向的模型 / 可用的词表
int run_model_load(LineCtx & ctx);
int run_vocab_load(LineCtx & ctx);

// 文本域：prompt 文本 → token 序列 → 文本
int run_tokenize(LineCtx & ctx);
int run_detokenize(LineCtx & ctx);

// 推理工域：token 序列 → 写满 KV cache 的 logits → 自回归生成
int run_prefill(LineCtx & ctx);
int run_decode(LineCtx & ctx);

// 数值域：量化字节 → F32 权重 / logits → 采样出的 token
int run_dequant(LineCtx & ctx);
int run_sample(LineCtx & ctx);

} // namespace lines
} // namespace minillama
