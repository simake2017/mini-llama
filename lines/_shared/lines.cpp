// ============================================================================
// 业务线的公共辅助：模型文件定位
//
// ① 定位
//   加载类线要拿到真的 GGUF 文件才能跑；本文件统一「文件在哪」的解析口径。
//
// ② 源码锚点
//   本仓库无上游对应物。行为对照上游的模型查找约定：
//     llama.cpp @ 88b47a755  src/llama-model.cpp: llama_model_params 默认模型路径解析
//     （上游从 argv / 环境变量取模型路径，取不到就报错退出——本线取不到就「跳过」）
//
// ③ 动机
//   【概念】线是**研究产物**，跑不起来时必须能区分两种情况：
//     「线写错了」和「环境没准备好」。所以取不到模型 → skip（不是 fail）。
//   这个区分很重要：否则 ctest 在没模型的机器上会红，而红的原因跟线本身无关。
//
// ④ demo + 多方样例
//   🟢 MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_model_load
//   🟡 ./build/mini-llama-lines prefill          （用仓库默认路径）
//   🔴 MINILLAMA_MODEL=/nowhere/x.gguf ./build/line_model_load   → 打印「跳过：找不到模型文件」
// ============================================================================

#include "lines.hpp"

#include <cstdio>
#include <cstdlib>

namespace minillama {
namespace lines {

std::string resolve_model_path() {
    // 环境变量**存在**就以它为准（哪怕是空串——空串 = 明确表示「这次不挂模型」，
    // ctest 就是这么把重测试摘出去的）；不存在才退回仓库默认路径。
    if (const char * env = std::getenv("MINILLAMA_MODEL")) {
        return env;
    }
    return "models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf";
}

} // namespace lines
} // namespace minillama
