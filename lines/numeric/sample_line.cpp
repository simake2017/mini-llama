// ============================================================================
// 线⑧ 采样：logits → softmax（未加温度）→ top-k 截断 → top-p 核采样 → 温度 → 按概率抽样
//
// 对照 llama.cpp @ 88b47a755:
//   src/llama-sampler.cpp  llama_sampler_top_k（截断）/ llama_sampler_top_p（核采样）
//                         llama_sampler_temp（温度）/ llama_sampler_dist（抽样）
//                         llama_sampler_greedy（temp=0 的退化分支）
//   common/common.h:259   默认采样链的**节序**：top-k → top-p → … → temperature → dist
//                         （即截断发生在温度生效之前——本仓库照此对齐）
//
// 跨模块：src/sampler.cpp（采样链）· src/model.cpp（softmax 内核 + logits 来源）
//         · src/main.cpp（CLI flag → SamplerParams）
//
// 学习问题：top-p 截断时，「跨过阈值的那一个 token」为什么要保留？
//           丢掉它和保留它，采样结果有什么区别？
//
// 跑：./build/line_sample   （这条线用合成 logits，不依赖模型文件）
// ============================================================================

#include "line.hpp"
#include "lines.hpp"
#include "model.h"
#include "sampler.h"

#include <cmath>
#include <string>
#include <vector>

using namespace minillama;
using namespace minillama::lines;

namespace {

// 一组能手工算出概率的 logits：{3, 2, 1, 0, -1}
std::vector<float> sample_logits() {
    return {3.0f, 2.0f, 1.0f, 0.0f, -1.0f};
}

// 观察采样器内部用的概率分布（走的是项目里真实的 softmax 内核）
std::vector<float> probs_of(const std::vector<float> & logits, float temp) {
    std::vector<float> p = logits;
    const float inv = 1.0f / temp;
    for (auto & x : p) x *= inv;
    softmax(p.data(), static_cast<int>(p.size()));
    return p;
}

} // namespace

// ---------------------------------------------------------------------------
// 线本体
// ---------------------------------------------------------------------------
int minillama::lines::run_sample(LineCtx & ctx) {
    const std::vector<float> logits = sample_logits();

    // ------------------------------------------------------------------
    ctx.step_at(1, "输入：一段未归一化的 logits（模型输出头的原始分数）",
                "src/model.cpp: Model::forward_token 的返回值（对照 llama_context 的 logits 输出）",
                "真实链路里长度 = 词表大小（32000~151936）；这里用 5 个可手算的值");
    for (size_t i = 0; i < logits.size(); i++) {
        ctx.log("logits[" + std::to_string(i) + "] = " + std::to_string(logits[i]));
    }

    // ------------------------------------------------------------------
    ctx.step_at(2, "softmax：把分数变成概率分布（截断的依据是**未加温度**的这份）",
                "src/sampler.cpp: Sampler::sample 的步骤 2 + src/model.cpp: softmax（数值稳定版）",
                "softmax 先减最大值，避免 exp 溢出");
    {
        const auto p = probs_of(logits, 1.0f);
        double sum = 0.0;
        for (float v : p) sum += v;
        for (size_t i = 0; i < p.size(); i++) {
            ctx.log("P(" + std::to_string(i) + ") = " + std::to_string(p[i]));
        }
        ctx.value("概率和", sum);
        ctx.check(std::fabs(sum - 1.0) < 1e-5, "概率和 = 1");
        ctx.check(p[0] > p[1] && p[1] > p[2] && p[2] > p[3] && p[3] > p[4],
                  "概率按 logits 降序排列");
    }

    // ------------------------------------------------------------------
    ctx.step_at(3, "温度的作用：temp 越大分布越平（多样性 ↑，稳定性 ↓）",
                "src/sampler.cpp: Sampler::sample 的步骤 5（对照 llama_sampler_temp）",
                "tests/test_generate.cpp: 温度尖峰用例");
    {
        const auto cold = probs_of(logits, 0.25f);
        const auto hot  = probs_of(logits, 2.0f);
        ctx.value("temp=0.25 时 P(0)", cold[0]);
        ctx.value("temp=2.0  时 P(0)", hot[0]);
        ctx.check(cold[0] > hot[0], "低温时最大 token 的概率更高（更尖）");
    }

    // ------------------------------------------------------------------
    ctx.step_at(4, "贪心分支：temp<=0 时直接取 argmax，截断与温度全部不参与",
                "src/sampler.cpp: Sampler::sample 的贪心段（对照 llama_sampler_greedy）",
                "tests/test_generate.cpp: 贪心可复现用例");
    {
        SamplerParams sp;
        sp.temp = 0.0f;
        Sampler s(sp);
        const int t = s.sample(logits.data(), static_cast<int>(logits.size()));
        ctx.value("贪心选出的下标", t);
        ctx.log("本线的 logits 是 {3,2,1,0,-1}——最大值在**开头**，所以 argmax = 0");
        ctx.check(t == 0, "贪心取 argmax = 第 0 个（logits 最大的是 3.0）");
    }

    // ------------------------------------------------------------------
    ctx.step_at(5, "top-k 截断：只保留概率最高的 k 个",
                "src/sampler.cpp: Sampler::sample 的步骤 3（对照 llama_sampler_top_k）",
                "tests/test_generate.cpp: top_k=1 的确定性用例");
    {
        SamplerParams sp;
        sp.temp = 1.0f;
        sp.top_k = 1;
        sp.top_p = 0.0f;   // 关掉 top-p，只看 top-k
        sp.seed = 7;
        Sampler s(sp);
        const int t = s.sample(logits.data(), static_cast<int>(logits.size()));
        ctx.value("top_k=1 选出的下标", t);
        ctx.check(t == 0, "top_k=1 时唯一候选就是 argmax（与随机无关）");
    }

    // ------------------------------------------------------------------
    ctx.step_at(6, "top-p 核采样：保留累积概率达 p 的最小集合，**含越界那一个**",
                "src/sampler.cpp: Sampler::sample 的步骤 4（对照 llama_sampler_top_p）",
                "tests/test_generate.cpp: top_p 截断用例");
    {
        // 累积概率（未加温度）：0.636409 / 0.870531 / 0.956660 ...
        // top_p=0.9 → 第三个候选把累积概率推过阈值 → 保留前 3 个
        SamplerParams sp;
        sp.temp = 1.0f;
        sp.top_k = 0;      // 不限数量，只看 top-p
        sp.top_p = 0.9f;
        sp.seed = 7;
        Sampler s(sp);
        std::vector<int> hits;
        for (int i = 0; i < 200; i++) {
            hits.push_back(s.sample(logits.data(), static_cast<int>(logits.size())));
        }
        bool saw0 = false, saw1 = false, saw2 = false, saw3 = false, saw4 = false;
        for (int h : hits) {
            if (h == 0) saw0 = true;
            if (h == 1) saw1 = true;
            if (h == 2) saw2 = true;
            if (h == 3) saw3 = true;
            if (h == 4) saw4 = true;
        }
        ctx.log("200 次采样命中：0=" + std::string(saw0 ? "是" : "否") +
                " 1=" + std::string(saw1 ? "是" : "否") +
                " 2=" + std::string(saw2 ? "是" : "否") +
                " 3=" + std::string(saw3 ? "是" : "否") +
                " 4=" + std::string(saw4 ? "是" : "否"));
        ctx.check(saw0 && saw1 && saw2, "前 3 个候选都出现过（累积概率 0.9567 >= 0.9）");
        ctx.check(!saw3 && !saw4, "第 4、5 个候选一次都没出现（被 top-p 截断）");
        ctx.log("「含越界那一个」：把累积概率推过 p 的那个 token 要留着——否则分布被砍得比 p 还狠");

        // 节序的可见后果：截断发生在温度生效之前，所以 temp 变大也不会放宽 top-p 的候选集
        //   temp=2 时未加温度的累积仍是 0.636/0.871/0.957 → 仍然只留 3 个
        //   （若先加温度再截断，累积变成 0.429/0.689/0.846/0.942 → 会留 4 个）
        SamplerParams sp2;
        sp2.temp = 2.0f;
        sp2.top_k = 0;
        sp2.top_p = 0.9f;
        sp2.seed = 7;
        Sampler s2(sp2);
        bool hot3 = false;
        for (int i = 0; i < 200; i++) {
            if (s2.sample(logits.data(), static_cast<int>(logits.size())) == 3) { hot3 = true; }
        }
        ctx.value("temp=2.0 时 token 3 是否被抽中", hot3 ? "是" : "否");
        ctx.check(!hot3, "temp 变大也不会放宽 top-p 的候选集（截断在温度之前）");
    }

    // ------------------------------------------------------------------
    ctx.step_at(7, "按概率抽样：截断后的集合内重新归一，再按累积概率抽",
                "src/sampler.cpp: Sampler::sample 的步骤 6（对照 llama_sampler_dist）",
                "tests/test_generate.cpp: 固定种子可复现用例");
    {
        SamplerParams sp;
        sp.temp = 1.0f;
        sp.top_k = 0;
        sp.top_p = 0.0f;
        sp.seed = 1234;
        Sampler s1(sp), s2(sp);
        const int a = s1.sample(logits.data(), static_cast<int>(logits.size()));
        const int b = s2.sample(logits.data(), static_cast<int>(logits.size()));
        ctx.value("两次采样结果", a);
        ctx.check(a == b, "同种子两次采样结果相同（随机数消费顺序一致）");
    }

    // ------------------------------------------------------------------
    ctx.step_at(8, "结局：挑出的 token id",
                "src/main.cpp: 生成循环里的 sampler.sample(...)(消费方)",
                "./build/mini-llama --model <模型> --prompt \"Hi\" --temp 0.8 --top-k 40 --top-p 0.95");
    {
        SamplerParams sp;
        sp.temp = 0.8f;
        sp.top_k = 40;
        sp.top_p = 0.95f;
        sp.seed = 1234;
        Sampler s(sp);
        const int t = s.sample(logits.data(), static_cast<int>(logits.size()));
        ctx.value("采样结果（真实 llama 的词表里就是某个 token id）", t);
        ctx.check(t >= 0 && t < static_cast<int>(logits.size()), "采样结果落在合法下标范围");
    }
    ctx.check(true, "采样链走完");

    return ctx.summary();
}

// ---------------------------------------------------------------------------
// ③ 自注册：编译进去就出现在 --list，不改任何公共文件
// ---------------------------------------------------------------------------
MINILLAMA_LINE_REGISTER(reg_sample, "sample", "logits → softmax → top-k → top-p → 温度 → 按概率抽样", ::minillama::lines::run_sample);

// ④ 独立 main：只在编译 line_sample 这个单跑目标时打开
#ifdef MINILLAMA_LINE_STANDALONE
MINILLAMA_LINE_STANDALONE_MAIN("sample", ::minillama::lines::run_sample)
#endif
