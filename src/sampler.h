// 采样器（单元 4）：从 logits 中按策略挑选下一个 token
//
// 对照真实 llama.cpp:
//   src/llama-sampler.cpp —— 链式采样器（chain），温度/top-k/top-p/min-p/
//   重复惩罚/grammar 约束等十几节，逐节变换 logits 或挑选 token
//
// 本最小版实现最常用的四节：
//   greedy（temp=0）-> temperature -> top-k -> top-p -> 按概率抽样
// 【概念】链式设计：真实版每一节是独立对象，可任意组合（llama_sampler_chain）；
//         我们把固定顺序写成一个函数，主干逻辑（裁剪分布 + 抽样）相同。

#pragma once

#include <cstdint>
#include <random>

namespace minillama {

struct SamplerParams {
    // 🟢 最小（调试/复现）：temp=0 纯贪心
    //    SamplerParams{0.0f, 1, 1.0f, 0}
    // 🟡 典型（聊天）：温度 + top-k + top-p
    //    SamplerParams{0.8f, 40, 0.95f, 1234}
    // 🔴 极限（开放生成）：高温 + 大 top-p，多样性优先
    //    SamplerParams{1.2f, 100, 0.98f, 随机种子}
    float    temp    = 0.8f;   // <= 0 时退化为贪心
    int      top_k   = 40;     // 只保留概率最高的 k 个（<=0 表示不限制）
    float    top_p   = 0.95f;  // 只保留累积概率达 p 的最小集合（<=0 表示不限制）
    uint32_t seed    = 1234;
};

class Sampler {
public:
    explicit Sampler(SamplerParams params) : p_(params), rng_(params.seed) {}

    // logits: 长度 n 的原始输出（未归一化）。返回选中的 token id。
    int sample(const float * logits, int n);

    // 固定种子可复现（同一个 Sampler 连续调用是顺序消费随机数的）
    void reset_seed(uint32_t seed) { rng_.seed(seed); }

private:
    SamplerParams p_;
    std::mt19937 rng_;
};

} // namespace minillama
