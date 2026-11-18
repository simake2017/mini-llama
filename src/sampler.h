// ============================================================================
// ① 定位：采样器（单元 4）—— 把一整条 logits 变成一个 token id
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:935:llama_sampler_greedy
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:1026:llama_sampler_dist
//    llama.cpp @ 88b47a755  common/common.h:259  默认 samplers 列表（链的节序）
//    llama.cpp @ 88b47a755  common/sampling.cpp:376  链尾追加 dist
//
// ③ 动机：
//    【概念】模型输出的是「每个词表项的原始分数」，采样器是推理链的最后一环：
//       它负责按策略把分数变成一个具体的 token —— 这一步是「模型」与「文本」的接口。
//    【对照】上游每一节是独立对象（llama_sampler_chain 可任意组合）；本仓库把
//       固定顺序压成一个函数。**节序不能随便改**：上游默认链是
//       top-k → top-p → temperature → dist（common/common.h:259 的列表顺序 +
//       common/sampling.cpp:376 在链尾追加 dist），即**截断发生在温度生效之前**。
//       顺序不同，同样的 --temp/--top-p 参数会选出不同的候选集合，逐 token 对拍立刻发散。
//    【简化】只实现最常用的四节（greedy / top-k / top-p / temperature + 抽样）；
//       上游链上还有 penalties / dry / typical-p / min-p / xtc / mirostat 等十余节，
//       承重判据见 docs/constraints/04-对齐约束.md（结论：边缘加固，可砍）。
//
// ④ demo + 多方样例：
//    🟢 SamplerParams{0.0f, 1, 1.0f, 0} + sample(logits, n) → 纯 argmax，不碰随机数
//    🟡 SamplerParams{0.8f, 40, 0.95f, 1234} → 温度 + top-k + top-p（同种子可复现）
//    🔴 SamplerParams{1.2f, 100, 0.98f, 随机种子} → 开放生成，多样性优先
//    调用方：src/main.cpp 构造采样器并跑生成循环、lines/numeric/sample_line.cpp
// ============================================================================

#pragma once

#include <cstdint>
#include <random>

namespace minillama {

// ============================================================================
// ① 定位：采样参数——四个旋钮决定「分布被裁多狠、温度多高、随机数从哪来」
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  common/common.h:227  min_keep（上游每一节都有最小保留数）
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:265:llama_sampler_temp_impl
//
// ③ 动机：
//    【概念】采样不是「掷骰子」这么简单，它是一串**逐节修改候选分布**的操作：
//       每个旋钮对应链上的一节，参数就是这节的输入。
//    【对照】上游把每一节拆成独立对象、各自带自己的参数与 min_keep 下限；
//       本仓库合并成一个结构体 —— 少了「任意组合」的自由度，但主干语义相同。
//    【陷阱】top_p 的判据是「累积概率达 p 的**最小**集合」，所以 p 越接近 1
//       留得越多、p 越小留得越少；top_k 与 top_p 是**两道串联的截断**，
//       先到的那道先生效（见 sample() 的 ③④ 两步）。
//
// ④ demo + 多方样例：
//    🟢 最小（调试/复现）：temp=0 纯贪心
//       SamplerParams{0.0f, 1, 1.0f, 0}
//    🟡 典型（聊天）：温度 + top-k + top-p
//       SamplerParams{0.8f, 40, 0.95f, 1234}
//    🔴 极限（开放生成）：高温 + 大 top-p，多样性优先
//       SamplerParams{1.2f, 100, 0.98f, 随机种子}
//    调用方：src/main.cpp 声明、:63~:67 由命令行参数填充
// ============================================================================
struct SamplerParams {
    float    temp    = 0.8f;   // <= 0 时退化为贪心
    int      top_k   = 40;     // 只保留概率最高的 k 个（<=0 表示不限制）
    float    top_p   = 0.95f;  // 只保留累积概率达 p 的最小集合（<=0 表示不限制）
    uint32_t seed    = 1234;
};

// ============================================================================
// ① 定位：采样器本体——持有参数 + 随机数发生器，把 logits 变成 token id
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:1040:llama_sampler_dist_apply
//                          （上游 dist 节的逆 CDF 抽样，本实现同思路）
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:265:llama_sampler_temp_impl
//
// ③ 动机：
//    【概念】采样器是**有状态**的：随机数发生器在成员里，连续调用 sample 会
//       顺序消费随机数。所以「同一个 seed」不等于「同一串输出」——还得从同一
//       状态开始（reset_seed 复位）。
//    【对照】上游的 dist 节用 double 累加概率，本实现用 float：数值分布形状
//       一致，边界 token 的命中概率会有 1e-7 量级差异，对拍 logits 时不构成问题。
//    【陷阱】贪心分支（temp<=0）**完全不碰随机数** —— 所以「采样结果可复现」
//       在贪心下是无条件成立的，不能拿来验证 seed 是否生效。
//
// ④ demo + 多方样例：
//    🟢 构造 Sampler({0.8f, 40, 0.95f, 1234}) → sample 一条 logits 得到一个 id
//    🟡 同一个 seed 连续构造两个 Sampler，各 sample 一次 → 结果相同
//    🔴 reset_seed(1234) 之后再次 sample → 回到第一次的结果
//    调用方：src/main.cpp 生成循环每步调一次 sample
// ============================================================================
class Sampler {
public:
    explicit Sampler(SamplerParams params) : p_(params), rng_(params.seed) {}

    // logits: 长度 n 的原始输出（未归一化）。返回选中的 token id。
    // n <= 0 时抛异常 —— 空分布没有可选项，静默返回 0 会把「没有候选」伪装成「选了 0」。
    int sample(const float * logits, int n);

    // 固定种子可复现（同一个 Sampler 连续调用是顺序消费随机数的）
    void reset_seed(uint32_t seed) { rng_.seed(seed); }

private:
    SamplerParams p_;
    std::mt19937 rng_;
};

} // namespace minillama
