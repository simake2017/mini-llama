// ============================================================================
// ① 定位：采样器实现（单元 4）—— 把一整条 logits 变成一个 token id
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:935:llama_sampler_greedy
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:265:llama_sampler_temp_impl
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:321:llama_sampler_top_k_impl
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:1343:llama_sampler_top_p
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:1040:llama_sampler_dist_apply
//    llama.cpp @ 88b47a755  common/common.h:259  默认 samplers 列表（链的节序）
//    llama.cpp @ 88b47a755  common/sampling.cpp:376  链尾追加 dist
//
// ③ 动机：
//    【概念】推理链的最后一环：模型吐出的是「每个词表项的原始分数」，采样器
//       按策略把它变成一个具体的 token —— 这一步是「模型」与「文本」的接口。
//    【对照】上游每一节是独立对象（llama_sampler_chain 可任意组合），本仓库压成
//       一个固定顺序的函数。**节序必须与上游默认链一致**（top-k → top-p →
//       temperature → dist，common/common.h:259 + common/sampling.cpp:376）：
//       否则同样的 --temp/--top-p 参数会选出不同的候选集合，逐 token 对拍立刻发散。
//    【概念】本文件的节序是「先算一份未加温度的概率 → 用它做两轮截断 → 再加温度
//       → 再抽样」。温度是单调变换，不改变排序，但会改变概率的集中程度——所以
//       截断的依据必须是**加温度之前**的那份分布。
//
// ④ demo + 多方样例：
//    🟢 最小：logits = {3, 2, 1, 0, -1}、temp = 0 → 贪心分支，直接返回 argmax（id 0），
//       完全不碰随机数
//    🟡 典型：同上 logits、temp = 1、top_p = 0.9 → 未加温度的 softmax =
//       {0.6364, 0.2341, 0.0861, 0.0317, 0.0117}，累计 0.6364 / 0.8705 / 0.9567，
//       第 3 个把累计推过 0.9 → 保留前 3 个 → token 3 / 4 一次都不会被抽中
//    🔴 极限：同上 logits、temp = 2、top_p = 0.9 → 截断依据仍是未加温度的分布
//       （累计 0.636 / 0.871 / 0.957 → 仍只留 3 个）；若改成「先加温度再截断」，
//       累计会变成 0.429 / 0.689 / 0.846 / 0.942 → 留 4 个。两者不等价，
//       这就是节序必须锁死的原因（见 lines/numeric/sample_line.cpp 第 6 跳）
//    调用方：src/main.cpp 生成循环每步调一次
// ============================================================================

#include "sampler.h"

#include "model.h"   // softmax

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace minillama {

// ============================================================================
// ① 定位：采样链的唯一入口 —— 一条 logits 进、一个 token id 出
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:935:llama_sampler_greedy
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:265:llama_sampler_temp_impl
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:321:llama_sampler_top_k_impl
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:1343:llama_sampler_top_p
//    llama.cpp @ 88b47a755  src/llama-sampler.cpp:1040:llama_sampler_dist_apply
//
// ③ 动机：
//    【对照】上游默认链的节序是 top-k → top-p → … → temperature → dist
//       （common/common.h:259 的默认 samplers 列表 + common/sampling.cpp:376 追加 dist），
//       即**截断发生在温度生效之前**。本实现照此对齐，节序见下方步骤 1~6。
//    【陷阱】空 logits 必须报错：没有候选时返回 0 会把「无分布」伪装成「选了 0 号
//       token」，调用方会拿着一个凭空来的 id 继续生成。
//    【概念】每一步都在**改写候选集合**：top-k 砍掉尾部、top-p 砍掉核外、
//       温度重塑分布、逆 CDF 抽样挑一个 —— 采样是「逐步收窄」而不是「一次判定」。
//
// ④ demo + 多方样例：
//    🟢 最小：temp = 0 → 贪心分支，返回 argmax，不消耗随机数
//    🟡 典型：temp = 0.8、top_k = 40、top_p = 0.95 → 截断后按概率抽样
//    🔴 极限：n = 0 → 抛「sample: 空 logits」；top_p 极小时 idx 只剩 1 个候选，
//       抽样结果恒定（概率质量全在它身上）
//    调用方：src/main.cpp 生成循环每步调一次
// ============================================================================
int Sampler::sample(const float * logits, int n) {
    if (n <= 0) { throw std::runtime_error("sample: 空 logits"); }

    // 步骤 1：贪心 —— 直接 argmax（temperature/top-k/top-p 都不影响结果）
    // 【对照】上游的 temp<=0 由 llama_sampler_temp_impl（src/llama-sampler.cpp:265）
    //         内部处理：把除最大值外的 logit 全设成 -inf，效果等价于 argmax。
    if (p_.temp <= 0.0f) {
        return static_cast<int>(std::max_element(logits, logits + n) - logits);
    }

    // 步骤 2：未加温度的概率分布 —— top-k / top-p 的截断依据
    // 【动机】温度是单调变换，不改变 token 之间的排序，但会改变概率的「集中程度」；
    //         上游在温度生效前截断，所以这里必须先算一份「原始」概率。
    //         举例：logits={3,2,1,0,-1}、temp=2、top_p=0.9 时
    //         原始分布的累计是 0.636/0.871/0.957 → 保留 3 个；
    //         加温后累计变成 0.429/0.689/0.846/0.942 → 会保留 4 个。两者不等价。
    std::vector<float> probs(logits, logits + n);
    softmax(probs.data(), n);

    // 步骤 3：top-k —— 只留概率最高的 k 个
    // 【性能】n 通常 3~15 万，partial_sort 比全排序快；真实版 ggml 有 top_k 内核
    int keep = n;
    if (p_.top_k > 0 && p_.top_k < n) {
        keep = p_.top_k;
    }

    // 步骤 4：按概率降序排，应用 top-p（核采样）截断
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) { idx[i] = i; }
    std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });
    idx.resize(keep);

    if (p_.top_p > 0.0f && p_.top_p < 1.0f) {
        float cum = 0.0f;
        size_t cut = idx.size();
        for (size_t i = 0; i < idx.size(); i++) {
            cum += probs[idx[i]];
            // 【对照】上游用 `cum_sum >= ctx->p` 且 last_idx = i+1
            //         （src/llama-sampler.cpp:1388，注释「the current iterate should be
            //         included in the set」）——跨过阈值的那一个**要留下**。
            if (cum >= p_.top_p) { cut = i + 1; break; }
        }
        idx.resize(cut);
    }
    if (idx.empty()) { idx.push_back(static_cast<int>(std::max_element(logits, logits + n) - logits)); }

    // 步骤 5：温度作用在存活候选的 logit 上（上游：temperature 在截断之后）
    //    随后在存活集合内重新归一 —— 被裁掉的那部分概率质量就此丢弃。
    std::vector<float> w(idx.size());
    const float inv = 1.0f / p_.temp;
    for (size_t i = 0; i < idx.size(); i++) { w[i] = logits[idx[i]] * inv; }
    softmax(w.data(), static_cast<int>(w.size()));

    // 步骤 6：逆 CDF 抽样 —— 画一个 [0,1) 的随机数，落在谁的累计区间里就选谁
    // 【对照】上游 llama_sampler_dist（src/llama-sampler.cpp:1026）用同样的思路，
    //         差别只在它用 double 累加、本实现用 float。
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    const float draw = uni(rng_);
    float cum = 0.0f;
    for (size_t i = 0; i < idx.size(); i++) {
        cum += w[i];
        if (draw <= cum) { return idx[i]; }
    }
    return idx.back();
}

} // namespace minillama
