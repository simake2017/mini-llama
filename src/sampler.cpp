// 采样器实现

#include "sampler.h"

#include "model.h"   // softmax

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace minillama {

int Sampler::sample(const float * logits, int n) {
    if (n <= 0) { throw std::runtime_error("sample: 空 logits"); }

    // 贪心：直接 argmax（temperature/top-k/top-p 都不影响结果）
    if (p_.temp <= 0.0f) {
        return static_cast<int>(std::max_element(logits, logits + n) - logits);
    }

    // ① 温度缩放 + softmax 成概率分布
    std::vector<float> probs(logits, logits + n);
    const float inv = 1.0f / p_.temp;
    for (auto & x : probs) { x *= inv; }
    softmax(probs.data(), n);

    // ② top-k：只留概率最高的 k 个
    // 【性能】n 通常 3~15 万，partial_sort 比全排序快；真实版 ggml 有 top_k 内核
    int keep = n;
    if (p_.top_k > 0 && p_.top_k < n) {
        keep = p_.top_k;
    }

    // ③ 按概率降序排，应用 top-p（核采样）截断
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
            if (cum > p_.top_p) { cut = i + 1; break; }  // 包含越界那一个 token
        }
        idx.resize(cut);
    }
    if (idx.empty()) { idx.push_back(static_cast<int>(std::max_element(logits, logits + n) - logits)); }

    // ④ 在截断后的集合内按概率抽样（部分概率质量被裁掉，先归一）
    float total = 0.0f;
    for (int i : idx) { total += probs[i]; }
    std::uniform_real_distribution<float> uni(0.0f, total);
    const float draw = uni(rng_);
    float cum = 0.0f;
    for (int i : idx) {
        cum += probs[i];
        if (draw <= cum) { return i; }
    }
    return idx.back();
}

} // namespace minillama
