// ============================================================================
// ① 定位：KVCache 的实现——把 [layer][pos][dim] 的索引公式落成具体的指针运算
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-kv-cache.cpp:64:llama_kv_cache::llama_kv_cache（构造：分配 cell 池）
//    llama.cpp @ 88b47a755  src/llama-kv-cache.cpp:958:llama_kv_cache::find_slot（cell 分配，本实现没有）
//    llama.cpp @ 88b47a755  src/llama-memory.h:73:struct llama_memory_i（4 种实现的统一接口）
//
// ③ 动机：
//    【对照】上游把 K/V 放进两张 ggml 张量、槽位由 cell 池分配；本实现是两块
//       std::vector<float> + 一个线性索引。数值行为一致（同一个位置读到同一个值），
//       丢掉的是「cell 的独立身份」——降级还原、不是等价还原
//       （承重判据见 docs/constraints/04-对齐约束.md §2.2 第 2 条）。
//    【简化】边界校验用容量 cap_ 而不是 size_：逐层推进的中间态里本位置正在被写、
//       size_ 还没涨，用 size_ 校验会误报越界（docs/04-kv-cache-sampling.md §17.2）。
//
// ④ demo + 多方样例：
//    🟢 n_layer = 2、dim = 4、cap = 8 → k_ / v_ 各 64 个 float；k(1, 3) 落在 [44, 48)
//    🟡 一次 forward_token 的写入顺序：第 0 层 append → 第 1 层 append → size_ += 1
//    🔴 越界读 pos >= cap_ → 抛 std::runtime_error（不是返回 nullptr，也不是静默截断）
// ============================================================================

#include "kvcache.h"

#include <algorithm>
#include <stdexcept>

namespace minillama {

// 分配：每层预留 cap_ 个位置、每位置 dim_ 个浮点，K / V 两块缓冲各自连续
KVCache::KVCache(uint32_t n_layer, uint32_t kv_dim, uint32_t cap)
    : n_layer_(n_layer), dim_(kv_dim), cap_(cap) {
    if (n_layer_ == 0 || dim_ == 0 || cap_ == 0) {
        throw std::runtime_error("KVCache 参数不能为 0");
    }
    k_.resize(n_layer_ * cap_ * dim_);
    v_.resize(n_layer_ * cap_ * dim_);
}

// 写入的位置是 size_（本位置在当前层还没写完，所以不是 size_ - 1）
// 【陷阱】size_ 只在最后一层写完才 +1 —— 「本位置已写入」≠「全局 size_」；
//          所以 k() / v() 的边界校验不能用 size_，只能用容量 cap_。
void KVCache::append(uint32_t layer, const float * k, const float * v) {
    if (size_ >= cap_) {
        throw std::runtime_error("KVCache 已满（超过 context length）");
    }
    const size_t base = (static_cast<size_t>(layer) * cap_ + size_) * dim_;
    std::copy(k, k + dim_, &k_[base]);
    std::copy(v, v + dim_, &v_[base]);
    if (layer == n_layer_ - 1) { size_++; }  // 最后一层写完才算一个位置完成
}

// 注意：边界校验用容量而非 size_ —— forward_token 逐层追加时，
// 前 n-1 层的本位置数据已写入但 size_ 要到最后层才递增。
const float * KVCache::k(uint32_t layer, size_t pos) const {
    if (pos >= cap_) { throw std::runtime_error("KVCache 访问越界: k(pos)"); }
    return &k_[(static_cast<size_t>(layer) * cap_ + pos) * dim_];
}

// 与 k() 同一套边界口径（同样用容量 cap_），只是取的是 V 缓冲
const float * KVCache::v(uint32_t layer, size_t pos) const {
    if (pos >= cap_) { throw std::runtime_error("KVCache 访问越界: v(pos)"); }
    return &v_[(static_cast<size_t>(layer) * cap_ + pos) * dim_];
}

} // namespace minillama
