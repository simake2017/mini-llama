// KVCache 实现

#include "kvcache.h"

#include <algorithm>
#include <stdexcept>

namespace minillama {

KVCache::KVCache(uint32_t n_layer, uint32_t kv_dim, uint32_t cap)
    : n_layer_(n_layer), dim_(kv_dim), cap_(cap) {
    if (n_layer_ == 0 || dim_ == 0 || cap_ == 0) {
        throw std::runtime_error("KVCache 参数不能为 0");
    }
    k_.resize(n_layer_ * cap_ * dim_);
    v_.resize(n_layer_ * cap_ * dim_);
}

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

const float * KVCache::v(uint32_t layer, size_t pos) const {
    if (pos >= cap_) { throw std::runtime_error("KVCache 访问越界: v(pos)"); }
    return &v_[(static_cast<size_t>(layer) * cap_ + pos) * dim_];
}

} // namespace minillama
