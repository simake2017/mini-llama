// KV Cache（单元 4）：逐 token 增量推理时缓存历史 K/V，避免每步重算全部历史
//
// 对照真实 llama.cpp:
//   src/llama-kv-cache*.cpp —— 4 种实现（默认、dramatic shift attention、
//   indirect iswa、dsv4），统一接口 llama_memory_i
//
// 本最小版是"朴素顺序版"：
//   - 只支持顺序追加（pos == size），不支持 seq_id / 随机插入 / slot 复用
//   - 不做 UniKV/DSA 这类"缓存压缩"（那是研究前沿，不是主干）
//
// 【概念】为什么 KV cache 是推理的命门：
//   生成第 t 个 token 的注意力需要所有历史位置的 K/V。
//   没有缓存：每步重算全部历史 → O(t²)；有了缓存：每步只算新位置的 K/V，
//   历史只读 → O(t)。LLM 推理的速度几乎完全由这个缓存的带宽决定。

#pragma once

#include <cstdint>
#include <vector>

namespace minillama {

class KVCache {
public:
    // cap = 最大缓存位置数（一般取模型 context_length）
    KVCache(uint32_t n_layer, uint32_t kv_dim, uint32_t cap);

    // 追加一个位置的 K/V（所有层）。调用方保证按 pos 递增顺序。
    void append(uint32_t layer, const float * k, const float * v);

    size_t size() const { return size_; }
    uint32_t dim() const { return dim_; }

    // 读取某层某位置的 K/V 向量
    const float * k(uint32_t layer, size_t pos) const;
    const float * v(uint32_t layer, size_t pos) const;

    // 清空（复用缓存对象开始新对话）
    void clear() { size_ = 0; }

private:
    uint32_t n_layer_;
    uint32_t dim_;
    size_t   cap_;
    size_t   size_ = 0;
    // 布局：k_[((layer*cap) + pos) * dim_ + i]
    // 按 layer 分块：同一层的 K 连续，便于注意力内层循环顺序访问
    std::vector<float> k_;
    std::vector<float> v_;
};

} // namespace minillama
