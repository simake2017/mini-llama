// ============================================================================
// ① 定位：KV Cache——逐 token 增量推理时缓存历史 K/V，让第 t 步只算新位置而不必重算全部历史
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-kv-cache.h:20:class llama_kv_cache : public llama_memory_i
//    llama.cpp @ 88b47a755  src/llama-memory.h:73:struct llama_memory_i（4 种实现的统一接口）
//    llama.cpp @ 88b47a755  src/llama-kv-cache.cpp:958:llama_kv_cache::find_slot（cell 分配，本实现没有）
//    llama.cpp @ 88b47a755  include/llama.h:785:llama_memory_seq_pos_max（上游是「某序列已占用的
//                          最大位置」，单序列下与本实现的 size() 同义）
//
// ③ 动机：
//    【概念】为什么 KV cache 是推理的命门：生成第 t 个 token 的注意力需要所有历史位置的
//       K/V。没有缓存，每步重算全部历史 → O(t²)；有了缓存，每步只算新位置的 K/V、历史
//       只读 → O(t)。LLM 推理的速度几乎完全由这块缓存的带宽决定。
//    【简化】本实现只保留「纯顺序追加（pos == size）」的单序列缓存：不做 seq_id、不做 slot
//       复用、不做 4 种缓存压缩实现（DSA / ISWA / dsv4）。**这是降级还原、不是等价还原**——
//       cell 身份被砍掉后，「位置」直接当 cell 用，多序列与乱序到达这两个概念就没了
//       （承重判据见 docs/constraints/04-对齐约束.md §2.2 第 2 条）。
//
// ④ demo + 多方样例：
//    🟢 最小用法：KVCache kv(n_layer, kv_dim, n_ctx) → append(l, k, v) → k(l, pos) 读回
//    🟡 典型：forward_token 对 pos = 0..n-1 逐位置推进，kv.size() 从 0 长到 n
//    🔴 容量用尽再 append → 抛「KVCache 已满（超过 context length）」；
//       读 pos >= cap_ → 抛「KVCache 访问越界」
// ============================================================================

#pragma once

#include <cstdint>
#include <vector>

namespace minillama {

// ============================================================================
// ① 定位：KVCache——按 [layer][pos][dim] 存放历史 K/V 的两块连续浮点缓冲
//
// ② 源码锚点：
//    llama.cpp @ 88b47a755  src/llama-kv-cache.h:20:class llama_kv_cache（cell 池 + 两张 ggml 张量）
//    llama.cpp @ 88b47a755  src/llama-kv-cells.h:32:class llama_kv_cells（每个 cell 有独立的 pos 标记）
//
// ③ 动机：
//    【概念】布局是 [layer][pos][dim]（按层分块、同层连续）：同一层的所有位置连着放，
//       注意力内层循环「按位置顺序扫历史」因此是顺序访存，不是跳着访存。
//    【对照】上游把 K/V 拆成两张 ggml 张量、cell 由 find_slot 分配；本实现是一维
//       std::vector<float> + 一个线性索引。差别落在「cell 身份」上：上游的 cell 有独立
//       于 pos 的 id，本实现的 cell 就是 pos（降级还原，见文件头【简化】段）。
//    【简化】寻址用 cap_（容量）而不是「已用位置数」——每个位置都预留满 cap_ 个槽位，
//       换来的是「任意 pos 可直接算出地址、不需要任何分配表」。
//
// ④ demo + 多方样例：
//    🟢 n_layer = 2、dim = 4、cap = 8 → k_ / v_ 各 64 个 float；
//       k(1, 3) 落在 k_ 的 [44, 48)（(1 * 8 + 3) * 4 = 44）
//    🟡 调用方：src/model.cpp 的 forward_token 对第 l 层读 kv.k(l, j)，j = 0..pos
//    🔴 n_layer / kv_dim / cap 任一为 0 → 构造时抛「KVCache 参数不能为 0」
// ============================================================================
class KVCache {
public:
    // cap = 最大缓存位置数（一般取模型 context_length）
    // 三个参数任一为 0 都没有合法布局，所以构造期就拒绝，而不是留一个空壳等到前向期才炸
    KVCache(uint32_t n_layer, uint32_t kv_dim, uint32_t cap);

    // ============================================================================
    // ① 定位：把一个位置的 K/V 写进缓存（调用方按层 0 → n_layer-1 逐层调用）
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  src/llama-kv-cache.cpp:958:llama_kv_cache::find_slot
    //                          → 选定 cell 后把 k / v 写进对应张量（本实现里 cell 就是位置）
    //
    // ③ 动机：
    //    【陷阱】**「本位置已写入」≠「全局 size_」**：size_ 只在最后一层写完才 +1，所以
    //       逐层推进的过程中存在一个中间态 —— 第 0..n-2 层的本位置数据已落盘，size_ 还没
    //       涨。这就是 k() / v() 的边界校验必须用容量 cap_ 而不是 size_ 的原因。
    //    【概念】容量用尽时抛异常而不是静默丢弃：上下文超长是不可恢复的错误，继续跑只会
    //       产出无意义的 logits。
    //
    // ④ demo + 多方样例：
    //    🟢 n_layer = 2：append(0, ...) 之后 size() 仍是 0；append(1, ...) 之后 size() = 1
    //    🟡 一次 forward_token 对 n_layer 层各调一次 append，全程结束时 size_ 恰好 +1
    //       （src/model.cpp）
    //    🔴 size_ >= cap_ 时抛「KVCache 已满（超过 context length）」——判的是 size_
    //       （已完成的位置数），不是「已写入的层数」
    // ============================================================================
    void append(uint32_t layer, const float * k, const float * v);

    // 已完成的位置数（= 已被写满所有层的槽位数），不是「已前向的 token 数」
    size_t size() const { return size_; }
    uint32_t dim() const { return dim_; }

    // ============================================================================
    // ① 定位：读出某层某位置的 K 向量（注意力回读历史用）
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  src/llama-kv-cache.cpp:1605:llama_kv_cache::set_input_k_idxs
    //                          → 按 cell 状态取出要读回的 K 位置（cell 有独立 pos 标记）
    //
    // ③ 动机：
    //    【陷阱】边界校验用**容量 cap_** 而不是 size_：逐层推进的中间态里，本位置的 K/V
    //       正在被写、也马上要被读，此时 size_ 还没涨。用 size_ 校验，第一个位置就会被
    //       误判越界（docs/04-kv-cache-sampling.md §17.2）。「还没写完」由
    //       forward_token 入口的 kv.size() == pos 协议管；边界要防的是「读到分配之外」。
    //
    // ④ demo + 多方样例：
    //    🟢 layer = 0、pos = 0 → 返回 k_ 的首元素地址
    //    🟡 调用方：src/model.cpp 的 forward_token 读 kv.k(l, j)，j = 0..pos
    //    🔴 pos >= cap_ → 抛「KVCache 访问越界: k(pos)」
    // ============================================================================
    const float * k(uint32_t layer, size_t pos) const;

    // ============================================================================
    // ① 定位：读出某层某位置的 V 向量（注意力按权重累加历史 V 时用）
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  src/llama-kv-cache.cpp:1621:llama_kv_cache::set_input_v_idxs
    //                          → 按 cell 状态取出要读回的 V 位置
    //
    // ③ 动机：
    //    【陷阱】与 k() 同一套协议：边界校验同样用容量 cap_ 而不是 size_。逐层推进的中间态
    //       里本位置的 V 正在被写、也马上要被读，此时 size_ 还没涨（理由见上面 k()）。
    //    【对照】索引公式与 k() 完全一致（同一份 [layer][pos][dim] 布局），两块缓冲分开只是
    //       为了写侧与读侧各自连续 —— 与上游「K/V 两张独立张量」的做法同形。
    //
    // ④ demo + 多方样例：
    //    🟢 layer = 0、pos = 0 → 返回 v_ 的首元素地址
    //    🟡 调用方：src/model.cpp 的 forward_token 读 kv.v(l, j) 累加注意力输出
    //    🔴 pos >= cap_ → 抛「KVCache 访问越界: v(pos)」
    // ============================================================================
    const float * v(uint32_t layer, size_t pos) const;

    // ============================================================================
    // ① 定位：清空缓存（复用同一个 KVCache 对象开始新对话）
    //
    // ② 源码锚点：
    //    llama.cpp @ 88b47a755  include/llama.h:733:llama_memory_seq_rm（按 seq 移除；
    //                          本实现只有一条序列，所以等价于「全部清掉」）
    //
    // ③ 动机：
    //    【简化】只清 size_、不清缓冲内容：旧数据会被下一次 append 逐位置覆盖，而「已清空」
    //       这个语义只需要位置数归零。缓冲复用的收益（不重新分配 n_layer * cap * dim 个
    //       float）正是 KVCache 作为可复用对象存在的原因。
    //
    // ④ demo + 多方样例：
    //    🟢 append 若干次后 clear() → size() 归 0
    //    🟡 调用方：lines/infer/prefill_line.cpp 第 6 跳用 clear() 复用同一对象重跑第 0 个位置
    //    🔴 clear() 之后不重新 append 就读 k(0, 0) → 不会抛异常，但内容是上一轮的陈旧值
    // ============================================================================
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
