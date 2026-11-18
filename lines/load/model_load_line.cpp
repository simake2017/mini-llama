// ============================================================================
// 线① 模型载入：--model X.gguf → 解析 GGUF → 映射超参 → 装载权重 → 可前向的 Model
//
// 对照 llama.cpp @ 88b47a755:
//   ggml/src/gguf.cpp          gguf_init_from_file / gguf_read_value / gguf_get_tensor_offset
//   src/llama-model-loader.cpp llama_model_loader::load_all_data / llama_model_loader::get_arr_n
//   src/llama-hparams.h        struct llama_hparams
//   src/llama-arch.cpp         LLM_KV_NAMES（超参 key 的「架构前缀 + 后缀」映射表）
//
// 跨模块：src/gguf.cpp · src/model.cpp · src/quant.cpp（量化张量的块大小校验）
//
// 学习问题：GGUF 里存的是「字符串 key + 字符串张量名」，为什么 llama.cpp 要专门做一层
//           架构前缀映射（LLM_KV_NAMES），而不是把 key 写死成 llama.*？
//
// 跑：MINILLAMA_MODEL=models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf ./build/line_model_load
// ============================================================================

#include "gguf.h"
#include "line.hpp"
#include "lines.hpp"
#include "model.h"

#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace minillama;
using namespace minillama::lines;

namespace {

// 文件是否存在（线要能区分「环境没准备好」和「线写错了」）
bool file_exists(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

// 张量类型 → 出现次数，用来一眼看出「这个模型是量化模型还是 FP 模型」
std::map<std::string, int> type_histogram(const GgufFile & gguf) {
    std::map<std::string, int> hist;
    for (const auto & t : gguf.tensors()) {
        hist[ggml_type_name(t.type)]++;
    }
    return hist;
}

} // namespace

// ---------------------------------------------------------------------------
// 线本体
// ---------------------------------------------------------------------------
int minillama::lines::run_model_load(LineCtx & ctx) {
    const std::string path = resolve_model_path();

    // ------------------------------------------------------------------
    ctx.step_at(1, "打开文件并建立内存映射",
                "src/gguf.cpp: GgufFile::load → open + mmap(MADV_WILLNEED)",
                "MINILLAMA_MODEL=<路径> ./build/line_model_load");
    ctx.value("模型路径", path);
    if (!file_exists(path)) {
        ctx.skip("找不到模型文件：" + path + "（设 MINILLAMA_MODEL 指向一个 GGUF）");
        return ctx.summary();
    }

    std::optional<GgufFile> gguf;
    try {
        gguf.emplace(GgufFile::load(path));
    } catch (const std::exception & e) {
        ctx.check(false, "GGUF 文件应当能正常解析", e.what(), "解析成功");
        return ctx.summary();
    }
    ctx.check(true, "GGUF 文件正常解析");
    ctx.value("GGUF 版本", gguf->version());
    ctx.value("张量数", gguf->tensor_count());
    ctx.value("对齐字节", gguf->alignment());

    // ------------------------------------------------------------------
    ctx.step_at(2, "读文件头：魔数 → 版本 → 张量数 → 元数据条数",
                "src/gguf.cpp: GgufFile::parse 的头部段（对应 gguf_init_from_file 的头解析）",
                "坏魔数应抛异常而不是段错误：tests/test_gguf.cpp 的坏魔数用例");
    ctx.hex("数据区起点", reinterpret_cast<uint64_t>(gguf->data_section()));
    ctx.log("头部字段顺序 = 魔数(4B) 版本(4B) 张量数(8B) 元数据条数(8B)");

    // ------------------------------------------------------------------
    ctx.step_at(3, "读元数据 KV（保序，字符串 key → 强类型值）",
                "src/gguf.cpp: Cursor::read_value 的类型分发（对应 gguf_read_value）",
                "tests/test_gguf.cpp: 标量 / string / array 的合成用例");
    if (const GgufValue * arch = gguf->find("general.architecture")) {
        ctx.value("general.architecture", std::get<std::string>(arch->v));
    }
    if (const GgufValue * name = gguf->find("general.name")) {
        ctx.value("general.name", std::get<std::string>(name->v));
    }
    ctx.check(gguf->find("general.architecture") != nullptr,
              "元数据里应有 general.architecture（后面的超参映射要用它算前缀）");

    // ------------------------------------------------------------------
    ctx.step_at(4, "读张量表：名字 / 维度 / 类型 / 相对偏移",
                "src/gguf.cpp: GgufFile::parse 的张量表段（对应 gguf_get_tensor_offset）",
                "维度是「最快维在前」：ne0 = 输入维");
    const auto & tensors = gguf->tensors();
    for (size_t i = 0; i < tensors.size() && i < 4; i++) {
        ctx.log("张量 " + tensors[i].name +
                " 类型=" + ggml_type_name(tensors[i].type) +
                " 维度={" + std::to_string(tensors[i].dims[0]) + "," +
                (tensors[i].dims.size() > 1 ? std::to_string(tensors[i].dims[1]) : "1") + "}");
    }
    ctx.log("张量表的**顺序不固定**——本模型第一个是 " +
            (tensors.empty() ? std::string("(空)") : tensors[0].name) +
            "，所以查找一律按名字精确匹配");
    ctx.check(gguf->find_tensor("token_embd.weight") != nullptr,
              "张量表里有 token_embd.weight");
    for (const auto & h : type_histogram(*gguf)) {
        ctx.log("类型分布 " + h.first + " × " + std::to_string(h.second));
    }

    // ------------------------------------------------------------------
    ctx.step_at(5, "按 general.alignment 对齐修正张量实际位置",
                "src/gguf.cpp: GgufFile::tensor_data 的对齐修正 + ggml_block_params",
                "tests/test_gguf.cpp: 数据区对齐与量化张量字节数的合成用例");
    ctx.log("对齐是**双重**的：数据区起点绝对对齐，张量之间再各自相对对齐");
    if (!tensors.empty()) {
        const TensorInfo * t = gguf->find_tensor("token_embd.weight");
        if (t) {
            ctx.value("token_embd 字节数", gguf->tensor_nbytes(*t));
            ctx.value("token_embd 原始偏移", t->offset);
        }
    }

    // ------------------------------------------------------------------
    ctx.step_at(6, "超参映射：架构前缀 + 后缀 → 强类型 HParams",
                "src/model.cpp: Model::load 的超参段（对照 llama-arch.cpp 的 LLM_KV_NAMES + llama_hparams）",
                "同一个后缀在不同架构下 key 名不同：llama.block_count / qwen3.block_count");
    Model model;
    try {
        model = Model::load(*gguf);
    } catch (const std::exception & e) {
        ctx.check(false, "模型应当能装载", e.what(), "装载成功");
        return ctx.summary();
    }
    const HParams & hp = model.hparams();
    ctx.value("层数 n_layer", hp.n_layer);
    ctx.value("维度 n_embd", hp.n_embd);
    ctx.value("注意力头 n_head", hp.n_head);
    ctx.value("KV 头 n_head_kv", hp.n_head_kv);
    ctx.value("head_dim", hp.head_dim());
    ctx.value("FFN 宽度 n_ff", hp.n_ff);
    ctx.value("词表 n_vocab", hp.n_vocab);
    ctx.value("rms_eps", hp.rms_eps);
    ctx.value("rope_theta", hp.rope_theta);
    ctx.check(hp.n_layer > 0 && hp.n_embd > 0 && hp.n_head > 0, "超参非零");
    ctx.check(hp.n_embd % hp.n_head == 0, "n_embd 能被 n_head 整除");
    ctx.check(hp.n_head % hp.n_head_kv == 0, "n_head 能被 n_head_kv 整除（GQA 的分组前提）");

    // ------------------------------------------------------------------
    ctx.step_at(7, "装载权重：embedding + 每层 9 张量 + 输出头（tied 回退）",
                "src/model.cpp: Model::load 的权重段（对照 llama_model_loader::load_all_data）",
                "尺寸不符在加载期报错：tests/test_model.cpp 的形状用例");
    ctx.value("embedding 类型", ggml_type_name(model.embedding().type));
    ctx.value("embedding 形状", std::to_string(model.embedding().ne0) + "x" +
                                  std::to_string(model.embedding().ne1));
    ctx.value("输出头 tied", model.tied_output());
    ctx.check(model.embedding().ne1 == hp.n_vocab,
              "embedding 行数 == 词表大小",
              std::to_string(model.embedding().ne1), std::to_string(hp.n_vocab));

    // ------------------------------------------------------------------
    ctx.step_at(8, "结局：Model 就绪，可以直接 forward_token",
                "src/model.cpp: Model::forward_token",
                "./build/mini-llama --model <同一文件> --prompt \"hi\" --n 4");
    ctx.log("模型已就绪——后续 prefill / decode 线都从这一步的产物出发");
    ctx.check(true, "模型载入完成");

    return ctx.summary();
}

// ---------------------------------------------------------------------------
// ③ 自注册：编译进去就出现在 --list，不改任何公共文件
// ---------------------------------------------------------------------------
MINILLAMA_LINE_REGISTER(reg_model_load, "model_load", "GGUF 文件 → 可前向的 Model（头/元数据/张量表 → 超参映射 → 权重装载）", ::minillama::lines::run_model_load);

// ④ 独立 main：只在编译 line_model_load 这个单跑目标时打开
#ifdef MINILLAMA_LINE_STANDALONE
MINILLAMA_LINE_STANDALONE_MAIN("model_load", ::minillama::lines::run_model_load)
#endif
