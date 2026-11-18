cmake_minimum_required(VERSION 3.16)
project(mini-llama CXX)

# 【简化】真实 llama.cpp 要适配几十种编译器/平台，我们只定位 Linux + GCC/Clang + C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

# GGUF 加载库（单元 1）
add_library(minillama_gguf src/gguf.cpp)
target_include_directories(minillama_gguf PUBLIC src)

# 分词器库（单元 2）
add_library(minillama_tok src/unicode.cpp src/tokenizer.cpp)
target_include_directories(minillama_tok PUBLIC src)
target_link_libraries(minillama_tok PUBLIC minillama_gguf)

enable_testing()

# 单元测试：每个单元一个测试可执行文件
add_executable(test_gguf tests/test_gguf.cpp)
target_link_libraries(test_gguf PRIVATE minillama_gguf)
target_include_directories(test_gguf PRIVATE tests)
add_test(NAME gguf COMMAND test_gguf)
# 可选：传入真实 GGUF 文件做冒烟测试（如 llama.cpp 的 vocab 文件）
set_tests_properties(gguf PROPERTIES ENVIRONMENT "TEST_GGUF_REAL_FILE=")

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE minillama_tok)
target_include_directories(test_tokenizer PRIVATE tests)
add_test(NAME tokenizer COMMAND test_tokenizer)
set_tests_properties(tokenizer PROPERTIES ENVIRONMENT "TEST_TOKENIZER_REAL_FILE=")

# 模型库（单元 3 前向；单元 4 KVCache 增量前向 + 采样；单元 5 量化反量化）
add_library(minillama_model src/model.cpp src/kvcache.cpp src/sampler.cpp src/quant.cpp)
target_include_directories(minillama_model PUBLIC src)
target_link_libraries(minillama_model PUBLIC minillama_gguf)

add_executable(test_model tests/test_model.cpp)
target_link_libraries(test_model PRIVATE minillama_model)
target_include_directories(test_model PRIVATE tests)
add_test(NAME model COMMAND test_model)
set_tests_properties(model PROPERTIES ENVIRONMENT "TEST_MODEL_REAL_FILE=")

add_executable(test_generate tests/test_generate.cpp)
target_link_libraries(test_generate PRIVATE minillama_model)
target_include_directories(test_generate PRIVATE tests)
add_test(NAME generate COMMAND test_generate)

# 单元 5 测试：量化反量化 + 混合量化前向
add_executable(test_quant tests/test_quant.cpp)
target_link_libraries(test_quant PRIVATE minillama_model)
target_include_directories(test_quant PRIVATE tests)
add_test(NAME quant COMMAND test_quant)
set_tests_properties(quant PROPERTIES ENVIRONMENT "TEST_QUANT_REAL_FILE=")

# CLI 主程序
add_executable(mini-llama src/main.cpp)
target_link_libraries(mini-llama PRIVATE minillama_model minillama_tok)

# ---------------------------------------------------------------------------
# 业务线（lines/）：按「一条线 = 一个端到端子流程」组织的研究产物
#
# 线的逻辑只在自己的 <线名>_line.cpp；公共 harness 在 lines/_shared/。
# lines/ 里不出现被研究项目的逻辑——要什么能力就回 src/ 加。
# ---------------------------------------------------------------------------
set(MINILLAMA_LINE_DIR ${CMAKE_SOURCE_DIR}/lines)

# 线名 + 源文件（一个线一个 cpp，文件名 <线名>_line.cpp）
# 注意：CMake 里 set() 的引号不保护分号，所以这里用两个平行的列表 + 下标配对
set(MINILLAMA_LINE_NAMES
    model_load
    vocab_load
    tokenize
    detokenize
    prefill
    decode
    dequant
    sample
)
set(MINILLAMA_LINE_FILES
    load/model_load_line.cpp
    load/vocab_load_line.cpp
    text/tokenize_line.cpp
    text/detokenize_line.cpp
    infer/prefill_line.cpp
    infer/decode_line.cpp
    numeric/dequant_line.cpp
    numeric/sample_line.cpp
)

set(MINILLAMA_LINE_SOURCES)
list(LENGTH MINILLAMA_LINE_NAMES MINILLAMA_LINE_COUNT)
math(EXPR MINILLAMA_LINE_LAST "${MINILLAMA_LINE_COUNT} - 1")
foreach(i RANGE ${MINILLAMA_LINE_LAST})
    list(GET MINILLAMA_LINE_NAMES ${i} line_name)
    list(GET MINILLAMA_LINE_FILES ${i} line_file)
    list(APPEND MINILLAMA_LINE_SOURCES ${MINILLAMA_LINE_DIR}/${line_file})
endforeach()

# 公共 harness（不含任何线的逻辑，也没有 main）
add_library(minillama_lines STATIC lines/_shared/line.cpp lines/_shared/lines.cpp)
target_include_directories(minillama_lines PUBLIC lines/_shared)
target_link_libraries(minillama_lines PUBLIC minillama_model minillama_tok)

# 合集：./build/mini-llama-lines [--list | <线名> | all] [--step]
add_executable(mini-llama-lines lines/_shared/main.cpp ${MINILLAMA_LINE_SOURCES})
target_link_libraries(mini-llama-lines PRIVATE minillama_lines)

# 单跑：每条线一个 line_<线名> 目标
foreach(i RANGE ${MINILLAMA_LINE_LAST})
    list(GET MINILLAMA_LINE_NAMES ${i} line_name)
    list(GET MINILLAMA_LINE_FILES ${i} line_file)
    add_executable(line_${line_name} ${MINILLAMA_LINE_DIR}/${line_file})
    target_link_libraries(line_${line_name} PRIVATE minillama_lines)
    target_compile_definitions(line_${line_name} PRIVATE MINILLAMA_LINE_STANDALONE)
endforeach()

# 线测试：quiet 跑一遍 8 条线
add_executable(test_lines tests/test_lines.cpp ${MINILLAMA_LINE_SOURCES})
target_link_libraries(test_lines PRIVATE minillama_lines)

# 默认这一条不挂模型：需要模型的线自动「跳过」（缺外部条件不是线写错了），
# 所以 ctest 保持秒级。用 MINILLAMA_MODEL=... ./build/test_lines 可以手动跑全量。
add_test(NAME lines COMMAND test_lines)
set_tests_properties(lines PROPERTIES ENVIRONMENT "MINILLAMA_MODEL=")

# 有模型文件时额外注册一条重测试：真跑全部 8 条线
# （TinyLlama 的朴素前向约 2~4 秒/token，全套约 1~2 分钟）
if(EXISTS ${CMAKE_SOURCE_DIR}/models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf)
    add_test(NAME lines_model COMMAND test_lines)
    set_tests_properties(lines_model PROPERTIES
        ENVIRONMENT "MINILLAMA_MODEL=${CMAKE_SOURCE_DIR}/models/TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf"
        TIMEOUT 900)
endif()
