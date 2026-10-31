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
