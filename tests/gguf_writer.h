// 测试共用：合成 GGUF 字节流构造器（单元 1/2 测试都依赖它）
//
// 与 src/gguf.cpp 的解析器互为逆运算，是 GGUF 格式契约的可执行文档。
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

namespace minitest {

class Writer {
public:
    void u8(uint8_t v)    { raw(&v, 1); }
    void i8(int8_t v)     { raw(&v, 1); }
    void u16(uint16_t v)  { raw(&v, 2); }
    void u32(uint32_t v)  { raw(&v, 4); }
    void i32(int32_t v)   { raw(&v, 4); }
    void u64(uint64_t v)  { raw(&v, 8); }
    void i64(int64_t v)   { raw(&v, 8); }
    void f32(float v)     { raw(&v, 4); }
    void f64(double v)    { raw(&v, 8); }
    void str(const std::string & s) { u64(s.size()); raw(s.data(), s.size()); }
    void raw_bytes(const char * p, size_t n) { raw(p, n); }

    // 元数据 KV 头：key + 值类型编号
    void kv_header(const std::string & key, uint32_t type) {
        str(key);
        u32(type);
    }

    void align(size_t a = 32) {
        while (buf_.size() % a != 0) { buf_.push_back('\0'); }
    }

    // 数据区内对齐：相对数据区起点（GGUF 约定：张量 offset 相对数据区起点累计对齐，
    // 数据区起点本身绝对对齐——gguf_write: offset = prev + PAD(nbytes)）
    void align_data(size_t data_start, size_t a = 32) {
        while ((buf_.size() - data_start) % a != 0) { buf_.push_back('\0'); }
    }

    const std::string & buf() const { return buf_; }

private:
    void raw(const void * p, size_t n) {
        buf_.append(static_cast<const char *>(p), n);
    }
    std::string buf_;
};

// 把字节流写成临时文件，返回路径（调用方负责 remove）
inline std::string write_temp(const std::string & bytes, const char * tag) {
    std::string path = std::string("/tmp/mini_llama_test_") + tag + ".gguf";
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}

} // namespace minitest
