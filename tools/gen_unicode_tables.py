#!/usr/bin/env python3
# 生成 src/unicode-tables.h：Unicode 字母/数字类别区间表
#
# 【对照】真实 llama.cpp 的 src/unicode-data.cpp（164K）同样由脚本从
#          UnicodeData.txt 生成。我们用 Python 标准库 unicodedata 现场生成，
#          区间数量小一个数量级 —— 分词器正则只需要 \p{L} 和 \p{N} 两类。
#
# 用法: python3 tools/gen_unicode_tables.py > src/unicode-tables.h

import sys
import unicodedata


def ranges_for(prefixes):
    """把所有 category 以 prefixes 开头的码点压缩成 (start, end) 闭区间列表。"""
    result = []
    start = None
    for cp in range(0x110000):
        cat = unicodedata.category(chr(cp))
        # startswith 接受元组: ("L",) 匹配 Lu/Ll/Lt/Lm/Lo 全部字母类
        hit = cat.startswith(prefixes)
        if hit and start is None:
            start = cp
        elif not hit and start is not None:
            result.append((start, cp - 1))
            start = None
    if start is not None:
        result.append((start, 0x10FFFF))
    return result


def emit(name, prefixes):
    rs = ranges_for(prefixes)
    lines = [f"static const uint32_t {name}[][2] = {{"]
    for i in range(0, len(rs), 4):
        chunk = ", ".join("{%#06x, %#06x}" % r for r in rs[i:i + 4])
        lines.append("    " + chunk + ",")
    lines.append("};")
    lines.append(f"static const size_t {name}_n = sizeof({name}) / sizeof({name}[0]);")
    return "\n".join(lines), len(rs)


def main():
    l_table, l_n = emit("UNICODE_LETTER_RANGES", ("L",))
    n_table, n_n = emit("UNICODE_NUMBER_RANGES", ("N",))
    print("// 本文件由 tools/gen_unicode_tables.py 生成，请勿手改。")
    print("// Unicode %s 数据（Python unicodedata）。" % unicodedata.unidata_version)
    print("#pragma once")
    print("#include <cstddef>")
    print("#include <cstdint>")
    print()
    print(f"// 字母类 L* 共 {l_n} 个区间")
    print(l_table)
    print()
    print(f"// 数字类 N* 共 {n_n} 个区间")
    print(n_table)
    print(f"// 生成统计: L={l_n} 区间, N={n_n} 区间", file=sys.stderr)


if __name__ == "__main__":
    main()
