#!/usr/bin/env python3
"""核对 src/ 注释里写的上游锚点是否真实存在（文件:行:符号）。

用法: python3 tools/audit_anchors.py [上游根目录]

解析 src/**/*.h|*.cpp 的注释行，抽出形如
    llama.cpp @ 88b47a755  ggml/src/gguf.cpp:983  gguf_init_from_file
的锚点，逐条到上游 grep 校验「符号在该行附近是否出现」。
只报告可疑项，不改文件。
"""

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = sys.argv[1] if len(sys.argv) > 1 else "/root/ai/llama.cpp"

# 锚点行形如：<path>:<line>  <symbol>   或   <path>:<line>:<symbol>
# 也允许 <path>:<line>（<中文说明>里带 symbol>）这种写法
PAT = re.compile(
    r"(?P<file>[A-Za-z0-9_./-]+\.(?:h|hpp|c|cpp|inc)):"
    r"(?P<line>\d+)\s*[:：]?\s*(?:[（(][^）)]*?)?"
    r"(?P<sym>[A-Za-z_][A-Za-z0-9_]{3,})"
)


def collect():
    hits = []
    for root, _dirs, files in os.walk(os.path.join(REPO, "src")):
        for fn in files:
            if not fn.endswith((".h", ".cpp")):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8") as fh:
                for no, line in enumerate(fh, 1):
                    if not line.lstrip().startswith("//"):
                        continue
                    # 只有带上游版本标记的行才是「上游锚点」；其余是本仓库内部引用
                    if "llama.cpp @" not in line:
                        continue
                    for m in PAT.finditer(line):
                        hits.append((os.path.relpath(path, REPO), no,
                                     m.group("file"), int(m.group("line")),
                                     m.group("sym"), line.strip()))
    return hits


def check(f, ln, sym):
    full = os.path.join(UPSTREAM, f)
    if not os.path.exists(full):
        return "FILE-MISSING"
    try:
        with open(full, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return "READ-FAIL"
    lo = max(0, ln - 1 - 6)
    hi = min(len(lines), ln + 6)
    window = "".join(lines[lo:hi])
    if sym in window:
        return "OK"
    # 全文是否出现过该符号（说明只是行号漂了）
    try:
        out = subprocess.run(["grep", "-rn", "-F", sym, full],
                             capture_output=True, text=True, timeout=20).stdout
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    if not out.strip():
        return "SYMBOL-MISSING"
    first = out.splitlines()[0]
    return "LINE-OFF:" + first


def main():
    hits = collect()
    bad = []
    for repo_file, no, uf, ln, sym, raw in hits:
        st = check(uf, ln, sym)
        if st != "OK":
            bad.append((repo_file, no, uf, ln, sym, st, raw))
    print(f"共 {len(hits)} 条锚点，可疑 {len(bad)} 条")
    for repo_file, no, uf, ln, sym, st, raw in bad:
        print(f"[{st}] {repo_file}:{no} -> {uf}:{ln} {sym}")
        print(f"    {raw[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
