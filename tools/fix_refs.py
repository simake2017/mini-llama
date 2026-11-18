#!/usr/bin/env python3
"""校正文档里对本仓库源码的 `src/xxx:行号` 引用（注释改造后行号会漂移）。

做法：用 git HEAD 里的旧源文件把「旧行号」还原成一个**定义级**符号
（函数定义 / class / struct / enum），再到当前源文件里找该符号的新行号。
只认定义级符号 —— 变量名、注释里的字样一律不认，避免改错。

用法:
    python3 tools/fix_refs.py            # 只报告
    python3 tools/fix_refs.py --apply    # 直接改 docs/ lines/ README.md PROJECT.md
"""

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REF = re.compile(r"(?<![\w/])(?P<file>src/[A-Za-z0-9_]+\.(?:cpp|h)):(?P<line>\d+)")
TYPEDEF = re.compile(r"^\s*(?:class|struct|enum(?: class)?)\s+(?P<sym>[A-Za-z_][A-Za-z0-9_]*)")
FUNCDEF = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_:<>,\s\*&]*?\s+)?"
    r"(?P<sym>[A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*$"
)
KEYWORDS = {"if", "for", "while", "switch", "return", "catch", "sizeof", "else",
            "do", "throw", "delete", "new", "case", "co_return", "operator"}

DOCS = []
TICK = re.compile(r"`([^`]{2,80})`")


def ctx_symbol(text, start, end):
    """取引用附近最近的反引号标识符（优先前面的，其次后面的）。"""
    before = TICK.findall(text[max(0, start - 70):start])
    after = TICK.findall(text[end:end + 70])
    for cand in before[-1:] + after[:1]:
        c = cand.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_:]*", c):
            return c
    return None


for base in ("docs", "lines"):
    for root, _d, files in os.walk(os.path.join(REPO, base)):
        for fn in files:
            if fn.endswith(".md"):
                DOCS.append(os.path.join(root, fn))
DOCS += [os.path.join(REPO, "README.md"), os.path.join(REPO, "PROJECT.md")]


def git_show(path):
    r = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{path}"],
                       capture_output=True, text=True)
    return r.stdout.splitlines() if r.returncode == 0 else None


def cur_lines(path):
    try:
        with open(os.path.join(REPO, path), encoding="utf-8") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def is_def(s):
    """这一行是不是一个定义（函数定义 / 类型定义）？"""
    if not s or s.startswith(("//", "*", "/*", "#", "}")):
        return False
    m = TYPEDEF.match(s)
    if m:
        return True
    m = FUNCDEF.match(s)
    if not m:
        return False
    sym = m.group("sym")
    if sym in KEYWORDS:
        return False
    # 变量初始化（`x = ...`）不算定义
    if re.search(r"\b" + re.escape(sym) + r"\s*=", s):
        return False
    return True


def old_symbol(old, ln):
    """旧文件第 ln 行往上找最近的定义行，返回它的符号名。"""
    if not old or ln > len(old):
        return None
    for i in range(ln - 1, max(-1, ln - 25), -1):
        s = old[i].strip()
        if not s or s.startswith(("//", "*", "/*")):
            continue
        if is_def(s):
            m = TYPEDEF.match(s) or FUNCDEF.match(s)
            return m.group("sym")
    return None


def new_line(lines, sym):
    """在当前源文件里找 sym 的**定义**行（类型定义 / 函数定义优先于调用点）。"""
    base = sym.split("::")[-1]
    code = [(i, l.strip()) for i, l in enumerate(lines, 1)
            if base in l and not l.strip().startswith("//")]
    # 1) 限定名定义：X::method( 或 X::method 的定义行
    if "::" in sym:
        for i, s in code:
            if re.search(re.escape(sym) + r"\s*\(", s):
                return i
        for i, s in code:
            if sym in s:
                return i
    # 2) 类型定义（class / struct / enum）
    for i, s in code:
        if re.search(r"\b(class|struct|enum)\s+" + re.escape(base) + r"\b", s):
            return i
    # 3) 函数定义：形如 `ret name(`（名字前面是类型而不是 . 或 ->）
    for i, s in code:
        m = re.search(r"(^|[\s\*&>])" + re.escape(base) + r"\s*\(", s)
        if m and not re.search(r"[.>]" + re.escape(base) + r"\s*\(", s):
            if not re.search(r"\b" + re.escape(base) + r"\s*=[^=]", s):
                return i
    # 4) 实在找不到定义就退到任意出现处
    for i, s in code:
        if re.search(r"\b" + re.escape(base) + r"\b", s):
            return i
    return None


def main():
    apply = "--apply" in sys.argv
    only = None
    for i, a in enumerate(sys.argv):
        if a == "--file":
            only = sys.argv[i + 1]
    docs = [d for d in DOCS if only is None or os.path.relpath(d, REPO) == only]
    old_cache, cur_cache = {}, {}
    total = changed = skipped = 0
    manual = []
    for doc in docs:
        text = open(doc, encoding="utf-8").read()
        rel = os.path.relpath(doc, REPO)
        out, last, edits = [], 0, []
        for m in REF.finditer(text):
            total += 1
            f = m.group("file")
            cur = int(m.group("line"))
            if f not in old_cache:
                old_cache[f] = git_show(f)
                cur_cache[f] = cur_lines(f)
            sym = old_symbol(old_cache[f], cur)
            if not sym:
                # 退一步：用文档上下文里离引用最近的反引号标识符
                sym = ctx_symbol(text, m.start(), m.end())
                if sym:
                    nl = new_line(cur_cache[f], sym)
                    if nl is not None and nl != cur:
                        changed += 1
                        edits.append((m.group(0), "~" + sym, nl,
                                      cur_cache[f][nl - 1].strip()[:64]))
                        if apply:
                            out.append(text[last:m.start()])
                            out.append(f"{f}:{nl}")
                            last = m.end()
                        continue
            if not sym:
                skipped += 1
                manual.append((rel, m.group(0), "旧行无定义"))
                continue
            nl = new_line(cur_cache[f], sym)
            if nl is None:
                skipped += 1
                manual.append((rel, m.group(0), f"{sym}（新文件里没有）"))
                continue
            if nl == cur:
                continue
            changed += 1
            edits.append((m.group(0), sym, nl, cur_cache[f][nl - 1].strip()[:64]))
            if apply:
                out.append(text[last:m.start()])
                out.append(f"{f}:{nl}")
                last = m.end()
        if edits:
            print(f"\n### {rel}")
            for a, b, c, d in edits:
                print(f"  {a} [{b}] → :{c}   {d}")
            if apply:
                out.append(text[last:])
                open(doc, "w", encoding="utf-8").write("".join(out))
    if manual:
        print(f"\n=== 需人工判断 {len(manual)} 处 ===")
        for r, a, b in manual:
            print(f"  {r}  {a}  {b}")
    print(f"\n共 {total} 处引用；可自动改 {changed} 处，需人工 {skipped} 处" +
          ("（已写入）" if apply else "（未写入，加 --apply 生效）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
