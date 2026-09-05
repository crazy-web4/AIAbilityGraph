#!/usr/bin/env python3
"""示例代码健康检查（文档 ↔ examples 一致性回归测试）。

两项检查：
1. 硬错误：examples/ 下所有 .py 文件必须能通过字节码编译（语法错误直接失败）。
2. 引用一致性：docs/ 中引用的 examples/... 路径必须真实存在。
   历史欠账（文档先行、示例待补）默认只报告不失败；加 --strict 时缺失引用也判失败。

用法：
    python3 scripts/check_examples.py            # 编译硬检查 + 缺失引用报告
    python3 scripts/check_examples.py --strict   # 缺失引用也算失败
"""

from __future__ import annotations

import argparse
import py_compile
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
DOCS_DIR = ROOT / "docs"

# 匹配 markdown 中的 examples/ 路径引用（反引号内或注释中）
REF_PATTERN = re.compile(r"examples/[0-9A-Za-z_./-]+")
TRACKED_SUFFIXES = (".py", ".sh", ".json")


def check_python_syntax() -> list[str]:
    """返回编译失败的文件列表。"""
    failures: list[str] = []
    for path in sorted(EXAMPLES_DIR.rglob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True, cfile=tempfile.NamedTemporaryFile(delete=False).name)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc.msg.splitlines()[0] if exc.msg else exc}")
    return failures


def collect_doc_references() -> set[str]:
    """收集 docs/ 下所有 markdown 引用的 examples 路径。"""
    refs: set[str] = set()
    for md in DOCS_DIR.rglob("*.md"):
        for match in REF_PATTERN.findall(md.read_text(encoding="utf-8")):
            clean = match.rstrip(".,;)")
            # 跳过讲解性占位符（如 examples/...、examples/xxx.py）
            if clean.endswith(TRACKED_SUFFIXES) and "..." not in clean and "xxx" not in clean:
                refs.add(clean)
    return refs


def check_references() -> list[str]:
    """返回缺失的引用路径列表。"""
    missing = [ref for ref in sorted(collect_doc_references()) if not (ROOT / ref).exists()]
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="examples 健康检查")
    parser.add_argument("--strict", action="store_true", help="缺失引用也判定为失败")
    args = parser.parse_args()

    syntax_failures = check_python_syntax()
    missing_refs = check_references()

    py_files = list(EXAMPLES_DIR.rglob("*.py"))
    print(f"扫描 {len(py_files)} 个 Python 示例、{len(collect_doc_references())} 个文档引用")

    if syntax_failures:
        print(f"\n❌ 语法错误 {len(syntax_failures)} 个：")
        for item in syntax_failures:
            print(f"  - {item}")
    else:
        print("✅ 所有 Python 示例编译通过")

    if missing_refs:
        by_chapter: dict[str, int] = defaultdict(int)
        for ref in missing_refs:
            by_chapter[Path(ref).parts[1] if len(Path(ref).parts) > 1 else "?"] += 1
        print(f"\n⚠️  文档引用但缺失的示例 {len(missing_refs)} 个（待补示例 backlog）：")
        for chapter, count in sorted(by_chapter.items()):
            print(f"  {chapter}: {count} 个")
        if args.strict:
            print("\n缺失清单：")
            for ref in missing_refs:
                print(f"  - {ref}")
    else:
        print("✅ 文档引用的示例全部存在")

    failed = bool(syntax_failures) or (args.strict and bool(missing_refs))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
