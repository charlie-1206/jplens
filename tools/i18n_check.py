"""介面文字的英文化（jplens/i18n.py 的 t()）：找出還沒包成 t() 的中文、收集要翻的字串、檢查英文對照表。

    python tools/i18n_check.py lint [檔案…]     # 列出還沒包 t() 的中文字串（不含 docstring、日誌、print、標了「# i18n: 不翻」的行）
    python tools/i18n_check.py collect           # 收集所有 t("…")、mark("…") 的字串 → i18n_work/strings.json（附出現位置、用途），列出還沒翻的
    python tools/i18n_check.py check             # 檢查 jplens/i18n_en.json：變數 {名稱} 一致、沒有中文殘留、過期的鍵、還沒翻的數量

規則（給改程式的人）：
- 給使用者看的中文包成 t("中文")；有變數的寫 t("已複製「{word}」", word=w)，不要用 f-string 包在 t() 外面。
- 不翻的（和資料比對的字串、存檔格式、日文例句、日誌）留原樣；lint 會列出來，確定不翻的在那一行加「# i18n: 不翻」。
- 文字不是寫死在程式裡、而是從資料來的（詞性、級數名稱）：顯示時 t(變數)，可能的值另外用 mark("名詞", "動詞", …) 列出來讓 collect 收得到。
"""

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
SRC = ROOT / "jplens"
EN = SRC / "i18n_en.json"
WORK = ROOT / "i18n_work"
# 資料、文字處理的模組（Sudachi 詞性、活用、正規表示式、日文、字典的詞性對照）：裡面的中文、日文不是介面文字，lint 不看。
# 其中會顯示的值（詞性名稱）在 words.py 用 mark() 列出。
SKIP_FILES = {"grammar_rules.py", "kanji_variants.py", "i18n.py", "analysis.py", "textutil.py", "grammar.py", "ruby.py",
              "dictionary.py", "words.py"}
CJK = re.compile(r"[぀-ヿ一-鿿]")
IDEO = re.compile(r"[一-鿿]")
FIELD = re.compile(r"\{(\w*)[^{}]*\}")
MARK = "# i18n: 不翻"
# 顯示時的用途：t() 是哪個呼叫的參數（給翻譯的人看長度限制）
SHORT = {"QPushButton", "QToolButton", "QCheckBox", "QRadioButton", "addTab", "_button", "_flat", "QAction",
         "addAction", "setText"}


def files(args, skip=SKIP_FILES) -> list[Path]:
    if args:
        return [Path(a).resolve() for a in args]
    return [p for p in sorted(SRC.rglob("*.py")) if p.name not in skip and "__pycache__" not in p.parts]


def _parents(tree) -> dict:
    parent = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent[id(c)] = n
    return parent


def _docstrings(tree) -> set:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            b = n.body[0]
            if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str):
                out.add(id(b.value))
    return out


def _call_name(call: ast.Call) -> str:
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def _call_base(call: ast.Call) -> str:
    f = call.func
    return getattr(f.value, "id", "") if isinstance(f, ast.Attribute) else ""


def _inside(node, parent, pred) -> ast.AST | None:
    p = parent.get(id(node))
    while p is not None:
        if pred(p):
            return p
        p = parent.get(id(p))
    return None


def _is_log(n) -> bool:
    return isinstance(n, ast.Call) and (_call_base(n) in ("log", "logging", "logger") or _call_name(n) == "print")


def _is_t(n) -> bool:
    return isinstance(n, ast.Call) and _call_name(n) in ("t", "mark")


def lint(paths) -> int:
    total = 0
    for p in paths:
        src = p.read_text(encoding="utf-8")
        lines = src.splitlines()
        tree = ast.parse(src)
        parent, docs = _parents(tree), _docstrings(tree)
        hits = []
        for n in ast.walk(tree):
            if isinstance(n, ast.JoinedStr):
                text = "".join(str(v.value) for v in n.values if isinstance(v, ast.Constant))
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
                if isinstance(parent.get(id(n)), ast.JoinedStr):
                    continue
                text = n.value
            else:
                continue
            if not CJK.search(text) or MARK in lines[n.lineno - 1]:
                continue
            if _inside(n, parent, _is_log) or _inside(n, parent, _is_t):
                continue
            hits.append((n.lineno, text))
        if hits:
            print(f"{p.relative_to(ROOT)}：{len(hits)} 個")
            for ln, text in sorted(hits):
                print(f"  {ln}: {text[:70]!r}")
        total += len(hits)
    print(f"合計 {total} 個還沒包 t() 的中文字串")
    return total


def collect() -> dict:
    """{字串: {"where": ["檔案:行"], "kind": "short"|"text"}}"""
    out = {}
    for p in files([], skip={"i18n.py"}):  # 資料模組裡也有 mark()（words.py 的詞性名稱），收集時不跳過
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src)
        parent = _parents(tree)
        for n in ast.walk(tree):
            if not _is_t(n):
                continue
            args = n.args if _call_name(n) == "mark" else n.args[:1]
            for a in args:
                if not (isinstance(a, ast.Constant) and isinstance(a.value, str)):
                    continue
                d = out.setdefault(a.value, {"where": [], "kind": "text"})
                d["where"].append(f"{p.relative_to(SRC).as_posix()}:{n.lineno}")
                outer = parent.get(id(n))
                if isinstance(outer, ast.Call) and _call_name(outer) in SHORT:
                    d["kind"] = "short"
    return out


def load_en() -> dict:
    try:
        return json.loads(EN.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def cmd_collect():
    keys = collect()
    en = load_en()
    WORK.mkdir(exist_ok=True)
    (WORK / "strings.json").write_text(json.dumps(keys, ensure_ascii=False, indent=1), encoding="utf-8")
    todo = {k: v for k, v in keys.items() if not en.get(k)}
    (WORK / "todo.json").write_text(json.dumps(todo, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"t() 字串 {len(keys)} 個，還沒翻 {len(todo)} 個 → {WORK / 'todo.json'}")


def check() -> int:
    keys, en = collect(), load_en()
    bad = 0
    for k, v in en.items():
        if k not in keys:
            print(f"過期（程式裡已經沒有）：{k!r}")
            continue
        if not isinstance(v, str) or not v.strip():
            continue
        if sorted(FIELD.findall(k)) != sorted(FIELD.findall(v)):
            print(f"變數不一致：{k!r} → {v!r}")
            bad += 1
        left = set(IDEO.findall(v)) - set(IDEO.findall(k))
        if left:
            print(f"英文裡有原文沒有的漢字 {''.join(sorted(left))}：{k!r} → {v!r}")
            bad += 1
    todo = [k for k in keys if not en.get(k)]
    print(f"t() 字串 {len(keys)} 個：已翻 {len(keys) - len(todo)}、還沒翻 {len(todo)}；有問題 {bad} 個")
    return bad


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "lint":
        lint(files(a[1:]))
    elif a[0] == "collect":
        cmd_collect()
    elif a[0] == "check":
        sys.exit(1 if check() else 0)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
