"""把新打包的 dist/jplens 更新到已經在用的資料夾（例如 C:\\jplens），只複製新增或內容不同的檔案。

    python packaging/update_install.py C:\\jplens          # 實際複製
    python packaging/update_install.py C:\\jplens --dry    # 只列出會複製哪些檔案

- 使用者資料（config.json、vocab/、logs/、captures/、cache/）不在 dist 裡，不會被動到。
- 比對方式：沒有的檔案、大小不同的檔案一定複製；大小相同、64 MB 以下的檔案比 SHA-256，不同才複製
  （模型、CUDA 這類大檔重新打包也不會變，大小相同就視為相同，不用每次讀幾 GB）。
- 查字工具開著的時候 jplens.exe 換不掉：先結束再執行。
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
SRC = Path(__file__).resolve().parents[1] / "dist" / "jplens"
HASH_LIMIT = 64 * 2**20


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def running(dst: Path) -> bool:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq jplens.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return "jplens.exe" in out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dst", help="要更新的資料夾（裡面有 jplens.exe）")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    dst = Path(a.dst)
    if not (SRC / "jplens.exe").exists():
        sys.exit(f"找不到 {SRC / 'jplens.exe'}：先執行 python packaging/build.py")
    if not (dst / "jplens.exe").exists():
        sys.exit(f"{dst} 裡沒有 jplens.exe：第一次安裝請直接複製整個 dist/jplens 資料夾")
    if not a.dry and running(dst):
        sys.exit("查字工具還開著（jplens.exe）：從系統匣「結束」後再執行")
    todo = []
    for s in SRC.rglob("*"):
        if not s.is_file():
            continue
        d = dst / s.relative_to(SRC)
        if not d.exists() or d.stat().st_size != s.stat().st_size:
            todo.append((s, d))
        elif s.stat().st_size <= HASH_LIMIT and sha(s) != sha(d):
            todo.append((s, d))
    size = sum(s.stat().st_size for s, _ in todo)
    print(f"要複製 {len(todo)} 個檔案，共 {size / 2**20:.1f} MB")
    for s, d in todo:
        print("  ", s.relative_to(SRC))
        if not a.dry:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    print("（只列出，沒有複製）" if a.dry else "更新完成，可以重新開啟 jplens.exe")


if __name__ == "__main__":
    main()
