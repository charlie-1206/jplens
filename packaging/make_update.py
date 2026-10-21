"""產生更新包：舊版的 Release zip 和新打包的資料夾逐檔比對，只放有變、新增的檔案。

    python packaging/make_update.py dist/jplens-v0.3.0-win64.zip dist/jplens
    → dist/jplens-v0.4.0-update-from-v0.3.0.zip（＋ .sha256）

比對用大小＋CRC32（舊版 zip 裡本來就記著每個檔案的 CRC32，不用解壓）。更新包裡照樣是 jplens/…：
使用者結束查字工具後，解壓到「放 jplens 資料夾的那一層」覆蓋，再執行 jplens\\刪除舊檔案.bat（新版不再用的檔案；
留著通常沒事，但 _internal 裡舊版的程式庫可能和新版的衝突）。使用者的資料（config.json、vocab\\…）不在比對範圍、不會動到。
"""

import hashlib
import re
import sys
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")


def crc32(path: Path) -> int:
    c = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            c = zlib.crc32(chunk, c)
    return c


def main(old_zip: str, new_dir: str):
    from jplens import __version__
    old_zip, new_dir = Path(old_zip), Path(new_dir)
    old_ver = re.search(r"v(\d+\.\d+\.\d+)", old_zip.name).group(1)
    top = new_dir.name  # jplens
    with zipfile.ZipFile(old_zip) as z:
        old = {i.filename[len(top) + 1:]: (i.file_size, i.CRC) for i in z.infolist()
               if not i.is_dir() and i.filename.startswith(top + "/")}
    new = {f.relative_to(new_dir).as_posix(): f for f in sorted(new_dir.rglob("*")) if f.is_file()}
    changed, added = [], []
    for rel, f in new.items():
        if rel not in old:
            added.append(rel)
        elif old[rel] != (f.stat().st_size, crc32(f)):
            changed.append(rel)
    removed = sorted(set(old) - set(new))
    out = new_dir.parent / f"jplens-v{__version__}-update-from-v{old_ver}.zip"
    bat = "@echo off\r\nchcp 65001 >nul\r\ncd /d \"%~dp0\"\r\n" + "".join(
        f"if exist \"{r.replace('/', chr(92))}\" del /q \"{r.replace('/', chr(92))}\"\r\n" for r in removed) + \
        f"echo 已刪除 v{old_ver} 用到、v{__version__} 不再用的 {len(removed)} 個檔案。\r\npause\r\n"
    note = (f"jplens v{old_ver} → v{__version__} 更新包（只適用 v{old_ver} 的 CUDA 版，也就是 v{old_ver} 唯一的版本）\r\n\r\n"
            "1. 在系統匣的「字」圖示按右鍵 → 結束。\r\n"
            "2. 把這個壓縮檔解壓到「放 jplens 資料夾的那一層」（裡面是 jplens\\…），遇到同名檔案選「取代」。\r\n"
            "   你的設定、單字本、翻譯快取（config.json、vocab\\、cache\\、captures\\）不在更新包裡，不會被蓋掉。\r\n"
            + (f"3. 執行 jplens\\刪除舊檔案.bat（刪掉新版不再用的 {len(removed)} 個檔案），然後照常開 jplens.exe。\r\n\r\n"
               if removed else "3. 照常開 jplens.exe。\r\n\r\n")
            + f"內容：更新 {len(changed)} 個檔案、新增 {len(added)} 個、刪除 {len(removed)} 個。\r\n\r\n"
            + ("新版不再用的檔案：\r\n" + "".join(f"  {r}\r\n" for r in removed) if removed else ""))
    size = 0
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in changed + added:
            z.write(new[rel], f"{top}/{rel}")
            size += new[rel].stat().st_size
        if removed:
            z.writestr(f"{top}/刪除舊檔案.bat", bat.encode("utf-8"))
        z.writestr(f"{top}/更新說明-v{__version__}.txt", ("﻿" + note).encode("utf-8"))
    tmp.replace(out)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_name(out.name + ".sha256").write_text(f"{h}  {out.name}\n", encoding="ascii")
    print(f"舊版 {len(old)} 個檔案、新版 {len(new)} 個：更新 {len(changed)}、新增 {len(added)}、刪除 {len(removed)}")
    print(f"更新包：{out}（解壓前 {size / 1e9:.2f} GB，壓縮後 {out.stat().st_size / 1e9:.2f} GB）")
    big = sorted(changed + added, key=lambda r: -new[r].stat().st_size)[:15]
    print("最大的檔案：")
    for r in big:
        print(f"  {new[r].stat().st_size / 1e6:8.1f} MB  {'新增' if r in added else '更新'}  {r}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
