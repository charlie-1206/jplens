"""打包成 Windows 執行檔：dist/jplens/jplens.exe（一個資料夾，整個複製到別台電腦就能用）。

    python packaging/build.py            # CUDA 版：帶 CUDA／cuDNN（約 3.8 GB，NVIDIA 顯示卡最快）→ dist/jplens/
    python packaging/build.py --no-cuda  # 不帶（小很多，但沒有 CUDA 的電腦只能用 CPU，很慢）
    python packaging/build.py --directml # DirectML 版（約 1.7 GB，任何顯示卡；比 CUDA 慢）→ dist/directml/jplens/
                                         # 要在裝了 onnxruntime-directml（不是 onnxruntime-gpu）的環境執行
    加 --zip：自我測試通過後壓成 dist/jplens-v版本-win64-cuda.zip（或 -directml.zip）＋ .sha256（版本是 jplens.__version__）

需要 pip install pyinstaller。步驟：畫圖示 → PyInstaller（packaging/jplens.spec）→ 複製 CUDA DLL 到 _internal/data/cuda/
→ 放 NOTICE.md、LICENSE、使用說明 → 跑 jplens.exe --selftest 確認模型、字典、斷詞都帶進去 → 清掉自我測試留下的 logs/。
使用者的資料（config.json、vocab/、logs/、captures/）會建在 jplens.exe 旁邊，所以重新打包前先把舊的 dist/jplens 裡的資料備份。
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DIRECTML = "--directml" in sys.argv
DIST = ROOT / "dist" / ("directml" if DIRECTML else "") / "jplens"
WORK = ROOT / "build"
# onnxruntime 的 CUDA 版要的程式庫（CUDA 12、cuDNN 9）
CUDA_DLLS = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cufft64_11.dll", "cudnn64_9.dll",
             "cudnn_adv64_9.dll", "cudnn_cnn64_9.dll", "cudnn_engines_precompiled64_9.dll",
             "cudnn_engines_runtime_compiled64_9.dll", "cudnn_graph64_9.dll", "cudnn_heuristic64_9.dll",
             "cudnn_ops64_9.dll"]

README = """螢幕查字（jplens）

啟動：雙擊 jplens.exe，右下角系統匣出現「字」圖示，等「就緒」的通知後，按住 Alt 用滑鼠選取畫面上的日文。
- 單字本、設定、匯出：在系統匣圖示上按右鍵。
- 遊戲（絕區零、鳴潮、星塔旅人等）以系統管理員身分執行時，查字工具也要用系統管理員身分：
  系統匣選「以系統管理員身分重新啟動」，或在設定 → 一般打開「以系統管理員身分執行」。
{gpu}
- 無字幕模式（直播、沒有字幕的動畫）：按 Alt＋L 開關，把電腦正在播的聲音即時轉成文字；開倍速看影片時在視窗右上選一樣的播放速度。
- 整句翻譯、句子講解要用自己的 Gemini API Key：設定 → 翻譯與 AI。沒有 Key 時其他功能照常。

你的資料都在這個資料夾裡：config.json（設定）、vocab\\（記錄的單字、句子）、cache\\（翻譯過的句子）、
captures\\（保存的畫面、聲音）、logs\\（紀錄檔）。
更新版本：先從系統匣結束查字工具，再擇一：
- 直接覆蓋（最簡單）：新版解壓到「放 jplens 資料夾的那一層」，遇到同名檔案選「取代」。
  你的資料（config.json、vocab\\、cache\\、captures\\）不在壓縮檔裡，不會被蓋掉。有附「更新包」的版本只要下載更新包，做法相同。
- 放到新資料夾：新版解壓到另一個資料夾 → 把舊資料夾的 config.json、vocab\\、cache\\（有用到的話加 captures\\）複製過去
  → 確認單字本都在再刪舊資料夾。有開「開機自動啟動」的話，到 設定 → 一般 關掉再打開一次（它記的是舊位置）。
CUDA 版和 DirectML 版互換時請用「放到新資料夾」（兩版的程式庫不同，蓋在一起會留下用不到的檔案）。
Gemini API Key 存在 Windows 認證管理員，不用重貼。

授權：程式碼 Apache-2.0（LICENSE）；字典、模型等資料的來源與授權見 NOTICE.md；附帶的第三方程式庫（Qt、{lib} 等）見 licenses\ 資料夾。
"""
GPU_TEXT = {
    "cuda": "- 這是 CUDA 版：有 NVIDIA 顯示卡會用顯示卡（快）；沒有會用 CPU（慢很多）。不是 NVIDIA 顯示卡的話請下載 DirectML 版。\n"
            "  設定 → 進階看得到目前用哪個。",
    "directml": "- 這是 DirectML 版：NVIDIA、AMD、Intel 顯示卡都能用（DirectML）；顯示卡比 CPU 還慢（例如較舊的內顯）時自動改用 CPU。\n"
                "  有 NVIDIA 顯示卡的話 CUDA 版比較快（辨識約快 2 倍）。設定 → 進階看得到目前用哪個。",
}


def make_icon(path: Path):
    from PIL import Image
    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtWidgets import QApplication
    from jplens.ui import make_icon as qt_icon
    app = QApplication.instance() or QApplication([])  # noqa: F841
    pm = qt_icon().pixmap(256, 256)
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pm.save(buf, "PNG")
    tmp = path.with_suffix(".png")
    tmp.write_bytes(bytes(buf.data()))
    Image.open(tmp).save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def cuda_source() -> Path:
    from jplens import gpu
    for d in gpu._candidates():
        if d and all((Path(d) / n).exists() for n in CUDA_DLLS):
            return Path(d)
    raise SystemExit("找不到 CUDA 12／cuDNN 9 的 DLL（data/cuda/ 或 torch\\lib）；用 --no-cuda 打包")


def main():
    with_cuda = "--no-cuda" not in sys.argv and not DIRECTML
    if DIRECTML:
        import onnxruntime
        if "DmlExecutionProvider" not in onnxruntime.get_available_providers():
            raise SystemExit("--directml 要在裝了 onnxruntime-directml 的環境執行（目前的 onnxruntime 沒有 DirectML）")
    src = cuda_source() if with_cuda else None
    WORK.mkdir(exist_ok=True)
    make_icon(WORK / "jplens.ico")
    keep = [n for n in ("config.json", "vocab", "logs", "captures") if (DIST / n).exists()]
    if keep:
        raise SystemExit(f"dist/jplens 裡有使用者資料（{'、'.join(keep)}），先備份或移走再打包")
    DIST.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(DIST.parent),
                    "--workpath", str(WORK / "pyi"), str(ROOT / "packaging" / "jplens.spec")], check=True)
    # PyInstaller 的中間檔裡也有一個 jplens.exe，但旁邊沒有 _internal，點了只會出現「Failed to load Python DLL」：刪掉免得點錯
    shutil.rmtree(WORK / "pyi", ignore_errors=True)
    if src:
        dst = DIST / "_internal" / "data" / "cuda"
        dst.mkdir(parents=True, exist_ok=True)
        for n in CUDA_DLLS:
            shutil.copy2(src / n, dst / n)
        print(f"CUDA：從 {src} 複製 {len(CUDA_DLLS)} 個 DLL")
    for n in ("NOTICE.md", "LICENSE"):
        shutil.copy2(ROOT / n, DIST / n)
    # 第三方程式庫的授權（Qt 的 LGPL 全文、NVIDIA、ONNX Runtime；packaging/licenses/README.txt）
    lic = DIST / "licenses"
    shutil.copytree(ROOT / "packaging" / "licenses", lic, dirs_exist_ok=True)
    import onnxruntime
    ort_dir = Path(onnxruntime.__file__).parent
    for n in ("LICENSE", "ThirdPartyNotices.txt"):
        if (ort_dir / n).exists():
            shutil.copy2(ort_dir / n, lic / f"onnxruntime-{n}" if n.endswith(".txt") else lic / "onnxruntime-LICENSE.txt")
    flavor = "directml" if DIRECTML else "cuda"
    readme = README.format(gpu=GPU_TEXT[flavor], lib="Microsoft DirectML" if DIRECTML else "NVIDIA CUDA")
    (DIST / "使用說明.txt").write_text(readme, encoding="utf-8-sig")

    print("自我測試（jplens.exe --selftest）…")
    r = subprocess.run([str(DIST / "jplens.exe"), "--selftest"], timeout=300)
    report = (DIST / "logs" / "selftest.txt").read_text(encoding="utf-8")
    print(report)
    shutil.rmtree(DIST / "logs", ignore_errors=True)
    size = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print(f"大小：{size / 1e9:.2f} GB（{DIST}）")
    if r.returncode != 0:
        raise SystemExit("自我測試失敗")
    if "--zip" in sys.argv:
        make_zip(DIST, ROOT / "dist" / f"jplens-v{__import__('jplens').__version__}-win64-{flavor}.zip")


def make_zip(folder: Path, out: Path):
    """整個資料夾壓成 zip（裡面是 jplens/…，和以前的 Release 一樣），旁邊放 .sha256。"""
    import hashlib
    import zipfile
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, Path(folder.name) / f.relative_to(folder))
    tmp.replace(out)
    h = hashlib.sha256()
    with open(out, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    out.with_name(out.name + ".sha256").write_text(f"{h.hexdigest()}  {out.name}\n", encoding="ascii")
    print(f"壓縮檔：{out}（{out.stat().st_size / 1e9:.2f} GB）")


if __name__ == "__main__":
    main()
