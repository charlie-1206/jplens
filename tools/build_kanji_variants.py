"""產生 jplens/kanji_variants.py：日文新字體 → 繁體字（気 → 氣、発 → 發），單字測驗比對「選項的中文含答案的漢字」用。

    python tools/build_kanji_variants.py

用 OpenCC 的 jp2t 逐字轉（打包版不帶 opencc，所以先轉好寫成程式檔）。只收一對一、轉了會變的字。
"""
import sys
from pathlib import Path

import opencc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "jplens" / "kanji_variants.py"


def main():
    conv = opencc.OpenCC("jp2t")
    pairs = []
    for cp in list(range(0x3400, 0x4DC0)) + list(range(0x4E00, 0xA000)):
        ch = chr(cp)
        t = conv.convert(ch)
        if t != ch and len(t) == 1:
            pairs.append((ch, t))
    jp = "".join(a for a, _ in pairs)
    tw = "".join(b for _, b in pairs)
    lines = [f'"{jp[i:i + 60]}"' for i in range(0, len(jp), 60)]
    lines_tw = [f'"{tw[i:i + 60]}"' for i in range(0, len(tw), 60)]
    OUT.write_text(
        '"""日文新字體 → 繁體字（tools/build_kanji_variants.py 用 OpenCC jp2t 產生，不要手改）。"""\n\n'
        "JP = (" + "\n      ".join(lines) + ")\n"
        "TW = (" + "\n      ".join(lines_tw) + ")\n"
        "TO_TW = dict(zip(JP, TW))\n",
        encoding="utf-8", newline="\n")
    print(f"{len(pairs)} 個字 → {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
