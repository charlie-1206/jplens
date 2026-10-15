"""反白模式離線測試：在真實截圖上模擬拖曳，輸出反白範圍的圖與辨識出的文字。

    python tests/eval/select_eval.py 輸出資料夾 [類別 ...]

截圖放在 testdata/<類別>/*.png（不進版控）。每張圖依版面自動產生幾組拖曳：
同一行中間一段、從一行中間拖到下一行中間、整個區塊。結果圖上的編號是閱讀順序。
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jplens.layout import build_layout, refine, select, small_text, zoom_strip  # noqa: E402
from jplens.recognize import Recognizer  # noqa: E402

TD = ROOT / "testdata"


def point_on(ln, frac):
    """行上第 frac 位置的點（行的中線上）。"""
    b = ln.box
    if ln.vertical:
        return (b.x0 + b.x1) / 2, b.y0 + frac * ln.length
    return b.x0 + frac * ln.length, (b.y0 + b.y1) / 2


def drags(lines):
    """挑幾組有代表性的拖曳：最長的行中間一段、多行區塊從第一行拖到最後一行（反向也拖一次）。"""
    if not lines:
        return []
    out = []
    longest = max(range(len(lines)), key=lambda i: lines[i].length)
    out.append(("同一行中間", point_on(lines[longest], 0.25), point_on(lines[longest], 0.7)))
    blocks = {}
    for i, ln in enumerate(lines):
        blocks.setdefault(ln.block, []).append(i)
    multi = [ids for ids in blocks.values() if 2 <= len(ids) <= 6]
    for ids in multi[:2]:
        first, last = lines[ids[0]], lines[ids[-1]]
        out.append((f"區塊 {first.block} 整段", point_on(first, 0.02), point_on(last, 0.98)))
    if multi:
        ids = multi[-1]
        out.append(("反向跨行", point_on(lines[ids[-1]], 0.5), point_on(lines[ids[0]], 0.3)))
    return out


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = sys.argv[2:] or sorted(p.name for p in TD.iterdir() if p.is_dir())
    rec = Recognizer()
    rec.warmup()
    for cat in cats:
        for f in sorted((TD / cat).glob("*.png"))[:4]:
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            t = time.perf_counter()
            lines = build_layout(rec.detect_screen(img), img)
            t_layout = time.perf_counter() - t
            vis = img.copy()
            for i, ln in enumerate(lines):
                b = ln.box
                cv2.rectangle(vis, (b.x0, b.y0), (b.x1, b.y1), (0, 160, 255) if ln.vertical else (255, 120, 0), 1)
                cv2.putText(vis, str(i), (b.x0, max(12, b.y0 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            print(f"\n== {cat}/{f.name}：{len(lines)} 行，版面 {t_layout:.2f} 秒")
            colors = [(0, 200, 0), (200, 0, 200), (0, 0, 255)]
            for n, (label, s, e) in enumerate(drags(lines)):
                use = lines
                # 和 app 相同：起點在橫排小字上就把那一帶放大重新偵測（app 在按下 Alt 時以游標位置做）
                if small_text(lines, (s[0] - 60, s[1] - 30, s[0] + 60, s[1] + 30)):
                    found = zoom_strip(rec.detect_screen, img, lines, s[1])
                    if found:
                        use = found[0]
                        print(f"  （{label}：小字放大重新偵測 {found[1]}）")
                segs = refine(img, use, select(use, s, e))
                t = time.perf_counter()
                text = rec.read_segments(img, use, segs)
                t_read = time.perf_counter() - t
                for sg in segs:
                    x0, y0, x1, y1 = use[sg.line].rect(sg.a, sg.b)
                    ov = vis.copy()
                    cv2.rectangle(ov, (x0, y0), (x1, y1), colors[n % 3], -1)
                    vis = cv2.addWeighted(ov, 0.35, vis, 0.65, 0)
                cv2.circle(vis, (int(s[0]), int(s[1])), 6, colors[n % 3], -1)
                cv2.drawMarker(vis, (int(e[0]), int(e[1])), colors[n % 3], cv2.MARKER_CROSS, 18, 3)
                print(f"  [{label}] 行 {[sg.line for sg in segs]}  {t_read:.2f} 秒：{text}")
            cv2.imencode(".jpg", vis)[1].tofile(str(out_dir / f"{cat}_{f.stem[-6:]}.jpg"))


if __name__ == "__main__":
    main()
