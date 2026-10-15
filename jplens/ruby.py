"""直排小說的振假名：讀出選到的字旁邊的振假名，連同底下的漢字，當成讀音提示（words.extract 的 ruby）。

人名（小鳥遊、常盤）Sudachi 不認識會被拆碎，常用字也可能選錯讀音（辛い：つらい／からい）；
書上有標振假名的地方，振假名就是正確答案。

只處理墨水投影找到的欄（inkcols._col_line 記下了 TLine.ruby：正文的左右、右邊振假名帶的範圍）：
1. 振假名帶裡有墨水的列，依空白分成一組一組（一組＝一個詞或一個字的振假名）
2. 每組對到正文的字格（inkcols.char_grid）：和這組重疊超過 OVERLAP 格的字就是它標的漢字
3. 漢字：這一欄讀出來的字數和格數相同時，直接取第幾個字（單獨裁一個漢字給 manga-ocr 常讀錯：絡 → 冬音）；
   對不上（直排裡的橫排數字、讀漏字）才單獨裁那幾格讀，而且要在這一欄的文字裡找得到
4. 振假名：每組裁一小塊，一起批次交給 manga-ocr；要全是假名才算數（裁到雜訊、標點的不要）
5. 一個字一組的振假名（小 た、鳥 かな、遊 し）：字格相連的幾組也合起來給一個（小鳥遊 たかなし），
   斷詞時人名要整個詞才對得上
"""

import re

import numpy as np

from .inkcols import INK, char_grid

OVERLAP = 0.4  # 振假名和字格重疊超過這麼多格，才算標在這個字上
GROUP_GAP = 0.3  # 振假名帶裡的空白超過這麼多個字寬就是下一組
PAD = 0.3  # 裁振假名、漢字時四周多留幾個字寬的背景（貼太緊 manga-ocr 容易讀錯）
_KANJI = re.compile(r"[一-鿿々〆ヶ]")
_KANA = re.compile(r"^[ぁ-ゟァ-ヿー]+$")


def _groups(img: np.ndarray, ln, y0: int, y1: int) -> list[tuple[int, int, int, int]]:
    """這一欄 [y0, y1) 範圍內的振假名組：[(x0, y0, x1, y1)]（整張圖的座標）。"""
    s, e, band, W, bg = ln.ruby
    if band - e < 3:
        return []
    H = img.shape[0]
    ya, yb = max(0, int(y0 - 0.5 * W)), min(H, int(y1 + 0.5 * W))  # 標在頭尾字上的振假名可能超出一點
    sub = img[ya:yb, e:band]
    g = sub.mean(axis=2) if sub.ndim == 3 else sub.astype(np.float32)
    mask = np.abs(g - bg) > INK
    rows = mask.any(axis=1)
    runs, start = [], None
    for i, v in enumerate(list(rows) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if runs and start - runs[-1][1] < GROUP_GAP * W:
                runs[-1][1] = i
            else:
                runs.append([start, i])
            start = None
    out = []
    for a, b in runs:
        cy = ya + (a + b) / 2
        if not y0 <= cy < y1 or b - a < 0.25 * W:  # 組的中心要在選到的範圍裡；太小的是雜訊、標點的尾巴
            continue
        cols = np.flatnonzero(mask[a:b].any(axis=0))
        # 振假名比正文小很多：太寬的是別的東西（隔壁欄的字、圖）
        if cols[-1] - cols[0] + 1 > 0.75 * W:
            continue
        out.append((e + int(cols[0]), ya + a, e + int(cols[-1]) + 1, ya + b))
    return out


def _grid(img: np.ndarray, ln):
    """這一欄的字格 (格距, 第一條格線的 y（整張圖的座標）)；估不出來回傳 None。"""
    s, e, band, W, bg = ln.ruby
    b = ln.box
    sub = img[b.y0:b.y1, s:e]
    g = sub.mean(axis=2) if sub.ndim == 3 else sub.astype(np.float32)
    grid = char_grid((np.abs(g - bg) > INK).sum(axis=1), W)
    return None if grid is None else (grid[0], grid[1] + b.y0)


def _cells(grid, gy0, gy1) -> list[int]:
    """和 [gy0, gy1) 重疊超過 OVERLAP 格的字格編號。"""
    P, ph = grid
    first, last = int(np.floor((gy0 - ph) / P)), int(np.floor((gy1 - ph) / P))
    return [i for i in range(first, last + 1) if min(gy1, ph + (i + 1) * P) - max(gy0, ph + i * P) >= OVERLAP * P]


def find(img: np.ndarray, lines, segs) -> list[dict]:
    """選到的片段旁邊的振假名（依閱讀順序）：[{seg: 第幾個片段, cells: (第一格, 最後一格)（從片段開頭算）,
    ncell: 片段的格數, base: 漢字範圍, ruby: 振假名範圍}]。"""
    out = []
    for k, sg in enumerate(segs):
        ln = lines[sg.line]
        if not ln.vertical or ln.ruby is None:
            continue
        x0, y0, x1, y1 = ln.rect(sg.a, sg.b)
        groups = _groups(img, ln, y0, y1)
        grid = _grid(img, ln) if groups else None
        if grid is None:
            continue
        P, ph = grid
        i0 = int(round((y0 - ph) / P))
        n = int(round((y1 - y0) / P))
        e = ln.ruby[1]
        for rb in groups:
            cells = _cells(grid, rb[1], rb[3])
            if not cells:
                continue
            top, bot = ph + cells[0] * P, ph + (cells[-1] + 1) * P
            out.append({"seg": k, "cells": (cells[0] - i0, cells[-1] - i0), "ncell": n,
                        "base": (ln.box.x0, int(round(top)), e + 1, int(round(bot))), "ruby": rb})
    return out


def read(rec, img: np.ndarray, lines, segs, parts: list[str] | None = None) -> list[tuple[str, str]]:
    """讀出振假名與它標的漢字：[(漢字, 平假名讀音)]（長的在前）。rec：Recognizer；
    parts：每個片段讀出來的文字（Recognizer.read_segment_parts），用來對字格取漢字。"""
    from .jlearn import analysis, textutil
    found = find(img, lines, segs)
    if not found:
        return []
    from .recognize import _gray_for_rec

    def crop(r, W):
        """只裁這一塊，四周補 PAD 個字寬的背景色（直接多裁會切到隔壁正文的字邊，振假名讀錯：孤 こ → に）。"""
        import cv2
        x0, y0, x1, y1 = r
        g = _gray_for_rec(img[y0:y1, x0:x1])
        p = max(2, int(PAD * W))
        return cv2.copyMakeBorder(g, p, p, p, p, cv2.BORDER_CONSTANT, value=int(np.median(g)) if g.size else 255)
    # 漢字：字數和格數對得上就直接取；對不上的才裁圖讀
    need = []
    for f in found:
        text = textutil.clean(parts[f["seg"]]) if parts else ""
        a, b = f["cells"]
        if text and len(text) == f["ncell"] and 0 <= a <= b < len(text):
            f["text"] = text[a:b + 1]
        else:
            f["text"] = None
            need.append(f)
    W = {f["seg"]: lines[segs[f["seg"]].line].ruby[3] for f in found}
    crops = [crop(f["ruby"], W[f["seg"]]) for f in found] + [crop(f["base"], W[f["seg"]]) for f in need]
    texts = [textutil.clean(t) for t in rec.rec_batch(crops)]
    for f, t in zip(found, texts):
        f["kana"] = analysis.to_hira(t) if _KANA.match(t) else ""
    for f, t in zip(need, texts[len(found):]):
        whole = textutil.clean(parts[f["seg"]]) if parts else ""
        if not parts or t in whole:  # 單獨讀的漢字要在這一欄的文字裡找得到
            f["text"] = t
    good = [f for f in found if f["text"] and f["kana"] and _KANJI.search(f["text"])
            and len(f["kana"]) <= 5 * len(f["text"])]
    out = []
    # 字格相連的幾組合起來（一個字一組的人名：小 た、鳥 かな、遊 し → 小鳥遊 たかなし）
    i = 0
    while i < len(good):
        j = i + 1
        while (j < len(good) and good[j]["seg"] == good[i]["seg"]
               and good[j]["cells"][0] == good[j - 1]["cells"][1] + 1 and good[j]["text"] and good[j - 1]["text"]):
            j += 1
        if j - i > 1:
            out.append(("".join(f["text"] for f in good[i:j]), "".join(f["kana"] for f in good[i:j])))
        i = j
    out += [(f["text"], f["kana"]) for f in good]
    return list(dict.fromkeys(out))
