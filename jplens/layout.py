"""反白模式的版面：螢幕上的文字行、閱讀順序，以及「從起點拖到終點」選到哪些字。

座標都是擷取圖（某一個螢幕）裡的像素。流程：
1. build_layout：偵測框 → 濾掉振假名 → 接回斷開的行 → 決定每行橫排／直排 → 依閱讀順序排列
2. select：起點、終點各自找到所在的行與「沿著行的位置」，中間的行整行選取（像文書軟體選字）
3. refine：切點移到最近的字距空隙，避免切到半個字（反白顯示與辨識用同一組切點）
"""

import os
from dataclasses import dataclass

import numpy as np

from .recognize import Box, _merge_boxes

SQUARISH = 1.5   # 長寬比小於這個值的框看不出方向，參考鄰近的行
RUBY_RATIO = 0.6  # 比旁邊平行的行細這麼多、又緊貼著它 → 振假名
# 偵測框的補丁規則（_resegment_mixed、_split_box、_trim_ruby、_is_ruby）：2026-10-15 起預設關閉，
# 直排小說改用墨水投影（inkcols.py），漫畫用漫畫模式。app 依 config.json 的 layout_rules；
# 環境變數 JPLENS_LAYOUT_RULES=1 也可打開（恢復之前的行為，對照測試用）
RULES = os.environ.get("JPLENS_LAYOUT_RULES") == "1"


@dataclass
class TLine:
    box: Box
    vertical: bool
    block: int = 0  # 所屬區塊（同一段落／同一個對話框）的編號，依閱讀順序
    # 墨水投影找到的直排小說欄才有（inkcols._col_line）：(正文墨水左緣 x, 右緣 x, 振假名帶的右緣 x, 字寬, 背景灰階)，
    # 讀振假名當讀音提示用（ruby.py）
    ruby: tuple | None = None

    @property
    def thickness(self) -> int:
        return self.box.w if self.vertical else self.box.h

    @property
    def length(self) -> int:
        return self.box.h if self.vertical else self.box.w

    def along(self, x, y) -> float:
        """點在這一行上的位置（從行首算起，直排是由上往下、橫排由左往右）。"""
        return (y - self.box.y0) if self.vertical else (x - self.box.x0)

    def distance(self, x, y) -> float:
        b = self.box
        dx = max(b.x0 - x, 0, x - b.x1)
        dy = max(b.y0 - y, 0, y - b.y1)
        return float(np.hypot(dx, dy))

    def rect(self, a: float, b: float) -> tuple[int, int, int, int]:
        """行上 [a, b) 這一段的矩形。"""
        x0, y0, x1, y1 = self.box.to_list()
        a, b = int(round(max(0, a))), int(round(min(self.length, b)))
        return (x0, y0 + a, x1, y0 + b) if self.vertical else (x0 + a, y0, x0 + b, y1)


@dataclass
class Segment:
    line: int     # 在 lines 裡的索引
    a: float      # 沿著行的起點
    b: float      # 沿著行的終點


def _is_ruby(r: Box, boxes: list[Box]) -> bool:
    """r 是否是某一行旁邊的振假名：和更粗的平行行緊貼、且範圍在那一行之內。"""
    for m in boxes:
        if m is r:
            continue
        th = min(m.w, m.h)  # 振假名一定在 m 旁邊一個字寬以內：遠的先跳過（整頁上百個框，大部分都不用細比）
        if r.x1 < m.x0 - th or r.x0 > m.x1 + th or r.y1 < m.y0 - th or r.y0 > m.y1 + th:
            continue
        if m.vertical and r.h > r.w:  # 直排：振假名在右邊（框可能和正文的框重疊）
            if r.w < RUBY_RATIO * m.w and -m.w <= r.x0 - m.x1 <= 0.5 * m.w and r.x1 > m.x1 - 0.2 * m.w \
                    and r.y0 >= m.y0 - r.w and r.y1 <= m.y1 + r.w:
                return True
        elif not m.vertical and r.w > r.h:  # 橫排：振假名在上面
            if r.h < RUBY_RATIO * m.h and -m.h <= m.y0 - r.y1 <= 0.5 * m.h and r.y0 < m.y0 + 0.2 * m.h \
                    and r.x0 >= m.x0 - r.h and r.x1 <= m.x1 + r.h:
                return True
        elif max(r.w, r.h) < SQUARISH * min(r.w, r.h):  # 一兩個字的振假名框接近正方形
            th = m.thickness
            if max(r.w, r.h) < RUBY_RATIO * th:
                if m.vertical and -th <= r.x0 - m.x1 <= 0.5 * th and r.x1 > m.x1 - 0.2 * th \
                        and m.y0 <= r.y0 <= m.y1:
                    return True
                if not m.vertical and -th <= m.y0 - r.y1 <= 0.5 * th and r.y0 < m.y0 + 0.2 * th \
                        and m.x0 <= r.x0 <= m.x1:
                    return True
    return False


def _orientation(b: Box, confident: list[Box]) -> bool:
    """看不出方向的短框（「あ、」「…！」）：跟著附近有明確方向、粗細相近的行；沒有就當橫排。"""
    if b.h > b.w * SQUARISH:
        return True
    if b.w > b.h * SQUARISH:
        return False
    size = max(b.w, b.h)
    near = [c for c in confident
            if abs(c.thickness - size) < 0.5 * size
            and max(c.x0 - b.x1, b.x0 - c.x1, c.y0 - b.y1, b.y0 - c.y1) < 2 * size]
    votes = sum(1 if c.vertical else -1 for c in near)
    return votes > 0


def _adjacent(p: TLine, q: TLine) -> bool:
    """同一個段落／對話框裡相鄰的兩行：同方向、粗細相近、行距小於一個字、而且有並排的部分。"""
    if p.vertical != q.vertical:
        return False
    th = min(p.thickness, q.thickness)
    if abs(p.thickness - q.thickness) > 0.6 * max(p.thickness, q.thickness):
        return False
    a, b = p.box, q.box
    if p.vertical:
        gap = max(a.x0 - b.x1, b.x0 - a.x1)
        overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    else:
        gap = max(a.y0 - b.y1, b.y0 - a.y1)
        overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    return -0.3 * th <= gap <= 1.0 * th and overlap > 0


def _order(lines: list[TLine]) -> list[TLine]:
    """閱讀順序：相鄰的行連成區塊（漫畫對話框裡長短、高低錯落的各欄也會連在一起）；
    區塊內直排由右而左、橫排由上而下；區塊之間大致由上而下，同一高度帶裡直排右邊先、橫排左邊先。"""
    n = len(lines)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    # 相鄰（_adjacent）的判斷一次用 numpy 算整排，連通的結果和逐對比相同
    if n > 1:
        c = np.array([(l.box.x0, l.box.y0, l.box.x1, l.box.y1) for l in lines], np.int64)
        v = np.array([l.vertical for l in lines])
        t = np.array([l.thickness for l in lines], np.int64)
        for i in range(n - 1):
            j = np.arange(i + 1, n)
            th = np.minimum(t[i], t[j])
            ok = (v[j] == v[i]) & (np.abs(t[i] - t[j]) <= 0.6 * np.maximum(t[i], t[j]))
            if v[i]:
                gap = np.maximum(c[i, 0] - c[j, 2], c[j, 0] - c[i, 2])
                ov = np.minimum(c[i, 3], c[j, 3]) - np.maximum(c[i, 1], c[j, 1])
            else:
                gap = np.maximum(c[i, 1] - c[j, 3], c[j, 1] - c[i, 3])
                ov = np.minimum(c[i, 2], c[j, 2]) - np.maximum(c[i, 0], c[j, 0])
            ok &= (gap >= -0.3 * th) & (gap <= 1.0 * th) & (ov > 0)
            for k in j[ok]:
                parent[find(i)] = find(int(k))
    groups: dict[int, list[TLine]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])
    blocks = [sorted(g, key=lambda l: (-l.box.x1, l.box.y0) if l.vertical else (l.box.y0, l.box.x0))
              for g in groups.values()]

    band = 3 * float(np.median([l.thickness for l in lines])) if lines else 1.0

    def key(blk):
        x0 = min(l.box.x0 for l in blk)
        x1 = max(l.box.x1 for l in blk)
        y0 = min(l.box.y0 for l in blk)
        return (int(y0 // band), -x1) if blk[0].vertical else (int(y0 // band), x0)
    out = []
    for k, blk in enumerate(sorted(blocks, key=key)):
        for ln in blk:
            ln.block = k
            out.append(ln)
    return out


def _ink_mask(img: np.ndarray, b: Box) -> np.ndarray:
    crop = img[b.y0:b.y1, b.x0:b.x1]
    g = crop.mean(axis=2) if crop.ndim == 3 else crop.astype(np.float32)
    return np.abs(g - np.median(g)) > 40


def _runs(empty: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """empty（布林陣列）中長度至少 min_len 的連續 True 區段（不含兩端）。"""
    out, start = [], None
    for i, e in enumerate(list(empty) + [False]):
        if e and start is None:
            start = i
        elif not e and start is not None:
            if i - start >= min_len and start > 0 and i < len(empty):
                out.append((start, i))
            start = None
    return out


def _trim_ruby(img: np.ndarray, b: Box, vertical: bool) -> Box:
    """偵測框把振假名一起框進來時（直排在右側、橫排在上方），切掉振假名那一條：
    在靠振假名那一側找一條空白，空白外側的墨水明顯比正文稀疏就切掉。"""
    mask = _ink_mask(img, b)
    counts = mask.sum(axis=0) if vertical else mask.sum(axis=1)
    th = len(counts)
    if th < 8 or counts.max() == 0:
        return b
    empty = counts <= 0.02 * counts.max()
    positions = range(int(0.55 * th), th - 1) if vertical else range(int(0.45 * th), 0, -1)
    for p in positions:
        if not empty[p]:
            continue
        main, band = (counts[:p], counts[p:]) if vertical else (counts[p:], counts[:p])
        if band.max() > 0 and main.mean() > 0 and band.mean() < 0.35 * main.mean():
            return Box(b.x0, b.y0, b.x0 + p, b.y1) if vertical else Box(b.x0, b.y0 + p, b.x1, b.y1)
        break  # 只看最靠近正文的那條空白
    return b


def _clusters(boxes: list[Box], gap: float) -> list[list[Box]]:
    """互相重疊或距離小於 gap 的框分成一組（同一個對話框）。"""
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if a.x0 - gap < b.x1 and b.x0 - gap < a.x1 and a.y0 - gap < b.y1 and b.y0 - gap < a.y1:
                parent[find(i)] = find(j)
    groups: dict[int, list[Box]] = {}
    for i, b in enumerate(boxes):
        groups.setdefault(find(i), []).append(b)
    return list(groups.values())


def _ink_columns(img: np.ndarray, u: Box, typical: float) -> list[Box] | None:
    """依墨水的左右分布把範圍 u 切成直排的欄；比主要的欄細很多的（振假名）不要。切不出合理的欄回傳 None。"""
    mask = _ink_mask(img, u)
    if not mask.any():
        return None
    has = mask.sum(axis=0) > 0
    runs, start = [], None
    for i, v in enumerate(list(has) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    # 字裡的空隙（い、川）不算欄與欄之間：接起來不會比一般的欄寬才接
    # （字小時振假名和正文只隔 1～2 像素，只看空隙大小會把「さそ」接進「誘」那一欄）
    single = max(e - s for s, e in runs)
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 0.15 * typical and r[1] - merged[-1][0] <= 1.15 * single:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    widest = max(e - s for s, e in merged)
    cols = []
    for k, (s, e) in enumerate(merged):
        if e - s < 0.55 * widest:  # 振假名約是正文的一半寬
            continue
        # 直排的振假名在正文右邊、緊貼著比它寬的那一欄（字小時振假名可能有正文的七成寬）
        if k > 0:
            ps, pe = merged[k - 1]
            if s - pe <= 0.3 * widest and e - s < 0.8 * (pe - ps):
                continue
        rows = np.flatnonzero(mask[:, s:e].any(axis=1))
        b = Box(u.x0 + s, u.y0 + int(rows[0]), u.x0 + e, u.y0 + int(rows[-1]) + 1)
        cols.append(_trim_ruby(img, b, True))
    # 切出來的欄太粗 = 欄與欄之間被振假名填滿、好幾欄連在一起（小說整頁），寧可不切
    if not cols or any(c.h < 1.5 * c.w or c.w > 1.6 * typical for c in cols):
        return None
    return cols


def _resegment_mixed(img: np.ndarray, boxes: list[Box], typical: float) -> list[Box]:
    """漫畫對話框：直排的幾欄旁邊夾著振假名時，偵測框常被切碎，而且各欄最上面的字會被框成一條橫的
    （三欄的對話框讀成亂序，其中一段還被當成振假名丟掉）。
    同一組裡同時有直的框和橫的框時，改用墨水分布重新切欄。整頁小說（全是直的）、字幕（全是橫的）不受影響。"""
    out = []
    for g in _clusters(boxes, 0.3 * typical):
        u = g[0]
        for b in g[1:]:
            u = u.union(b)
        tall = [b for b in g if b.h >= 1.5 * b.w]
        # 只處理對話框大小的組：小說整頁的欄很長（30 字以上），頁首的橫排標題碰到欄頂也會被當成「直橫混在一起」
        mixed = (tall and any(b.w >= 1.2 * b.h for b in g) and len(g) >= 2 and u.w <= 12 * typical
                 and u.h <= 12 * float(np.median([b.w for b in tall])))
        if not mixed:
            out += g
            continue
        # 字寬用這一組直框的寬度（漫畫同一頁的字大小差很多，整頁的中位數不準）；
        # 振假名的細框、把兩欄框在一起的粗框都不算
        med = float(np.median([b.w for b in tall]))
        normal = [b for b in tall if 0.6 * med <= b.w <= 1.6 * med] or tall
        cols = _ink_columns(img, u, float(np.median([b.w for b in normal])))
        if cols and _swallows_columns(cols, normal):
            cols = None
        out += cols if cols else g
    return out


def _swallows_columns(cols: list[Box], tall: list[Box]) -> bool:
    """切出來的某一欄裡並排著兩個原本的直框 = 欄與欄之間被振假名填滿、好幾欄連成一欄（小說整頁）。"""
    for c in cols:
        inside = [b for b in tall if c.x0 <= (b.x0 + b.x1) / 2 <= c.x1]
        for i, a in enumerate(inside):
            for b in inside[i + 1:]:
                if min(a.x1, b.x1) - max(a.x0, b.x0) < 0.3 * min(a.w, b.w):
                    return True
    return False


def _split_box(img: np.ndarray, b: Box, typical: float) -> list[Box]:
    """偵測門檻低時，相鄰的兩欄（或兩個對話框裡上下相接的欄）偶爾會被框成一個：
    1. 比一般行粗很多 → 在整條都沒有墨水的地方切成多欄
    2. 沿著行的方向有一大段空白（超過 1.5 個字）→ 切成兩段
    另外，比一般行稍粗的框可能把振假名一起框進來，先切掉。"""
    vertical = b.h > b.w
    if (b.w if vertical else b.h) > 1.15 * typical:
        b = _trim_ruby(img, b, vertical)
    mask = _ink_mask(img, b)
    if not mask.any():
        return [b]
    pieces = [b]
    th = b.w if vertical else b.h
    if th > 1.6 * typical:
        across = mask.any(axis=0) if vertical else mask.any(axis=1)
        cuts = [(s + e) // 2 for s, e in _runs(~across, max(2, int(0.08 * typical)))]
        edges = [0] + cuts + [th]
        pieces = [Box(b.x0 + p, b.y0, b.x0 + q, b.y1) if vertical else Box(b.x0, b.y0 + p, b.x1, b.y0 + q)
                  for p, q in zip(edges, edges[1:]) if q - p >= 0.3 * typical]
    out = []
    for p in pieces:
        m = _ink_mask(img, p)
        if not m.any():
            continue
        along = m.any(axis=1) if vertical else m.any(axis=0)
        idx = np.flatnonzero(along)
        lo, hi = idx[0], idx[-1] + 1  # 縮到有字的範圍
        pth = p.w if vertical else p.h
        cuts = [(s, e) for s, e in _runs(~along[lo:hi], int(1.5 * pth))]
        spans, prev = [], lo
        for s, e in cuts:
            spans.append((prev, lo + s))
            prev = lo + e
        spans.append((prev, hi))
        for s, e in spans:
            pad = max(1, pth // 8)
            if vertical:
                out.append(Box(p.x0, max(b.y0, p.y0 + s - pad), p.x1, min(b.y1, p.y0 + e + pad)))
            else:
                out.append(Box(max(b.x0, p.x0 + s - pad), p.y0, min(b.x1, p.x0 + e + pad), p.y1))
    return out or [b]


def _dedupe(boxes: list[Box]) -> list[Box]:
    """同方向、幾乎在同一條線上又互相重疊的框（同一欄被偵測兩次）合併成一個。
    （和已留下的框一次用 numpy 比，取第一個符合的；結果和逐一比相同，字多的畫面快很多）"""
    boxes = sorted(boxes, key=lambda b: -(b.w * b.h))
    out: list[Box] = []
    ox0, oy0, ox1, oy1 = (np.zeros(len(boxes), np.int64) for _ in range(4))
    for b in boxes:
        n = len(out)
        if n:
            x0, y0, x1, y1 = ox0[:n], oy0[:n], ox1[:n], oy1[:n]
            ow, oh = x1 - x0, y1 - y0
            bv = b.h > b.w
            if bv:
                cross = np.minimum(b.x1, x1) - np.maximum(b.x0, x0)
                along = np.minimum(b.y1, y1) - np.maximum(b.y0, y0)
                th = np.minimum(b.w, ow)
            else:
                cross = np.minimum(b.y1, y1) - np.maximum(b.y0, y0)
                along = np.minimum(b.x1, x1) - np.maximum(b.x0, x0)
                th = np.minimum(b.h, oh)
            hit = np.flatnonzero(((oh > ow) == bv) & (cross >= 0.6 * th) & (along > 0))
            if len(hit):
                k = int(hit[0])
                out[k] = u = out[k].union(b)
                ox0[k], oy0[k], ox1[k], oy1[k] = u.x0, u.y0, u.x1, u.y1
                continue
        out.append(b)
        ox0[n], oy0[n], ox1[n], oy1[n] = b.x0, b.y0, b.x1, b.y1
    return out


def _separate(boxes: list[Box]) -> list[Box]:
    """相鄰兩欄（行）的框互相重疊時（偵測框會往外擴，旁邊有振假名時更寬），在重疊區中間分開，
    否則辨識其中一欄時會讀到另一欄的半個字（「一緒に」被讀兩次）。"""
    boxes = list(boxes)
    # 框只會被切小：一開始就沒有重疊的兩個框之後也不會重疊，不用比（照原本的順序只比一開始有重疊的組合）
    if len(boxes) < 2:
        return boxes
    c = np.array([(b.x0, b.y0, b.x1, b.y1) for b in boxes], np.int64)
    touch = ((np.minimum(c[:, None, 2], c[None, :, 2]) > np.maximum(c[:, None, 0], c[None, :, 0]))
             & (np.minimum(c[:, None, 3], c[None, :, 3]) > np.maximum(c[:, None, 1], c[None, :, 1])))
    np.fill_diagonal(touch, False)
    for i, j in zip(*np.nonzero(touch)):
        a, b = boxes[i], boxes[j]
        if (a.h > a.w) != (b.h > b.w):
            continue
        if a.h > a.w:  # 直排：a 在 b 的左邊、x 有重疊、y 有並排
            if a.x0 < b.x0 < a.x1 < b.x1 and min(a.y1, b.y1) > max(a.y0, b.y0):
                mid = (b.x0 + a.x1) // 2
                boxes[i] = Box(a.x0, a.y0, mid, a.y1)
                boxes[j] = Box(mid, b.y0, b.x1, b.y1)
        else:          # 橫排：a 在 b 的上面
            if a.y0 < b.y0 < a.y1 < b.y1 and min(a.x1, b.x1) > max(a.x0, b.x0):
                mid = (b.y0 + a.y1) // 2
                boxes[i] = Box(a.x0, a.y0, a.x1, mid)
                boxes[j] = Box(b.x0, mid, b.x1, b.y1)
    return boxes


def cross_pad(lines: list["TLine"], k: int, pad: int) -> tuple[int, int]:
    """辨識第 k 行時，垂直於行的方向往兩側各多留幾個像素：預設 pad，但不超過到旁邊那行距離的一半。"""
    ln = lines[k]
    lo = hi = pad
    for m in lines:
        if m is ln or m.vertical != ln.vertical:
            continue
        a, b = ln.box, m.box
        if ln.vertical:
            if min(a.y1, b.y1) <= max(a.y0, b.y0):
                continue
            if b.x1 <= a.x0:
                lo = min(lo, max(0, (a.x0 - b.x1) // 2))
            elif b.x0 >= a.x1:
                hi = min(hi, max(0, (b.x0 - a.x1) // 2))
        else:
            if min(a.x1, b.x1) <= max(a.x0, b.x0):
                continue
            if b.y1 <= a.y0:
                lo = min(lo, max(0, (a.y0 - b.y1) // 2))
            elif b.y0 >= a.y1:
                hi = min(hi, max(0, (b.y0 - a.y1) // 2))
    return lo, hi


def build_layout(boxes: list[Box], img: np.ndarray | None = None,
                 exclude: tuple[int, int, int, int] | None = None) -> list[TLine]:
    """img：截圖（用來拆開被框在一起的多欄）；exclude：要忽略的範圍（例如本程式自己的結果視窗）。"""
    if exclude:
        ex0, ey0, ex1, ey1 = exclude
        boxes = [b for b in boxes if not (b.x0 >= ex0 and b.y0 >= ey0 and b.x1 <= ex1 and b.y1 <= ey1)]
    if img is not None and boxes and RULES:
        main = [b for b in boxes if not _is_ruby(b, boxes)] or boxes  # 一般行的粗細（不算振假名）
        thick = [min(b.w, b.h) for b in main if max(b.w, b.h) >= 2 * min(b.w, b.h)]
        typical = float(np.median(thick)) if thick else float(np.median([min(b.w, b.h) for b in main]))
        boxes = _resegment_mixed(img, boxes, typical)
        # 先拆開再判斷振假名：隔壁欄連振假名被框成一個寬框時，拆開前拿寬框比，正文欄會被當成振假名丟掉
        boxes = [p for b in boxes for p in _split_box(img, b, typical)]
        boxes = [b for b in boxes if not _is_ruby(b, boxes)]
        boxes = _dedupe(boxes)
    elif img is not None:
        boxes = _dedupe(boxes)
    elif RULES:
        boxes = [b for b in boxes if not _is_ruby(b, boxes)]
    boxes = _separate(_merge_boxes(boxes))
    confident = [b for b in boxes if max(b.w, b.h) >= SQUARISH * min(b.w, b.h)]
    return _order([TLine(b, _orientation(b, confident)) for b in boxes])


# ---------------------------------------------------------------- 選取

def hit(lines: list[TLine], x, y, slack: float = 0.35) -> int | None:
    """點在哪一行上（容許行的粗細 slack 倍的誤差）；不在任何一行上回傳 None。"""
    best, best_d = None, None
    for i, ln in enumerate(lines):
        d = ln.distance(x, y)
        if d <= slack * ln.thickness and (best_d is None or d < best_d):
            best, best_d = i, d
    return best


def nearest(lines: list[TLine], x, y) -> int | None:
    if not lines:
        return None
    return min(range(len(lines)), key=lambda i: lines[i].distance(x, y) / max(1, lines[i].thickness))


def select(lines: list[TLine], start, end) -> list[Segment]:
    """從 start 拖到 end（都是 (x, y)）選到的片段，依閱讀順序。起點不在文字上回傳 []。"""
    i = hit(lines, *start)
    if i is None:
        return []
    j = hit(lines, *end)
    if j is None:
        j = nearest(lines, *end)
    a = lines[i].along(*start)
    b = lines[j].along(*end)
    if (j, b) < (i, a):
        i, a, j, b = j, b, i, a
    segs = []
    for k in range(i, j + 1):
        lo = a if k == i else 0
        hi = b if k == j else lines[k].length
        lo, hi = max(0.0, lo), min(float(lines[k].length), hi)
        if hi - lo >= 2:
            segs.append(Segment(k, lo, hi))
    return segs


def box_select(lines: list[TLine], box: tuple[int, int, int, int]) -> list[Segment]:
    """框選：矩形碰到的每一行，取矩形範圍內的那一段（依閱讀順序）。
    只擦到一點邊的行不算（小說直排框選兩個字時，常會擦到隔壁一欄，多讀出隔壁的一個字）。"""
    x0, y0, x1, y1 = box
    segs = []
    for k, ln in enumerate(lines):
        b = ln.box
        ix0, iy0, ix1, iy1 = max(b.x0, x0), max(b.y0, y0), min(b.x1, x1), min(b.y1, y1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        across = (ix1 - ix0) if ln.vertical else (iy1 - iy0)
        if across < 0.4 * ln.thickness:
            continue
        a, e = (iy0 - b.y0, iy1 - b.y0) if ln.vertical else (ix0 - b.x0, ix1 - b.x0)
        if e - a >= 2:
            segs.append(Segment(k, float(a), float(e)))
    return segs


# 同一張圖、同一行的墨水量存起來：反白拖曳時滑鼠每動一下都要 refine，直排小說一整欄每次重算要約 15 ms。
# (圖, {行: 墨水量})；換一張圖整個換掉（主執行緒與背景執行緒都會用，只做整個替換與字典的單一操作）
_profile_cache: tuple = (None, {})


def _ink_profile(img: np.ndarray, ln: TLine) -> np.ndarray:
    """沿著行的「墨水量」：每個位置和背景（中位數）差多少。"""
    global _profile_cache
    owner, memo = _profile_cache
    if owner is not img:
        memo = {}
        _profile_cache = (img, memo)
    b = ln.box
    key = (b.x0, b.y0, b.x1, b.y1, ln.vertical)
    prof = memo.get(key)
    if prof is None:
        crop = img[b.y0:b.y1, b.x0:b.x1]
        g = crop.mean(axis=2) if crop.ndim == 3 else crop.astype(np.float32)
        ink = np.abs(g - np.median(g))
        prof = ink.sum(axis=1) if ln.vertical else ink.sum(axis=0)
        if len(memo) > 500:
            memo.clear()
        memo[key] = prof
    return prof


def _snap(profile: np.ndarray, pos: float, back: float, ahead: float) -> float:
    """把切點移到 [pos - back, pos + ahead] 裡最靠近 pos 的字距空隙（墨水量接近最少的位置）。"""
    lo, hi = int(max(0, pos - back)), int(min(len(profile), pos + ahead + 1))
    if hi - lo < 2:
        return pos
    window = profile[lo:hi]
    low, top = window.min(), window.max()
    quiet = np.flatnonzero(window <= low + 0.1 * (top - low) + 1e-6)
    return float(lo + quiet[np.argmin(np.abs(lo + quiet - pos))])


def refine(img: np.ndarray, lines: list[TLine], segs: list[Segment], box: bool = False) -> list[Segment]:
    """片段兩端（行首行尾以外）移到字距空隙。
    反白：游標所在的那個字一定算進去，起點只往前找空隙、終點只往後找
    （找最近的空隙時，游標在字的後半會漏掉那個字：優しく → しく）。
    框選（box=True）：只算大半在框裡的字，切點移到最近的空隙
    （框邊稍微壓到前一個字時不能把它算進去：框「中止」讀成「を中止」）。"""
    out = []
    for s in segs:
        ln = lines[s.line]
        a, b = s.a, s.b
        if a > 0 or b < ln.length:
            prof = _ink_profile(img, ln)
            t = ln.thickness
            back, ahead = (0.5 * t, 0.5 * t) if box else (1.0 * t, 0.15 * t)
            if a > 0:
                a = _snap(prof, a, back, ahead)
            if b < ln.length:
                b = _snap(prof, b, ahead, back)
        if b - a >= 0.3 * ln.thickness:  # 太短（不到半個字）就不要
            out.append(Segment(s.line, a, b))
    return out


def plan_drag(img: np.ndarray, lines: list[TLine], start, end) -> tuple[list[TLine], list[Segment]]:
    """反白要讀哪些片段：回傳 (行, 片段)。直排小說先用墨水投影找欄（inkcols.plan_drag），
    否則照偵測的版面選（select → refine）。"""
    from . import inkcols  # inkcols 也 import 這個模組
    if inkcols.ENABLED:
        found = inkcols.plan_drag(img, lines, start, end)
        if found is not None:
            return found
    return lines, refine(img, lines, select(lines, start, end))


def plan_box(img: np.ndarray, lines: list[TLine], box: tuple[int, int, int, int], detect=None):
    """框選要怎麼讀：直排小說先用墨水投影找欄（inkcols.plan_ink），找不到才照偵測的版面（_plan_box_det）。
    detect：偵測函式（Recognizer.detect_screen）。框住的是橫排小字時，框附近放大重新偵測（zoom_lines），
    依行讀的結果改成 ("line", (放大偵測的行, 片段))。"""
    from . import inkcols
    if inkcols.ENABLED:
        found = inkcols.plan_ink(img, lines, box)
        if found is not None:
            return "line", found
    if detect is not None and small_text(lines, box):
        zl = zoom_lines(detect, img, box)
        if zl:
            how, arg = _plan_box_det(img, zl, box)
            return ("line", (zl, arg)) if how == "lines" else (how, arg)
    return _plan_box_det(img, lines, box)


# ---------------------------------------------------------------- 小字：放大重新偵測
# 偵測模型對很小（約 14 像素）、行距又密（約 4 像素）的橫排字會失靈：整行漏掉，或抓到好幾行黏在一起的碎框
# （記事本裡的文法清單，tests/eval/real/20261015_215122）。同一塊放大 2 倍再偵測就正常；
# 整個螢幕放大要 1.5 秒，所以只放大選取位置附近。
ZOOM_SMALL = 24     # 碰到的行粗細中位數不超過這麼多像素算小字（正常偵測的 14 像素字約 19～21）
ZOOM_SCALE = 2.0
ZOOM_STRIP = 160    # 反白：以游標為中心、上下各多少像素的整條橫帶放大重新偵測
ZOOM_MARGIN = 200   # 框選：框的左右多放大偵測幾個像素


def small_text(lines: list[TLine], region) -> bool:
    """region 碰到的行是橫排小字：方向明確（長 ≥ 3 倍粗）的行多數是橫排，而且所有碰到的行粗細中位數 ≤ ZOOM_SMALL。
    直排（小說交給墨水投影、漫畫交給漫畫模式）和一般大小的字不放大。"""
    x0, y0, x1, y1 = region
    touched = [ln for ln in lines if ln.box.x0 < x1 and ln.box.x1 > x0 and ln.box.y0 < y1 and ln.box.y1 > y0]
    sure = [ln for ln in touched if ln.length >= 3 * ln.thickness]
    if not sure or sum(not ln.vertical for ln in sure) < 0.6 * len(sure):
        return False
    return float(np.median([ln.thickness for ln in touched])) <= ZOOM_SMALL


def zoom_boxes(detect, img: np.ndarray, crop, scale: float = ZOOM_SCALE) -> list[Box]:
    """crop 範圍放大 scale 倍偵測，框換回原圖座標。"""
    import cv2
    x0, y0, x1, y1 = crop
    part = cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return [Box(x0 + int(b.x0 / scale), y0 + int(b.y0 / scale),
                x0 + int(np.ceil(b.x1 / scale)), y0 + int(np.ceil(b.y1 / scale))) for b in detect(part)]


def zoom_lines(detect, img: np.ndarray, box) -> list[TLine]:
    """框選：框附近（左右多 ZOOM_MARGIN、上下多 1.5 行）放大重新偵測的行（只在這一次框選用，不改整個螢幕的版面）。
    左右只多 2 個字時，框邊附近的字常被漏掉（「例句：両親と」整段沒偵測到），多留 120 像素以上就正常。"""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    crop = (max(0, x0 - ZOOM_MARGIN), max(0, y0 - 30), min(W, x1 + ZOOM_MARGIN), min(H, y1 + 30))
    return build_layout(zoom_boxes(detect, img, crop), img)


def zoom_strip(detect, img: np.ndarray, lines: list[TLine], y: int, exclude=None):
    """反白：以 y 為中心的整條橫帶放大重新偵測，換掉版面裡這一帶的行。回傳 (新的行, (帶的上緣, 下緣))；
    放大後什麼都沒找到回傳 None。碰到橫帶上下邊的框是被切到的行，不要（畫面的上下邊例外）。"""
    H, W = img.shape[:2]
    y0, y1 = max(0, int(y) - ZOOM_STRIP), min(H, int(y) + ZOOM_STRIP)
    boxes = [b for b in zoom_boxes(detect, img, (0, y0, W, y1))
             if (b.y0 > y0 + 1 or y0 == 0) and (b.y1 < y1 - 1 or y1 == H)]
    if not boxes:
        return None
    top, bot = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
    old = [ln.box for ln in lines if not top <= (ln.box.y0 + ln.box.y1) / 2 <= bot]
    return build_layout(old + boxes, img, exclude), (top, bot)


def _plan_box_det(img: np.ndarray, lines: list[TLine], box: tuple[int, int, int, int]):
    """框選要怎麼讀（依偵測框與版面規則）。回傳 ("whole", (裁切範圍, 要塗成背景的範圍))、("lines", 片段)、
    ("line", ([框當成的一行], [片段])) 或 ("block", None)。
    - 框住的是一個對話框大小的範圍（6 行以內、每行約 12 字以內、框不超過約 6×13 個字），而且框到兩行以上
      或框裡偵測不到字：整塊交給 manga-ocr。裁切範圍是「框」加上框碰到的行（行可能比框寬一點，字才不會被切半；
      偵測漏掉的欄也會被框包進去）。漫畫的欄夾著振假名時偵測常漏欄、把兩欄黏成一欄，依行讀會漏字或亂序
    - 框得大（小說、字幕、好幾個對話框）：讀框碰到的每一行落在框裡的那段
    - 框附近完全沒有偵測到字、框又很大：細長一欄（行）的形狀就把框當成一行讀（"line"），
      否則 ("block", None)，交給舊的整塊辨識"""
    x0, y0, x1, y1 = box
    raw = box_select(lines, box)
    # 框得剛好兩三個字大時，框邊可能各切到頭尾的字一點，「大半在框裡」的規則會把字全刪掉：
    # 框確實碰到文字行，就用框的原始範圍
    segs = refine(img, lines, raw, box=True) or raw
    if segs:
        thick = float(np.median([lines[g.line].thickness for g in segs]))
    else:  # 字的大小用附近的行估計
        near = [ln.thickness for ln in lines if ln.box.x1 > x0 - 300 and ln.box.x0 < x1 + 300
                and ln.box.y1 > y0 - 300 and ln.box.y0 < y1 + 300]
        thick = float(np.median(near)) if near else 40.0
    small = (len(segs) <= 6 and all(g.b - g.a <= 12 * thick for g in segs)
             and min(x1 - x0, y1 - y0) <= 6.5 * thick and max(x1 - x0, y1 - y0) <= 13 * thick)
    # 只框到一行的一部分（小說、字幕）時依行讀比較準（整塊讀會把頭尾的字讀錯：生徒会 → 自徒会）；
    # 整塊讀用在框住兩行以上（對話框），或框裡偵測不到字（漏欄、兩欄黏成一欄時，框只擦到那條行的邊）
    if small and len(segs) == 1:
        # 框裡只偵測到一小塊（例如旁邊的振假名「はや」），框住的那一欄卻沒偵測到：也整塊讀
        a, b, c, d = lines[segs[0].line].rect(segs[0].a, segs[0].b)
        if (c - a) * (d - b) >= 0.35 * (x1 - x0) * (y1 - y0):
            small = False
    if small:
        cx0, cy0, cx1, cy1 = x0, y0, x1, y1
        for g in segs:
            a, b, c, d = lines[g.line].rect(g.a, g.b)
            cx0, cy0, cx1, cy1 = min(cx0, a), min(cy0, b), max(cx1, c), max(cy1, d)
        pad = max(2, int(thick * 0.15))
        H, W = img.shape[:2]
        crop = (max(0, cx0 - pad), max(0, cy0 - pad), min(W, cx1 + pad), min(H, cy1 + pad))
        # 框邊切到一半的字要塗掉，manga-ocr 會把半個字猜成別的字（中止 → 中止は）：
        # 框只擦到邊的隔壁行整條塗掉；選到的行，切點以外的部分塗掉。偵測漏掉的欄不受影響，照樣會讀到
        chosen = {g.line: g for g in segs}
        blank = []
        for k, ln in enumerate(lines):
            b = ln.box
            if b.x1 <= crop[0] or b.x0 >= crop[2] or b.y1 <= crop[1] or b.y0 >= crop[3]:
                continue
            g = chosen.get(k)
            if g is None:
                blank.append(b.to_list())
                continue
            if g.a > 0:
                blank.append(list(ln.rect(0, g.a)))
            if g.b < ln.length:
                blank.append(list(ln.rect(g.b, ln.length)))
        return "whole", (crop, blank)
    if segs:
        return "lines", segs
    # 版面漏了這一欄（行），框又是細長一欄（行）的形狀：直接把框當成一行讀。
    # 交給整塊辨識的話，細長的裁切圖重新偵測會把同一欄切成好幾段、又抓到框邊的振假名和半個字，讀出一堆碎字
    w, h = x1 - x0, y1 - y0
    if max(w, h) >= 3 * min(w, h):
        # 字的粗細只看附近同方向、夠長的行（瀏覽器分頁、網址列的小字不算）；附近沒有就用框的短邊
        same = [ln.thickness for ln in lines if ln.vertical == (h > w) and ln.length >= 3 * ln.thickness
                and ln.box.x1 > x0 - 300 and ln.box.x0 < x1 + 300 and ln.box.y1 > y0 - 300 and ln.box.y0 < y1 + 300]
        if min(w, h) <= 1.6 * (float(np.median(same)) if same else min(w, h)):
            ln = TLine(Box(x0, y0, x1, y1), h > w)
            # 框的頭尾切到半個字：切點移到最近的字距空隙（只算大半在框裡的字）
            prof, t = _ink_profile(img, ln), ln.thickness
            a, b = _snap(prof, 0, 0, 0.5 * t), _snap(prof, ln.length, 0.5 * t, 0)
            if b - a < 0.3 * t:
                a, b = 0.0, float(ln.length)
            return "line", ([ln], [Segment(0, a, b)])
    return "block", None

