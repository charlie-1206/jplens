"""直排小說的框選與反白：不靠偵測框與版面規則，用墨水投影找欄。

預設開啟（app 依 config.json 的 ink_columns；環境變數 JPLENS_INK=0 可關掉，對照測試用）：
- 框選：layout.plan_box 先試 plan_ink（小說兩三欄的小框也是：整塊讀會讀錯字、多出幻覺的字）
- 反白：layout.plan_drag 先試 plan_drag
找不到合理的直排欄（橫排、背景不是單色、圖裡的線條、看不出直排）就回傳 None，照舊用偵測的版面。
漫畫對話框旁邊有圖、欄很短，幾乎都會被擋下來照舊整塊讀（漫畫請用漫畫模式）。

找欄：框選範圍（上下至少看 SPAN、左右各多看 1.5 個字寬）的墨水做垂直投影（每個 x 有幾個墨水像素）：
1. 正文欄的投影是一條又高又寬的峰，振假名是緊貼在右邊、又窄又低的峰，中間通常有一兩個像素幾乎沒有墨水
2. 字寬 W＝夠高的峰的寬度中位數；比 W 窄很多、貼在欄右邊、墨水少很多的峰是振假名，不要
3. 頁首的橫排標題（頁碼、書名）和正文之間隔著一段空白的列：先只留夠高的列帶再投影
4. 每欄的行框是墨水範圍左右各多 PAD 個字寬
切點：直排每個字佔一格，切點移到字格的格線（char_grid；估不出來的欄照舊用 layout.refine 找字距空隙）。
回傳 ([每欄一行], [片段])，交給 recognize.read_segments。
"""

import os

import numpy as np

from .layout import Segment, TLine, refine
from .recognize import Box

INK = 40  # 和背景（中位數）差這麼多才算墨水，同 layout._ink_mask
FLAT = 12  # 和背景差這麼少算「背景」：小說的範圍裡至少六成要是背景
MARGIN = 200  # 找欄時框的左右最多多看幾個像素
BAR = 0.6  # 反白找欄：畫面上下緣一整列超過這個比例和背景不同的橫條（Windows 工作列）不算在找欄範圍裡
ENABLED = os.environ.get("JPLENS_INK", "1") != "0"
PAD = 0.25  # 每欄左右各多留幾個字寬（見 plan_ink）
MORE = 200  # 反白：找欄時拖曳範圍左右多找幾個像素（存起來，見 _drag_columns）
EDGE = 0.2  # 反白：游標離字格格線不到這麼多格時，算成在格線上（見 plan_drag）
SPAN = 400  # 找欄時至少看這麼高的範圍（小說的字 20～60 像素，約 7～20 個字）
RUBY_BAND = 0.8  # 振假名帶：正文右緣往右最多幾個字寬


def _runs(flags: np.ndarray) -> list[list[int]]:
    out, start = [], None
    for i, v in enumerate(list(flags) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start, i])
            start = None
    return out


def _split_valleys(cnt: np.ndarray, s: int, e: int) -> list[list[int]]:
    """一段連續有墨水的範圍，在幾乎沒有墨水的谷底（≤ 峰的 3%）切開：振假名和正文只隔一兩個像素時，
    中間常有反鋸齒的淡墨，不會是 0。"""
    low = cnt[s:e] <= max(1, 0.03 * cnt[s:e].max())
    return [[s + a, s + b] for a, b in _runs(~low)]


def _body_rows(mask: np.ndarray, w: float) -> np.ndarray:
    """只留夠高的列帶（直排正文）：頁首的頁碼、書名是一列橫排小字，和正文之間隔著一段空白。"""
    rows = mask.any(axis=1)
    bands = _runs(rows)
    # 正文裡字與字的空隙不到半個字；段落結束只是某一欄變短，其他欄還有字，整列不會空
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < 0.6 * w:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    tall = [b for b in merged if b[1] - b[0] >= 2.5 * w]
    if not tall:
        return mask
    keep = np.zeros(len(rows), bool)
    for s, e in tall:
        keep[s:e] = True
    return mask & keep[:, None]


def _width_guess(cnt: np.ndarray) -> float | None:
    """粗估字寬：投影超過最高峰一成的範圍（正文欄）的寬度中位數。橫排標題、振假名的投影都很低，不影響。"""
    if cnt.max() == 0:
        return None
    core = [e - s for s, e in _runs(cnt >= 0.1 * cnt.max()) if e - s >= 3]
    return float(np.median(core)) if core else None


def find_columns(img: np.ndarray, box: tuple[int, int, int, int]):
    """框選範圍附近的直排欄（左右的 x 範圍，整張圖的座標，由左而右）、字寬、背景灰階；找不到合理的欄回傳 None。"""
    H, Wd = img.shape[:2]
    x0, y0, x1, y1 = box
    # 找欄時上下多看一段（至少 SPAN 像素高）：框只有兩三個字高時，字裡的空隙（ょ、く）會被當成欄與欄之間，
    # 字寬估成三分之一。讀的時候還是只讀框裡的那段
    extra = max(0, (SPAN - (y1 - y0)) // 2)
    y0, y1 = max(0, y0 - extra), min(H, y1 + extra)
    # 只轉框附近的灰階（整個螢幕轉一次要 60 ms）
    lx0, lx1 = max(0, x0 - MARGIN), min(Wd, x1 + MARGIN)
    sub = img[y0:y1, lx0:lx1]
    gray = sub.mean(axis=2) if sub.ndim == 3 else sub.astype(np.float32)
    bg = float(np.median(gray[:, x0 - lx0:x1 - lx0]))
    W = _width_guess((np.abs(gray[:, x0 - lx0:x1 - lx0] - bg) > INK).sum(axis=0))
    if W is None:
        return None
    # 框只切到一欄的一部分時，框裡估的字寬偏小：左右多看 1.5 個字寬再估一次（框邊的欄才完整），變大就再擴一次
    for _ in range(3):
        ex0, ex1 = max(lx0, int(x0 - 1.5 * W)), min(lx1, int(x1 + 1.5 * W))
        diff = np.abs(gray[:, ex0 - lx0:ex1 - lx0] - bg)
        mask = diff > INK
        # 小說是單色底：大部分像素和背景幾乎一樣。動畫、遊戲畫面、漫畫的圖（明暗漸層、線條）不是，
        # 投影會把頭髮、衣服的線條當成欄
        if mask.mean() > 0.35 or (diff < FLAT).mean() < 0.6:
            return None
        mask = _body_rows(mask, W)
        cnt = mask.sum(axis=0)
        if cnt.max() == 0:
            return None
        # 字寬：投影超過最高峰一成的範圍（振假名、橫排標題的投影很低），不算碰到左右邊（被切掉一半）的
        core = [e - s for s, e in _runs(cnt >= 0.1 * cnt.max()) if s > 0 and e < len(cnt) and e - s >= 3]
        if not core:
            break
        med = float(np.median(core))
        new = float(np.median([c for c in core if c >= 0.6 * med]))
        grew = new > 1.1 * W
        W = new
        if not grew:
            break
    pieces = [p for s, e in _runs(cnt > 0) for p in _split_valleys(cnt, s, e)]
    # 字寬改用整個峰的寬度（上面的估計只算投影超過一成的範圍，比字窄一點）：夠高、沒碰到左右邊的峰的中位數
    top = max(int(cnt[s:e].max()) for s, e in pieces)
    full = [e - s for s, e in pieces if s > 0 and e < len(cnt) and cnt[s:e].max() >= 0.3 * top and e - s >= 0.6 * W]
    if full:
        W = max(W, float(np.median(full)))
    # 1. 同一欄被字裡的空隙（い、川、ハ）切開的接回來：接起來不超過一個字寬
    #    振假名和正文的峰高差很多，不接（振假名緊貼在下一欄左邊時也一樣）
    merged = []
    for s, e in pieces:
        if merged and s - merged[-1][1] < 0.25 * W and e - merged[-1][0] <= 1.2 * W:
            p, q = cnt[merged[-1][0]:merged[-1][1]].max(), cnt[s:e].max()
            if min(p, q) >= 0.35 * max(p, q):
                merged[-1][1] = e
                continue
        merged.append([s, e])
    # 2. 太寬的（兩欄被振假名連起來）在中間墨水最少的地方切開
    cols = []
    for s, e in merged:
        while e - s > 1.6 * W:
            lo, hi = s + int(0.6 * W), e - int(0.6 * W)
            if hi <= lo:
                break
            cut = lo + int(np.argmin(cnt[lo:hi]))
            cols.append([s, cut])
            s = cut + 1
        cols.append([s, e])
    # 3. 振假名：比字寬窄很多、緊貼在左邊那一欄的右邊；其他太窄的（雜訊、框邊切到的半個字）也不要
    #    （字小時振假名可能有正文的七成寬：貼在欄右邊、墨水又少很多的也算）
    out, prev = [], None
    for s, e in cols:
        p = int(cnt[s:e].max())
        ruby = prev is not None and s - prev[1] <= 0.3 * W and e - s < 0.8 * W and p < 0.35 * prev[2]
        if e - s >= 0.6 * W and not ruby:
            out.append((ex0 + s, ex0 + e))
            prev = (s, e, p)
    if not out or any(e - s > 1.6 * W for s, e in out):
        return None
    return out, W, bg


def _vertical_here(lines: list[TLine], x0, y0, x1, y1) -> bool:
    """偵測到的行裡，碰到這個範圍、方向明確的大多是直排（或一行都沒有）：字幕、網頁這類橫排不用投影。"""
    touched = [ln for ln in lines if ln.box.x0 < x1 and ln.box.x1 > x0 and ln.box.y0 < y1 and ln.box.y1 > y0
               and max(ln.box.w, ln.box.h) >= 2 * min(ln.box.w, ln.box.h)]
    return not touched or sum(ln.vertical for ln in touched) >= 0.6 * len(touched)


def _textlike(rows: np.ndarray, W: float) -> bool:
    """一欄裡有墨水的列（排好序）：一個字一個字排下來，字與字之間大多有空白的列。
    圖裡的線條（頭髮、衣服、斜線）是連續的，空白很少。"""
    span = rows[-1] - rows[0] + 1
    has = np.zeros(span, bool)
    has[rows - rows[0]] = True
    return len(_runs(~has)) >= 0.4 * span / W


def _col_line(cols, k: int, W: float, ly0: int, ly1: int, bg: float) -> TLine:
    """第 k 欄的行：左右各多 PAD 個字寬（不超過到隔壁欄距離的一半），和偵測框差不多粗。
    太貼字時 manga-ocr 把直排的「ー」讀成「1」；refine 找字距空隙的範圍也是依行的粗細。
    也記下右邊的振假名帶（正文右緣到右邊那欄之前，最多 RUBY_BAND 個字寬），讀振假名用（ruby.py）。"""
    s, e = cols[k]
    pad = PAD * W
    left = (s - cols[k - 1][1]) / 2 if k > 0 else pad
    right = (cols[k + 1][0] - e) / 2 if k + 1 < len(cols) else pad
    band = int(min(cols[k + 1][0] - 1 if k + 1 < len(cols) else e + RUBY_BAND * W, e + RUBY_BAND * W))
    return TLine(Box(int(round(s - min(pad, left))), ly0, int(round(e + min(pad, right))), ly1), True,
                 ruby=(s, e, band, W, bg))


def _body(img: np.ndarray, x0: int, x1: int, y0: int, y1: int, W: float, bg: float) -> np.ndarray:
    sub = img[y0:y1, x0:x1]
    g = sub.mean(axis=2) if sub.ndim == 3 else sub.astype(np.float32)
    return _body_rows(np.abs(g - bg) > INK, W)


def plan_ink(img: np.ndarray, lines: list[TLine], box: tuple[int, int, int, int], need: int = 1):
    """框選：框裡有 need 欄以上的直排正文時回傳 ([每欄一行], [片段])；否則 None。"""
    x0, y0, x1, y1 = box
    if not _vertical_here(lines, x0, y0, x1, y1):
        return None
    found = find_columns(img, box)
    if found is None:
        return None
    cols, W, bg = found
    H = img.shape[0]
    chosen = [k for k, (s, e) in enumerate(cols) if min(e, x1) - max(s, x0) >= 0.4 * (e - s)]
    if len(chosen) < need:
        return None
    wy0, wy1 = max(0, int(y0 - 2 * W)), min(H, int(y1 + 2 * W))
    # 每欄的上下範圍：框把頁首的書名、頁碼一起框進來時，它們會黏在欄頭被讀進去（「けい」＋正文）。
    # 和找欄時一樣，只留夠高的列帶
    rx0 = cols[min(chosen)][0]
    body = _body(img, rx0, cols[max(chosen)][1], wy0, wy1, W, bg)
    tl, segs, cps = [], [], []
    tall = 0
    for k in sorted(chosen, reverse=True):  # 由右而左
        s, e = cols[k]
        rows = np.flatnonzero(body[:, s - rx0:e - rx0].any(axis=1))
        if len(rows) == 0:
            continue
        ly0, ly1 = wy0 + int(rows[0]), wy0 + int(rows[-1]) + 1
        a, b = max(0, y0 - ly0), min(ly1 - ly0, y1 - ly0)
        if b - a < 2:
            continue
        inside = rows[(rows >= y0 - wy0) & (rows < y1 - wy0)]
        if len(inside) and inside[-1] - inside[0] >= 3 * W:
            if not _textlike(inside, W):
                return None
            tall += 1
        tl.append(_col_line(cols, k, W, ly0, ly1, bg))
        segs.append(Segment(len(tl) - 1, float(a), float(b)))
        cps.append(body[:, s - rx0:e - rx0].sum(axis=1))
    # 至少要有一欄在框裡有 3 個字以上高：才看得出是直排（不是橫排字裡的空隙）
    if len(segs) < need or tall == 0:
        return None
    # 切點移到字格的格線（框邊切到的字大半在框裡才算＝四捨五入到最近的格線）
    out = []
    for g in segs:
        ln = tl[g.line]
        a = _grid_cut(cps[g.line], W, y0 - wy0, round) if g.a > 0 else 0.0
        b = _grid_cut(cps[g.line], W, y1 - wy0, round) if g.b < ln.length else float(ln.length)
        if a is None or b is None:  # 這欄估不出字格：照舊找字距空隙
            out += refine(img, tl, [g], box=True) or [g]
            continue
        if g.a > 0:
            a = max(0.0, a + wy0 - ln.box.y0)
        if g.b < ln.length:
            b = min(float(ln.length), b + wy0 - ln.box.y0)
        if b - a >= 0.3 * W:
            out.append(Segment(g.line, a, b))
    return tl, out or segs


def plan_drag(img: np.ndarray, lines: list[TLine], start, end):
    """反白：起點在直排正文上時，回傳 ([每欄一行], [片段])（起點那欄從起點到欄尾、中間的欄整欄、終點那欄從欄頭到終點，
    游標所在的字一定算進去）；起點不在直排正文上回傳 None（照舊用偵測的版面）。"""
    (sx, sy), (ex, ey) = start, end
    H, Wd = img.shape[:2]
    near = [ln for ln in lines if ln.distance(sx, sy) <= 0.35 * ln.thickness]
    if near and not min(near, key=lambda ln: ln.distance(sx, sy)).vertical:
        return None
    # 找欄的範圍：起點到終點之間（左右至少各 30 像素，find_columns 會再往外看）、整個畫面的高度：
    # 中間的欄整欄都要選，段落最後一欄很短、碰不到起點終點的高度也一樣
    x0, x1 = max(0, int(min(sx, ex)) - 30), min(Wd, int(max(sx, ex)) + 30)
    y0, y1 = max(0, int(min(sy, ey))), min(H, int(max(sy, ey)) + 1)
    if not _vertical_here(lines, x0, y0, x1, y1):
        return None
    found = _drag_columns(img, x0, x1)
    if found is None:
        return None
    cols, W, bg, body, lo = found

    def col_at(x, strict):
        dist = [max(s - x, 0, x - e) for s, e in cols]
        k = int(np.argmin(dist))
        return k if not strict or dist[k] <= 0.35 * W else None
    ks, ke = col_at(sx, True), col_at(ex, False)
    if ks is None:
        return None
    rows_s = np.flatnonzero(body[:, cols[ks][0] - lo:cols[ks][1] - lo].any(axis=1))
    if len(rows_s) == 0 or not rows_s[0] - 0.35 * W <= sy <= rows_s[-1] + 0.35 * W:
        return None  # 起點不在字上（欄尾以下的空白）
    if len(rows_s) >= 3 * W and not _textlike(rows_s, W):
        return None
    # 閱讀順序：右邊的欄先、同一欄由上而下；終點在起點前面就對調
    if (-ke, ey) < (-ks, sy):
        ks, ke, sy, ey = ke, ks, ey, sy
    tl, segs = [], []
    for k in range(ks, ke - 1, -1):
        s, e = cols[k]
        cp = body[:, s - lo:e - lo].sum(axis=1)
        rows = np.flatnonzero(cp)
        if len(rows) == 0:
            continue
        ly0, ly1 = int(rows[0]), int(rows[-1]) + 1
        a = sy if k == ks else ly0
        b = ey if k == ke else ly1
        # 字格：起點往前到那個字的開頭、終點往後到那個字的結尾；估不出來就照 refine。
        # 游標落在字與字的交界附近（離格線不到兩成格）時不算進交界另一邊的字
        if k == ks and a > ly0:
            a = _grid_cut(cp, W, a, lambda t: np.floor(t + EDGE))
        if k == ke and b < ly1:
            b = _grid_cut(cp, W, b, lambda t: np.ceil(t - EDGE))
        tl.append(_col_line(cols, k, W, ly0, ly1, bg))
        if a is None or b is None:
            raw = Segment(len(tl) - 1, max(0.0, (sy if k == ks else ly0) - ly0), min(float(ly1 - ly0), (ey if k == ke else ly1) - ly0))
            segs += refine(img, tl, [raw]) if raw.b - raw.a >= 2 else []
            continue
        a, b = max(0.0, a - ly0), min(float(ly1 - ly0), b - ly0)
        if b - a >= 2:
            segs.append(Segment(len(tl) - 1, a, b))
    return (tl, segs) if segs else None


# (圖, x0, x1, 結果)：整個替換（按下 Alt 後背景執行緒會先算好一份，見 prewarm）
_drag_cache: tuple = (None, 0, 0, None)


def prewarm(img: np.ndarray, lines: list[TLine], x, y):
    """按下 Alt 時游標在直排正文附近：在背景先把反白要用的欄找好存起來，拖曳第一下不用在主執行緒等幾十 ms。"""
    H, Wd = img.shape[:2]
    x0, x1 = max(0, int(x) - 30), min(Wd, int(x) + 30)
    if any(ln.vertical for ln in lines) and _vertical_here(lines, x0, max(0, int(y) - 100), x1, min(H, int(y) + 100)):
        _drag_columns(img, x0, x1)


def _drag_columns(img: np.ndarray, x0: int, x1: int):
    """反白用的欄：(欄, 字寬, 背景, 整個畫面高度的正文列帶遮罩（頁首、瀏覽器的分頁列不算）, 遮罩左邊的 x)；找不到回傳 None。
    拖曳時滑鼠每動一下都要算一次：多找左右各 MORE 像素存起來，拖曳範圍還在裡面就不用重算
    （整個畫面高度找欄要幾十 ms）。找不到（漫畫、圖）也存起來，不然在漫畫上拖曳每動一下都要重找一次。"""
    global _drag_cache
    c_img, c_x0, c_x1, c_found = _drag_cache
    if c_img is img and c_x0 <= x0 and x1 <= c_x1:
        return c_found
    H, Wd = img.shape[:2]
    out, rx0, rx1 = None, max(0, x0 - MORE), min(Wd, x1 + MORE)
    for a, b in ((rx0, rx1), (x0, x1)):  # 多找的範圍碰到圖、通知視窗就只找拖曳範圍
        top, bot = _edge_bars(img, a, b)
        found = find_columns(img, (a, top, b, bot))
        if found is not None:
            cols, W, bg = found
            lo = cols[0][0]
            body = np.zeros((H, cols[-1][1] - lo), bool)
            body[top:bot] = _body(img, lo, cols[-1][1], top, bot, W, bg)
            out = (cols, W, bg, body, lo)
            rx0, rx1 = a, b
            break
    _drag_cache = (img, rx0, rx1, out)
    return out


def _edge_bars(img: np.ndarray, x0: int, x1: int) -> tuple[int, int]:
    """反白找欄看整個畫面高度：去掉畫面最上、最下整列幾乎都不是背景的橫條（Windows 工作列），回傳 (上緣, 下緣)。
    工作列會讓每個 x 都有墨水，所有欄黏成一個大欄（字寬估成 460，讀到好幾欄的亂字）。"""
    sub = img[:, x0:x1]
    g = sub.mean(axis=2) if sub.ndim == 3 else sub.astype(np.float32)
    full = (np.abs(g - np.median(g)) > INK).mean(axis=1) > BAR
    top, bot = 0, len(full)
    while top < bot and full[top]:
        top += 1
    while bot > top and full[bot - 1]:
        bot -= 1
    return top, bot


def _grid_cut(col: np.ndarray, W: float, y: float, rnd) -> float | None:
    """y 移到這一欄的字格格線：rnd＝round（最近的格線，框選）、floor（往前到字的開頭）、ceil（往後到字的結尾）。
    估不出字格、或那條格線附近不是空白（這裡不照格子排）回傳 None。"""
    grid = char_grid(col, W)
    if grid is None:
        return None
    P, ph = grid
    g = ph + float(rnd((y - ph) / P)) * P
    i = int(round(g))
    near = col[max(0, i - int(0.15 * P)):i + int(0.15 * P) + 1]
    if len(near) and near.min() > 0.1 * col.max():
        return None
    return g


def char_grid(col: np.ndarray, W: float) -> tuple[float, float] | None:
    """一欄（直排）的字格：每個字佔一格、格距固定。col＝沿著欄的墨水投影；回傳格距 P 與第一條格線的位置。
    格線上的墨水要明顯比平均少才算找到（不照格子排的回傳 None）。"""
    ink = np.flatnonzero(col)
    if len(ink) == 0:
        return None
    top, bot = int(ink[0]), int(ink[-1]) + 1
    r = col[top:bot].astype(np.float64)
    if len(r) < 6 * W:
        return None

    def score(P):
        n = int((len(r) - 1) / P) + 1
        phases = np.arange(0, P, 0.5)
        idx = np.round(phases[:, None] + P * np.arange(n)[None, :]).astype(int)
        valid = idx < len(r)
        sc = np.where(valid, r[np.minimum(idx, len(r) - 1)], 0).sum(axis=1) / valid.sum(axis=1)
        k = int(np.argmin(sc))
        return float(sc[k]), float(P), float(phases[k])
    coarse = min(score(P) for P in np.arange(0.9 * W, 1.45 * W, 0.25))
    best = min(score(P) for P in np.arange(coarse[1] - 0.25, coarse[1] + 0.26, 0.02))
    if best[0] > 0.15 * r.mean():
        return None
    return best[1], top + best[2]
