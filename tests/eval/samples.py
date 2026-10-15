"""產生已知標準答案的測試圖：橫排對話框、字幕、網頁小說、直排小說、漫畫對話框、深色底直排。

每個樣本是「使用者框選出來的區域」，回傳 (名稱, BGR 圖, 標準答案)。標準答案不含換行。
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GOTHIC = r"C:\Windows\Fonts\YuGothM.ttc"
GOTHIC_B = r"C:\Windows\Fonts\YuGothB.ttc"
SERIF = r"C:\Windows\Fonts\NotoSerifTC-VF.ttf"  # 本機沒有日文明朝體；思源宋體也有假名

# 直排時要轉 90 度的字
_ROTATE = set("ー～〜…‥―─（）()「」『』【】")
# 直排時放在字格右上角的字
_TOP_RIGHT = set("、。，．")
_SMALL = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")


_CMAPS = {}


def _has_glyph(font, ch):
    from fontTools.ttLib import TTFont
    path = font.path
    if path not in _CMAPS:
        _CMAPS[path] = TTFont(path, fontNumber=0).getBestCmap()
    return ord(ch) in _CMAPS[path]


def _glyph(ch, font, color, size):
    """把單一個字畫在 size×size 的透明圖上（直排用）。字型缺字時改用游ゴシック。"""
    if not _has_glyph(font, ch):
        font = ImageFont.truetype(GOTHIC, font.size)
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    l, t, r, b = d.textbbox((0, 0), ch, font=font)
    d.text(((size - (r - l)) / 2 - l, (size - (b - t)) / 2 - t), ch, font=font, fill=color)
    if ch in _ROTATE:
        im = im.rotate(-90)
    return im


def draw_vertical(img, columns, font, x_right, y_top, color, pitch=1.05, col_gap=1.6, stroke=None):
    """由右而左畫直排文字。columns：每欄的文字。"""
    s = font.size
    cell = int(s * pitch)
    for ci, col in enumerate(columns):
        x = int(x_right - s - ci * s * col_gap)
        for j, ch in enumerate(col):
            y = int(y_top + j * cell)
            if ch in _TOP_RIGHT:
                g = Image.new("RGBA", (s, s), (0, 0, 0, 0))
                ImageDraw.Draw(g).text((s * 0.55, -s * 0.45), ch, font=font, fill=color)
            else:
                g = _glyph(ch, font, color, s)
            if stroke:
                a = np.asarray(g)[:, :, 3]
                k = np.ones((stroke[0] * 2 + 1,) * 2, np.uint8)
                halo = Image.new("RGBA", g.size, stroke[1] + (0,))
                halo.putalpha(Image.fromarray(cv2.dilate(a, k)))
                img.paste(halo, (x, y), halo)
            img.paste(g, (x, y), g)


def _bg_photo(w, h, seed):
    """有色塊與雜訊的背景（模擬遊戲／動畫畫面）。"""
    rng = np.random.default_rng(seed)
    small = rng.integers(40, 220, (6, 8, 3)).astype(np.uint8)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(np.clip(bg + rng.normal(0, 6, bg.shape), 0, 255).astype(np.uint8))


def _bgr(im):
    return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def game_dialog():
    lines = ["「もう少しだけ待ってくれないか。", "まだ確かめたいことが残っているんだ」"]
    im = _bg_photo(980, 230, 1)
    ov = Image.new("RGBA", im.size, (10, 15, 40, 200))
    im = Image.alpha_composite(im.convert("RGBA"), ov)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(GOTHIC_B, 38)
    d.text((30, 18), "レオン", font=ImageFont.truetype(GOTHIC_B, 30), fill=(255, 220, 120))
    for i, ln in enumerate(lines):
        d.text((40, 80 + i * 60), ln, font=f, fill="white")
    return _bgr(im), "レオン" + "".join(lines)


def anime_sub():
    text = "あの日の約束、今でも覚えてる？"
    im = _bg_photo(900, 110, 2)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(GOTHIC_B, 46)
    d.text((40, 25), text, font=f, fill="white", stroke_width=4, stroke_fill="black")
    return _bgr(im), text


def web_novel():
    lines = ["彼女は窓の外を眺めながら、小さくため息をついた。",
             "昨日の出来事が、まだ頭から離れない。",
             "あれは本当に偶然だったのだろうか。",
             "考えれば考えるほど、分からなくなっていく。"]
    im = Image.new("RGB", (900, 260), (252, 252, 250))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(GOTHIC, 30)
    for i, ln in enumerate(lines):
        d.text((20, 20 + i * 58), ln, font=f, fill=(30, 30, 30))
    return _bgr(im), "".join(lines)


NOVEL_COLS = ["　春の終わりの夕暮れ、駅前の古い喫茶店で",
              "私は彼を待っていた。約束の時間はとうに過ぎ",
              "ている。窓の向こうでは、桜の花びらが風に舞",
              "い、人々が足早に通り過ぎていく。もう来ない",
              "のかもしれない、と思いかけたそのとき、扉の",
              "鈴が鳴った。"]


def vertical_novel():
    im = Image.new("RGB", (520, 820), (248, 244, 232))
    f = ImageFont.truetype(SERIF, 34)
    draw_vertical(im, NOVEL_COLS, f, x_right=500, y_top=20, color=(25, 25, 25), col_gap=1.75)
    return _bgr(im), "".join(c.replace("　", "") for c in NOVEL_COLS)


def manga_bubble():
    cols = ["そんなこと", "言われなくても", "分かってるよ！"]
    rng = np.random.default_rng(3)
    base = np.full((420, 330), 200, np.uint8)
    yy, xx = np.mgrid[:420, :330]
    base[(xx + yy) % 7 < 2] = 120  # 網點
    base = np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
    im = Image.fromarray(base).convert("RGB")
    d = ImageDraw.Draw(im)
    d.ellipse([15, 10, 315, 410], fill="white", outline="black", width=3)
    f = ImageFont.truetype(GOTHIC_B, 36)
    draw_vertical(im, cols, f, x_right=235, y_top=70, color="black", col_gap=1.3)
    return _bgr(im), "".join(cols)


def vertical_dark():
    cols = ["夜明けまで", "あと少しだけ", "そばにいて"]
    im = _bg_photo(300, 460, 4)
    im = Image.alpha_composite(im.convert("RGBA"), Image.new("RGBA", im.size, (0, 0, 0, 140)))
    f = ImageFont.truetype(GOTHIC_B, 40)
    draw_vertical(im, cols, f, x_right=270, y_top=20, color=(255, 255, 255, 255), col_gap=1.5,
                  stroke=(2, (0, 0, 0)))
    return _bgr(im), "".join(cols)


ALL = [game_dialog, anime_sub, web_novel, vertical_novel, manga_bubble, vertical_dark]


def load_all():
    return [(fn.__name__, *fn()) for fn in ALL]


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    for name, img, gt in load_all():
        cv2.imwrite(f"{out}/{name}.png", img)
        print(name, img.shape, gt)
