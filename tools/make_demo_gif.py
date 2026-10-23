"""README 的示範動畫 docs/demo.gif：在一份假資料的日文文件上反白兩句，顯示查字結果、記錄一個字。

    python tools/make_demo_gif.py [輸出.gif]

不是螢幕錄影（全域掛鉤會忽略程式模擬的按鍵）：文件是這裡畫的圖，但辨識、反白範圍、查字、結果視窗都是真的
（Recognizer → layout.plan_drag → read_segments → words.extract → ResultPopup.grab()），每一格照 app 的樣子合成：
反白色塊用 HighlightOverlay 的顏色、結果視窗用 place_near 的規則放在選取範圍旁邊。
英文介面、英文解釋（GitHub 的訪客大多看英文）。單字本用暫存資料夾，不碰真的單字本；不呼叫 Gemini。
文件內容是自己寫的句子，沒有著作權問題。
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
os.environ["JPLENS_UI_LANG"] = "en"  # 要在 import jplens 的介面模組之前

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from PySide6.QtCore import QBuffer, QCoreApplication, QEvent, QIODevice, QPoint, QThread  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

app = QApplication([])
from jplens import lang  # noqa: E402
from jplens.config import Config  # noqa: E402
from jplens.dictionary import Dictionary  # noqa: E402
from jplens.i18n import t  # noqa: E402
from jplens.layout import build_layout, plan_drag  # noqa: E402
from jplens.recognize import Recognizer  # noqa: E402
from jplens.ui import ResultPopup  # noqa: E402
from jplens.vocab import VocabStore  # noqa: E402
from jplens.words import extract  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "demo.gif"
W, H = 1500, 860
FPS = 12
CAPTION_H = 70
SCALE = 0.72  # GIF 的大小（1080 寬）
FONT = "C:/Windows/Fonts/YuGothM.ttc"
TITLE = "週末の小さな旅"
BODY = ["先週の土曜日、友達と二人で鎌倉へ出かけた。",
        "朝から雨が降りそうな空だったけれど、",
        "駅に着いたころにはすっかり晴れていた。",
        "海沿いの道を歩きながら、久しぶりに昔の話をした。",
        "帰りの電車で、また来ようと約束した。"]
X0, Y0, LINE_H, SIZE = 70, 160, 66, 32
HL = (60, 150, 255, 90)  # ui.HighlightOverlay 的顏色


def document() -> Image.Image:
    im = Image.new("RGB", (W, H), (236, 238, 242))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((30, 30, W - 30, H - 30), 14, fill=(255, 255, 255), outline=(210, 214, 220))
    d.text((X0, 70), TITLE, font=ImageFont.truetype(FONT, 40), fill=(30, 30, 35))
    f = ImageFont.truetype(FONT, SIZE)
    for i, s in enumerate(BODY):
        d.text((X0, Y0 + i * LINE_H), s, font=f, fill=(40, 40, 45))
    return im


def char_pos(line: int, k: int, end=False) -> tuple[int, int]:
    """第 line 行第 k 個字（end＝這個字的右邊）的螢幕座標（字的垂直中間）。"""
    f = ImageFont.truetype(FONT, SIZE)
    s = BODY[line]
    x = X0 + f.getlength(s[:k + (1 if end else 0)]) - (2 if end else -2)
    return int(x), Y0 + line * LINE_H + SIZE // 2 + 4


def pump(n=30):
    for _ in range(n):
        QCoreApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QThread.msleep(10)


def grab(w) -> Image.Image:
    pump()
    pm = w.grab()
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pm.save(buf, "PNG")
    im = Image.open(__import__("io").BytesIO(bytes(buf.data()))).convert("RGB")
    return im  # 實體像素（150% 縮放的螢幕上，結果視窗的字比文件的字小一點，和實際使用時差不多）


def cursor(d: ImageDraw.ImageDraw, x: int, y: int, alt: bool):
    pts = [(0, 0), (0, 26), (7, 20), (12, 31), (17, 29), (12, 18), (21, 18)]
    d.polygon([(x + a, y + b) for a, b in pts], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    if alt:  # 按住 Alt 的提示
        f = ImageFont.truetype(FONT, 18)
        d.rounded_rectangle((x + 22, y + 28, x + 70, y + 56), 6, fill=(40, 42, 48), outline=(120, 124, 132))
        d.text((x + 31, y + 30), "Alt", font=f, fill=(255, 255, 255))


class Demo:
    def __init__(self):
        self.doc = document()
        self.img = np.asarray(self.doc)[:, :, ::-1].copy()
        self.rec = Recognizer()
        self.rec.warmup()
        self.lines = build_layout(self.rec.detect_screen(self.img), self.img)
        self.dic = Dictionary()
        cfg = Config()
        cfg.explain_lang = "en"
        cfg.translate_timing = "manual"
        cfg.min_level = 5  # 示範時每個級數都顯示（預設只顯示 N3 以上）
        cfg.font_size = "l"
        lang.set_explain("en")
        self.store = VocabStore(Path(tempfile.mkdtemp()) / "saved_words.jsonl")
        self.store.dictionary = self.dic
        self.popup = ResultPopup(cfg, self.store, "")
        self.frames: list[Image.Image] = []
        self.caption = ""  # 文件下方的步驟說明

    def segs(self, start, end):
        lines, segs = plan_drag(self.img, self.lines, start, end)
        return lines, segs

    def frame(self, mouse, alt=False, rects=(), popup=None, at=None, n=1):
        im = self.doc.copy()
        if rects:
            ov = Image.new("RGBA", im.size)
            od = ImageDraw.Draw(ov)
            for r in rects:
                od.rectangle(r, fill=HL)
            im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
        if popup is not None:
            sh = Image.new("RGBA", (popup.width + 24, popup.height + 24), (0, 0, 0, 0))
            ImageDraw.Draw(sh).rounded_rectangle((12, 16, popup.width + 12, popup.height + 18), 10, fill=(0, 0, 0, 60))
            im.paste(sh, (at[0] - 12, at[1] - 12), sh)
            im.paste(popup, at)
        d = ImageDraw.Draw(im)
        if mouse is not None:
            cursor(d, *mouse, alt)
        full = Image.new("RGB", (W, H + CAPTION_H), (236, 238, 242))  # 文件下方一條步驟說明
        full.paste(im, (0, 0))
        if self.caption:
            f = ImageFont.truetype("C:/Windows/Fonts/segoeuisl.ttf", 36)
            d = ImageDraw.Draw(full)
            d.text(((W - d.textlength(self.caption, font=f)) / 2, H - 6), self.caption, font=f, fill=(45, 80, 150))
        self.frames += [full] * n

    def move(self, a, b, secs, **kw):
        steps = max(1, int(secs * FPS))
        for i in range(1, steps + 1):
            f = i / steps
            f = f * f * (3 - 2 * f)
            self.frame((int(a[0] + (b[0] - a[0]) * f), int(a[1] + (b[1] - a[1]) * f)), **kw)

    def lookup(self, start, end, mouse_from):
        """移到起點 → 按住 Alt 拖到終點（反白跟著長）→ 放開出結果。回傳 (結果視窗的圖, 位置, 反白)。"""
        self.move(mouse_from, start, 0.6)
        self.frame(start, alt=True, n=4)
        steps = int(1.4 * FPS)
        rects = []
        for i in range(1, steps + 1):
            p = (int(start[0] + (end[0] - start[0]) * i / steps), int(start[1] + (end[1] - start[1]) * i / steps))
            lines, segs = self.segs(start, p)
            rects = [lines[g.line].rect(g.a, g.b) for g in segs]
            self.frame(p, alt=True, rects=rects)
        lines, segs = self.segs(start, end)
        text = self.rec.read_segments(self.img, lines, segs)
        words = extract([text], self.dic)
        print("讀到", text, [w.word for w in words])
        self.popup.show_result([text], words, 0.18, "")
        pop = grab(self.popup)
        x1 = max(r[2] for r in rects)
        y0 = min(r[1] for r in rects)
        at = (min(x1 + 16, W - pop.width - 20), max(20, min(y0 - 40, H - pop.height - 20)))
        self.frame(end, rects=rects, n=2)
        return pop, at, rects

    def run(self):
        start, end = char_pos(1, 3), char_pos(1, len(BODY[1]) - 1, end=True)  # 「雨が降りそうな空だったけれど」
        self.caption = "Hold Alt and drag over any Japanese text on screen"
        self.frame((400, 620), n=6)
        pop, at, rects = self.lookup(start, end, (400, 620))
        self.caption = "Words, readings, JLPT levels and grammar, in a fraction of a second"
        self.frame(end, rects=rects, popup=pop, at=at, n=int(2.6 * FPS))
        # 按第一個「＋ Save」
        self.caption = "Save words to your vocabulary book for quizzes and review"
        btn = next(b for b in self.popup.findChildren(QPushButton) if b.text() == t("＋ 記錄") and b.isVisible())
        c = btn.mapTo(self.popup, QPoint(btn.width() // 2, btn.height() // 2))
        dpr = pop.width / self.popup.width()  # 截圖是實體像素、元件座標是邏輯像素
        target = (at[0] + int(c.x() * dpr), at[1] + int(c.y() * dpr))
        self.move(end, target, 0.7, rects=rects, popup=pop, at=at)
        self.frame(target, rects=rects, popup=pop, at=at, n=3)
        btn.click()
        pop = grab(self.popup)
        self.frame(target, rects=rects, popup=pop, at=at, n=int(1.4 * FPS))
        # 第二句：「海沿いの道を歩きながら、久しぶりに昔の話をした」
        self.popup.hide()
        self.caption = "Works on games, anime, manga, novels and web pages"
        start2, end2 = char_pos(3, 0), char_pos(3, len(BODY[3]) - 1, end=True)
        pop, at, rects = self.lookup(start2, end2, target)
        self.frame(end2, rects=rects, popup=pop, at=at, n=int(2.2 * FPS))
        # 往下捲到文法
        area = self.popup.scroll
        bar = area.verticalScrollBar()
        c = area.mapTo(self.popup, QPoint(area.width() // 2, area.height() // 2))
        dpr = pop.width / self.popup.width()
        wheel = (at[0] + int(c.x() * dpr), at[1] + int(c.y() * dpr))
        self.move(end2, wheel, 0.6, rects=rects, popup=pop, at=at)
        for i in range(1, 7):
            bar.setValue(bar.maximum() * i // 6)
            self.frame(wheel, rects=rects, popup=grab(self.popup), at=at)
        self.frame(wheel, rects=rects, popup=grab(self.popup), at=at, n=int(3.0 * FPS))

    def save(self):
        size = (int(W * SCALE), int((H + CAPTION_H) * SCALE))
        frames = [f.resize(size, Image.LANCZOS) for f in self.frames]
        sample = frames[::max(1, len(frames) // 12)]  # 共用一個調色盤：從整段挑幾格拼起來算（每格各自算會閃）
        strip = Image.new("RGB", (size[0], size[1] * len(sample)))
        for i, f in enumerate(sample):
            strip.paste(f, (0, i * size[1]))
        pal = strip.quantize(colors=224, method=Image.Quantize.MEDIANCUT)
        q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
        OUT.parent.mkdir(parents=True, exist_ok=True)
        q[0].save(OUT, save_all=True, append_images=q[1:], duration=int(1000 / FPS), loop=0, optimize=True)
        print(f"{OUT}：{len(frames)} 格、{len(frames) / FPS:.1f} 秒、{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    demo = Demo()
    demo.run()
    demo.save()
