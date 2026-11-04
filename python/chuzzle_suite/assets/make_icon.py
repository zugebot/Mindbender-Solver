# python/chuzzle_suite/assets/make_icon.py
"""Draws the Chuzzle Suite icon: four chuzzles (the title bar's dots) on a dark rounded tile.
Writes chuzzle_suite.ico (16-256 px) and chuzzle_suite.png (the window icon) next to this file.

Run: python python/chuzzle_suite/assets/make_icon.py"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
# the title bar's dots (title_bar._app_icon), clockwise from the top left
BALLS = ((230, 90, 90), (80, 200, 120), (240, 190, 70), (90, 170, 255))
TILE_TOP, TILE_BOTTOM, TILE_LINE = (44, 48, 60), (22, 24, 30), (70, 76, 94)


def _mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def draw(size: int) -> Image.Image:
    s = 8                                   # supersampling
    n = size * s
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    small = size <= 24                      # tiny sizes: no outline, bigger balls
    pad = 0 if small else n * 0.03
    radius = n * 0.22

    # tile with a vertical gradient
    grad = Image.new("RGBA", (n, n))
    gd = ImageDraw.Draw(grad)
    for y in range(n):
        gd.line([(0, y), (n, y)], fill=_mix(TILE_TOP, TILE_BOTTOM, y / n) + (255,))
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([pad, pad, n - 1 - pad, n - 1 - pad], radius, fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    if not small:
        d.rounded_rectangle([pad, pad, n - 1 - pad, n - 1 - pad], radius, outline=TILE_LINE + (255,),
                            width=max(s, round(n * 0.012)))

    # four chuzzles: shaded ball, darker rim, a soft highlight top left
    r = n * (0.205 if small else 0.185)
    offset = n * (0.235 if small else 0.215)
    centres = [(n / 2 - offset, n / 2 - offset), (n / 2 + offset, n / 2 - offset),
               (n / 2 + offset, n / 2 + offset), (n / 2 - offset, n / 2 + offset)]
    for (cx, cy), colour in zip(centres, BALLS):
        if not small:   # drop shadow
            sh = Image.new("RGBA", (n, n), (0, 0, 0, 0))
            ImageDraw.Draw(sh).ellipse([cx - r, cy - r + n * 0.025, cx + r, cy + r + n * 0.025], fill=(0, 0, 0, 110))
            img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(n * 0.02)))
            d = ImageDraw.Draw(img)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_mix(colour, (0, 0, 0), 0.28) + (255,))
        ri = r * 0.86
        d.ellipse([cx - ri, cy - ri - r * 0.06, cx + ri, cy + ri - r * 0.06], fill=colour + (255,))
        if size >= 20:
            hl = Image.new("RGBA", (n, n), (0, 0, 0, 0))
            hr = r * 0.42
            hx, hy = cx - r * 0.3, cy - r * 0.36
            ImageDraw.Draw(hl).ellipse([hx - hr, hy - hr * 0.75, hx + hr, hy + hr * 0.75], fill=(255, 255, 255, 150))
            img.alpha_composite(hl.filter(ImageFilter.GaussianBlur(r * 0.12)))
            d = ImageDraw.Draw(img)
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    images = [draw(size) for size in SIZES]
    images[-1].save(HERE / "chuzzle_suite.ico", sizes=[(s, s) for s in SIZES], append_images=images[:-1])
    draw(64).save(HERE / "chuzzle_suite.png")
    preview = Image.new("RGBA", (sum(SIZES) + 10 * len(SIZES) + 10, 276), (240, 240, 240, 255))
    x = 10
    for im in images:
        preview.alpha_composite(im, (x, 10))
        x += im.width + 10
    preview.save(HERE / "icon_preview.png")
    print("wrote", HERE / "chuzzle_suite.ico")


if __name__ == "__main__":
    main()
