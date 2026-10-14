"""web/gen_sprites.py - builds the page's Chuzzle sprites from the game textures in
python/deluxe_clone/textures/, the same way python/deluxe_clone/engine/textures.py does:
the grey body sheet multiplied by the colour, plus 60 x the highlight mask, cut by the
body's alpha mask (yellow uses its own painted sheet), then eyes and pupils on top.

Writes web/assets/:
  chuzzles.webp  8 rows (colours 0-7) x 24 frames of 80 px: 20 idle sway frames, then one
                 stretched frame per direction (down, left, right, up) for sliding
  huge.webp      the same for the 2x2 piece: 8 rows x 14 frames of 180 px (10 sway + 4)
  eyes.webp      4 eye shapes x 6 pupil colours, 80 px, laid over a normal ball
  eyes-huge.webp the same 24 for the 2x2 piece, 180 px
  frame.webp     the board's rainbow border (Border.png with its alpha)
  cursors.webp   the game's cursors (the replay guide uses the pink and yellow wands)
  bkg1-9.webp    the backdrops' pattern, as seamless tiles

    python web/gen_sprites.py
"""
import pathlib

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEX = ROOT / "python/deluxe_clone/textures"
OUT = ROOT / "web/assets"

# engine/textures.py colorsMain (yellow, 4, is a painted sheet; its 2x2 uses a tint)
TINTS = {0: (178, 8, 8), 1: (8, 174, 8), 2: (8, 8, 255), 3: (255, 105, 2),
         5: (207, 7, 207), 6: (255, 255, 255), 7: (87, 197, 221)}
HUGE_YELLOW = (255, 255, 0)

# engine/chuzzle.py: eye placement and pupil centres per eye shape
NORMAL_EYES_POS = (-2, -3)
EYE_PUPIL_CENTERS = [((34, 60), (53, 62)), ((33, 62), (53, 63)), ((34, 63), (52, 62)), ((35, 62), (54, 62))]

DIR_SEGMENTS = (0, 1, 2, 3)    # frames 20 + 10s.. (normal) / 10 + 5s.. (huge): down, left, right, up


def rgba(name):
    return np.asarray(Image.open(TEX / name).convert("RGBA")).astype(np.float32)


def mask(name):
    return rgba(name)[..., 0]           # PopCap alpha sheets keep the alpha in the red channel


def tinted(base, hilite, alpha, color, add):
    rgb = base[..., :3] * (np.array(color, np.float32) / 255.0) + (add * hilite / 255.0)[..., None]
    return np.dstack([np.clip(rgb, 0, 255), alpha])


def to_image(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")


def build_balls():
    base = rgba("Chuzzle-Base.png")
    hilite = mask("Chuzzle-Hilite_Alpha.png")
    alpha = mask("Chuzzle-Base_Alpha.png")
    yellow = rgba("Chuzzle-Yellow.png")[:base.shape[0], :base.shape[1]]
    frames = list(range(20)) + [20 + 10 * s + 6 for s in DIR_SEGMENTS]
    sheet = Image.new("RGBA", (80 * len(frames), 80 * 8))
    for color in range(8):
        if color == 4:
            strip = np.dstack([yellow[..., :3], alpha])
        else:
            strip = tinted(base, hilite, alpha, TINTS[color], 60)
        img = to_image(strip)
        for k, f in enumerate(frames):
            sheet.paste(img.crop((81 * f, 2, 81 * f + 80, 82)), (80 * k, 80 * color))
    return sheet


def build_huge():
    base = rgba("ChuzzleHuge-Base.png")
    hilite = mask("ChuzzleHuge-Hilite_Alpha.png")
    alpha = mask("ChuzzleHuge-Base_Alpha.png")
    frames = list(range(10)) + [10 + 5 * s + 3 for s in DIR_SEGMENTS]
    sheet = Image.new("RGBA", (180 * len(frames), 180 * 8))
    for color in range(8):
        img = to_image(tinted(base, hilite, alpha, HUGE_YELLOW if color == 4 else TINTS[color], 60))
        for k, f in enumerate(frames):
            sheet.paste(img.crop((179 * f, 0, 179 * f + 179, 179)), (180 * k, 180 * color))
    return sheet


def eye_parts():
    eyes = Image.open(TEX / "Eyes.png").convert("RGBA")
    eyes.putalpha(Image.open(TEX / "Eyes_Alpha.png").convert("RGBA").split()[0])
    pupils = Image.open(TEX / "Pupils.png").convert("RGBA")
    pupils.putalpha(Image.open(TEX / "Pupils_Alpha.png").convert("RGBA").split()[0])
    ew, pw = eyes.width // 4, pupils.width // 6
    return ([eyes.crop((ew * i, 0, ew * (i + 1), eyes.height)) for i in range(4)],
            [pupils.crop((pw * i, 0, pw * (i + 1), pupils.height)) for i in range(6)])


def build_eyes(size, eyes_pos, centers_for):
    shapes, pupils = eye_parts()
    sheet = Image.new("RGBA", (size * 6, size * 4))
    for s, eye in enumerate(shapes):
        for p, pupil in enumerate(pupils):
            tile = Image.new("RGBA", (size, size))
            tile.alpha_composite(eye, (max(0, eyes_pos[0]), max(0, eyes_pos[1])),
                                 (max(0, -eyes_pos[0]), max(0, -eyes_pos[1])))
            for cx, cy in centers_for(s):
                x = eyes_pos[0] + cx - pupil.width // 2
                y = eyes_pos[1] + cy - pupil.height // 2
                tile.alpha_composite(pupil, (x, y))
            sheet.paste(tile, (size * p, size * s))
    return sheet


def build_eyes_huge(size=180, scale=2.3, left_pupil=(70, 100)):
    # engine/chuzzle.py puts the 2x2 piece's pupils 45 px apart (70,95 and 115,95), which
    # is the normal eye texture drawn about 2.3x; scale it and line it up by the pupils.
    shapes, pupils = eye_parts()
    big = lambda im: im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
    sheet = Image.new("RGBA", (size * 6, size * 4))
    for s, eye in enumerate(shapes):
        eye = big(eye)
        (lx, ly), (rx, ry) = EYE_PUPIL_CENTERS[s]
        ox = round(left_pupil[0] - lx * scale)
        oy = round(left_pupil[1] - ly * scale)
        for p, pupil in enumerate(pupils):
            pupil = big(pupil)
            tile = Image.new("RGBA", (size + 64, size + 64))     # margin, cropped below
            tile.alpha_composite(eye, (ox + 32, oy + 32))
            for cx, cy in ((lx, ly), (rx, ry)):
                tile.alpha_composite(pupil, (round(ox + cx * scale - pupil.width / 2) + 32,
                                             round(oy + cy * scale - pupil.height / 2) + 32))
            sheet.paste(tile.crop((32, 32, 32 + size, 32 + size)), (size * p, size * s))
    return sheet


def build_cursors():
    # CustomCursors.png: 5 cursors of 64 px (pink wand, clock, orange wand, I-beam, yellow wand)
    cur = Image.open(TEX / "CustomCursors.png").convert("RGBA")
    cur.putalpha(Image.open(TEX / "CustomCursors_Alpha.png").convert("RGBA").split()[0])
    return cur


def build_frame():
    frame = Image.open(TEX / "Border.png").convert("RGBA")
    frame.putalpha(Image.open(TEX / "Border_Alpha.png").convert("RGBA").split()[0])
    return frame


def backdrop_tile(i):
    # The backdrops (800x600) carry the game's board panel at x 229-764 (Border.png's
    # place); the strip left of its shadow is pure pattern. Mirrored both ways it tiles seamlessly.
    strip = Image.open(TEX / f"Backdrops/bkg{i}.jpg").convert("RGB").crop((0, 0, 190, 600))
    row = Image.new("RGB", (380, 600))
    row.paste(strip, (0, 0))
    row.paste(strip.transpose(Image.Transpose.FLIP_LEFT_RIGHT), (190, 0))
    tile = Image.new("RGB", (380, 1200))
    tile.paste(row, (0, 0))
    tile.paste(row.transpose(Image.Transpose.FLIP_TOP_BOTTOM), (0, 600))
    return tile


def main():
    OUT.mkdir(exist_ok=True)
    save = lambda im, name, q=88: im.save(OUT / name, "WEBP", quality=q, method=6)
    save(build_balls(), "chuzzles.webp")
    save(build_huge(), "huge.webp", 80)
    save(build_eyes(80, NORMAL_EYES_POS, lambda s: EYE_PUPIL_CENTERS[s]), "eyes.webp", 92)
    save(build_eyes_huge(), "eyes-huge.webp", 92)
    save(build_frame(), "frame.webp", 90)
    save(build_cursors(), "cursors.webp", 92)
    for i in range(1, 10):
        save(backdrop_tile(i), f"bkg{i}.webp", 82)
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:16} {f.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
