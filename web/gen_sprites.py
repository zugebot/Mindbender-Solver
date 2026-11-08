"""web/gen_sprites.py - builds the page's art from the game textures in
python/deluxe_clone/textures/, the same way python/deluxe_clone/engine/textures.py does:
the grey body sheet multiplied by the colour, plus 60 x the highlight mask, cut by the
body's alpha mask (yellow uses its own painted sheet). Eyes and pupils stay separate,
because the page moves the pupils at run time the way engine/chuzzle.py does.

Writes web/assets/:
  chuzzles.webp   8 rows (colours 0-7) x 60 frames of 80 px, the game's full bank:
                  20 idle sway frames, then 4 directions (down, left, right, up) x 10 speeds
  huge.webp       the 2x2 chuzzle: 8 rows x 30 frames (10 sway + 4 x 5), 179 px in 180 px slots
  eyes.webp       the 4 eye shapes (Eyes.png, 85 x 88 each)
  pupils.webp     the 6 pupil colours (Pupils.png, 14 x 14 each)
  frame.webp      the board's rainbow border (Border.png with its alpha)
  cursors.webp    the game's cursors (the replay guide uses the pink and yellow wands)
  orbs.webp       the game's blank round button (GameButton.png): normal, hover, pressed, 120 px each
  candy.webp      the candy wrapper behind the Scramble button (ScrambulatorButtonBG.png)
  fizzy.webp      the fizzy star (21 x 20): the match outline round a marked chuzzle
  arrows.webp     the Scrambulator's arrows (5 frames of 60 x 104), on the candy's orange ends
  scrollbtn.webp  the scroll arrows (ScrollUp / ScrollDown.png): 2 rows (up, down) x 3 frames
                  of 40 x 30 (normal, hover, pressed)
  scrollthumb.webp  the options slider's star (SliderThumb.png, 30 px), the scroll thumb
  scrolltrack.webp  the options slider's bar (OptionSliderBar.png) stood upright: 21 x 191
  naked, wince, angry, beauty, cheeks, rosy, fuzz, star, splode, shockwave
                  the chuzzles' poke reactions (Chuzzle.cpp antics, Animators.cpp)
  sounds/*.ogg    the chuzzle voices for them (from the decomp's exe/Sounds/speaks)
  bkg1-9.webp     the backdrops' pattern, as seamless tiles
  bauhaus36.webp  the game's Bauhaus 93 bitmap font (textures/fonts/Bauhaus9336_Alpha.gif)
and web/font.js, that font's glyph metrics (from tools/font/Bauhaus9336.py).

    python web/gen_sprites.py
"""
import ast
import json
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


def rgba(name):
    return np.asarray(Image.open(TEX / name).convert("RGBA")).astype(np.float32)


def mask(name):
    return rgba(name)[..., 0]           # PopCap alpha sheets keep the alpha in the red channel


def tinted(base, hilite, alpha, color, add):
    rgb = base[..., :3] * (np.array(color, np.float32) / 255.0) + (add * hilite / 255.0)[..., None]
    return np.dstack([np.clip(rgb, 0, 255), alpha])


def to_image(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")


def with_alpha(name, alpha_name):
    im = Image.open(TEX / name).convert("RGBA")
    im.putalpha(Image.open(TEX / alpha_name).convert("RGBA").split()[0])
    return im


def build_balls():
    # loadChuzzleTextures: 60 frames, frame i = (81 i, 2, 80, 80)
    base = rgba("Chuzzle-Base.png")
    hilite = mask("Chuzzle-Hilite_Alpha.png")
    alpha = mask("Chuzzle-Base_Alpha.png")
    yellow = rgba("Chuzzle-Yellow.png")[:base.shape[0], :base.shape[1]]
    sheet = Image.new("RGBA", (80 * 60, 80 * 8))
    for color in range(8):
        strip = np.dstack([yellow[..., :3], alpha]) if color == 4 else tinted(base, hilite, alpha, TINTS[color], 60)
        img = to_image(strip)
        for f in range(60):
            sheet.paste(img.crop((81 * f, 2, 81 * f + 80, 82)), (80 * f, 80 * color))
    return sheet


def build_huge():
    # loadChuzzleTexturesFat: 30 frames, frame i = (179 i, 0, 179, 179)
    base = rgba("ChuzzleHuge-Base.png")
    hilite = mask("ChuzzleHuge-Hilite_Alpha.png")
    alpha = mask("ChuzzleHuge-Base_Alpha.png")
    sheet = Image.new("RGBA", (180 * 30, 180 * 8))
    for color in range(8):
        img = to_image(tinted(base, hilite, alpha, HUGE_YELLOW if color == 4 else TINTS[color], 60))
        for f in range(30):
            sheet.paste(img.crop((179 * f, 0, 179 * f + 179, 179)), (180 * f, 180 * color))
    return sheet


def build_font():
    # tools/font/Bauhaus9336.py: CharList / WidthList / RectList / OffsetList, ascent 43, space 12
    src = (ROOT / "python/deluxe_clone/tools/font/Bauhaus9336.py").read_text()
    lists = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
            name = node.targets[0].attr
            if name in ("CharList", "WidthList", "RectList", "OffsetList"):
                lists[name] = ast.literal_eval(node.value)
    glyphs = {c: {"r": r, "o": o, "w": w}
              for c, w, r, o in zip(lists["CharList"], lists["WidthList"], lists["RectList"], lists["OffsetList"])}
    metrics = {"size": 36, "ascent": 43, "height": 54, "space": 12, "glyphs": glyphs}
    a = Image.open(TEX / "fonts/Bauhaus9336_Alpha.gif").convert("L")
    atlas = Image.new("RGBA", a.size, (255, 255, 255, 0))
    atlas.putalpha(a)
    return atlas, metrics


def build_frame():
    return with_alpha("Border.png", "Border_Alpha.png")


def build_cursors():
    # CustomCursors.png: 5 cursors of 64 px (pink wand, clock, orange wand, I-beam, yellow wand)
    return with_alpha("CustomCursors.png", "CustomCursors_Alpha.png")


def build_orbs():
    # GameButton.png: the game's blank round button, 3 states of 202 px (normal blue, hover
    # purple, pressed pink), scaled to 120 px each for the page's buttons
    im = with_alpha("GameButton.png", "GameButton_Alpha.png")
    w = im.width // 3
    sheet = Image.new("RGBA", (120 * 3, 120))
    for k in range(3):
        sheet.paste(im.crop((w * k, 0, w * k + w, im.height)).resize((120, 120), Image.LANCZOS), (120 * k, 0))
    return sheet


def build_candy():
    # ScrambulatorButtonBG.png: the orange candy wrapper behind the Scrambulator's Scramble button
    return with_alpha("ScrambulatorButtonBG.png", "ScrambulatorButtonBG_Alpha.png")


def build_scroll():
    # The game's scroll arrows and its options slider (star thumb on a red-rimmed groove),
    # for the page's scrollbars
    btn = Image.new("RGBA", (120, 60))
    btn.paste(with_alpha("ScrollUp.png", "ScrollUp_Alpha.png"), (0, 0))
    btn.paste(with_alpha("ScrollDown.png", "ScrollDown_Alpha.png"), (0, 30))
    thumb = with_alpha("SliderThumb.png", "SliderThumb_Alpha.png")
    track = with_alpha("OptionSliderBar.png", "OptionSliderBar_Alpha.png").transpose(Image.Transpose.ROTATE_90)
    return {"scrollbtn.webp": btn, "scrollthumb.webp": thumb, "scrolltrack.webp": track}


def white_alpha(name):
    # an alpha-only texture (PopCap keeps it in the red channel): white with that alpha, tinted
    # at run time the way the game's SetColor does
    a = Image.open(TEX / name).convert("RGBA").split()[0]
    im = Image.new("RGBA", a.size, (255, 255, 255, 0))
    im.putalpha(a)
    return im


def build_fx():
    # The poke reactions and markers (src/game/Chuzzle.cpp, Animators.cpp), one file each:
    return {
        "fizzy.webp": white_alpha("FizzyStar_Alpha.png"),          # 21 x 20: the match outline's star
        "arrows.webp": with_alpha("Scrambulator_Arrow.png", "Scrambulator_Arrow_Alpha.png"),  # 5 x 60 x 104
        "naked.webp": with_alpha("NakedChuzzle.png", "NakedChuzzle_Alpha.png"),   # 64: bald
        "wince.webp": with_alpha("Wince.png", "Wince_Alpha.png"),                 # 40 x 12
        "angry.webp": with_alpha("AngryEyes.png", "AngryEyes_Alpha.png"),         # 50 x 27
        "beauty.webp": with_alpha("Beauty.png", "Beauty_Alpha.png"),              # 114 x 121
        "cheeks.webp": with_alpha("ChuzzleCheeks.png", "ChuzzleCheeks_Alpha.png"),  # 98 x 50
        "rosy.webp": white_alpha("RosyCheeks_Alpha.png"),          # 98 x 50, over the cheeks
        "fuzz.webp": white_alpha("Fuzz_Alpha.png"),                # 16 x 18 tuft
        "star.webp": white_alpha("Star_Alpha.png"),                # 64, drawn yellow
        "splode.webp": with_alpha("Chuzzle-Splode.png", "Chuzzle-Splode_Alpha.png"),  # 17 x 149
        "shockwave.webp": with_alpha("RainbowSplode.png", "RainbowSplode_Alpha.png"),  # 256
    }


SOUNDS = ["poke_boredom", "pokechuzzle", "oggleoggle", "cough", "smallsneeze", "hugesneeze",
          "hurk", "SweetRelief", "handsoff"]


def copy_sounds():
    # The game's chuzzle voices (exe/Sounds/speaks/*.ogg in the decomp's copy of the game).
    src = pathlib.Path("X:/chuzzle-decomp/exe/Sounds/speaks")
    out = OUT / "sounds"
    out.mkdir(exist_ok=True)
    for name in SOUNDS:
        f = src / f"{name}.ogg"
        if f.exists():
            (out / f"{name.lower()}.ogg").write_bytes(f.read_bytes())


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
    for stale in ("eyes-huge.webp",):
        (OUT / stale).unlink(missing_ok=True)
    save = lambda im, name, q=88: im.save(OUT / name, "WEBP", quality=q, method=6)
    save(build_balls(), "chuzzles.webp", 84)
    save(build_huge(), "huge.webp", 80)
    save(with_alpha("Eyes.png", "Eyes_Alpha.png"), "eyes.webp", 94)
    save(with_alpha("Pupils.png", "Pupils_Alpha.png"), "pupils.webp", 94)
    save(build_frame(), "frame.webp", 90)
    save(build_cursors(), "cursors.webp", 92)
    save(build_orbs(), "orbs.webp", 92)
    save(build_candy(), "candy.webp", 92)
    for name, im in {**build_fx(), **build_scroll()}.items():
        save(im, name, 92)
    copy_sounds()
    atlas, metrics = build_font()
    atlas.save(OUT / "bauhaus36.webp", "WEBP", lossless=True)
    (ROOT / "web/font.js").write_text(
        "// web/font.js - generated by web/gen_sprites.py; do not edit.\n"
        "// The game's Bauhaus 93 bitmap font (assets/bauhaus36.webp), metrics from\n"
        "// python/deluxe_clone/tools/font/Bauhaus9336.py: glyph -> atlas rect r, offset o, advance w.\n"
        f"self.MB_FONT = {json.dumps(metrics, separators=(',', ':'))};\n", newline="\n")
    for i in range(1, 10):
        save(backdrop_tile(i), f"bkg{i}.webp", 82)
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:16} {f.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
