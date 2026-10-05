# Jerrin Shirks
# tools/texture_utils.py

# native imports
import os
from pathlib import Path

import pygame as p
from PIL import Image

# custom imports
from tools.point import Point


# https://nerdparadise.com/programming/pygameblitopacity
def blit_alpha(target, source, location, opacity):
    temp = p.Surface((source.get_width(), source.get_height()), p.SRCALPHA).convert_alpha()
    temp.fill((0, 0, 0, 0))
    temp.blit(source, (0, 0))
    temp.set_alpha(opacity)
    target.blit(temp, location.tuple())


def rotate(surface, angle, pivot=(0, 0), returnRect=True):
    rotated_image = p.transform.rotozoom(surface, -angle, 1)
    rect = rotated_image.get_rect(center=pivot)
    if returnRect:
        return rotated_image, rect
    return rotated_image


def splitTexture(texture, widthCount):
    size = Point((texture.get_width(), texture.get_height()))
    surfaces = [None] * widthCount
    width = size.x // widthCount

    for i in range(widthCount):
        surf = p.Surface((int(size.x / widthCount), size.y), p.SRCALPHA).convert_alpha()
        surf.fill((0, 0, 0, 0))

        # Normal blit. Do not use BLEND_RGBA_ADD here.
        surf.blit(texture, (0, 0), (width * i, 0, width, size.y))
        surfaces[i] = surf

    return surfaces


def _ensure_cached_dir():
    Path("cached").mkdir(exist_ok=True)


def _pil_to_pygame_rgba(im: Image.Image):
    im = im.convert("RGBA")
    return p.image.fromstring(im.tobytes(), im.size, "RGBA").convert_alpha()


def _pygame_to_pil_rgba(surface: p.Surface) -> Image.Image:
    raw_str = p.image.tostring(surface, "RGBA", False)
    return Image.frombytes("RGBA", surface.get_size(), raw_str).convert("RGBA")


def createAlpha(filename):
    """
    Creates a visible grayscale alpha texture from an alpha source.

    This is okay for textures that are themselves alpha-only visual effects.
    Do not use this to mask a color texture with BLEND_RGBA_MULT.
    Use applyAlphaMask() for that.
    """

    _ensure_cached_dir()

    new_filename = "cached/" + filename.split("/")[-1]

    if os.path.isfile(new_filename):
        return p.image.load(new_filename).convert_alpha()

    im = Image.open(filename).convert("RGBA")
    pix = im.load()

    for x in range(im.size[0]):
        for y in range(im.size[1]):
            c = pix[x, y]
            a = c[0]
            pix[x, y] = (a, a, a, a)

    im.save(new_filename)
    return _pil_to_pygame_rgba(im)


def applyAlphaMask(color_surface: p.Surface, alpha_filename: str) -> p.Surface:
    """
    Applies an alpha image to a color surface without darkening RGB.

    This fixes black/dark fringes on antialiased edges.

    Output:
        RGB = original color_surface RGB
        A   = red channel of alpha image multiplied by existing alpha
    """

    color_im = _pygame_to_pil_rgba(color_surface)
    alpha_im = Image.open(alpha_filename).convert("RGBA")

    if alpha_im.size != color_im.size:
        alpha_im = alpha_im.resize(color_im.size, Image.Resampling.LANCZOS)

    color_r, color_g, color_b, color_a = color_im.split()
    mask_r, _mask_g, _mask_b, mask_a = alpha_im.split()

    # Most PopCap alpha sheets store the useful alpha in the red/grayscale channel.
    # Preserve any existing alpha too.
    final_alpha = Image.new("L", color_im.size, 0)

    color_a_px = color_a.load()
    mask_r_px = mask_r.load()
    final_px = final_alpha.load()

    for y in range(color_im.size[1]):
        for x in range(color_im.size[0]):
            final_px[x, y] = (color_a_px[x, y] * mask_r_px[x, y]) // 255

    out = Image.merge("RGBA", (color_r, color_g, color_b, final_alpha))
    return _pil_to_pygame_rgba(out)


def convertToAlpha(surface):
    pil_image = _pygame_to_pil_rgba(surface)
    pix = pil_image.load()

    for x in range(pil_image.size[0]):
        for y in range(pil_image.size[1]):
            c = pix[x, y]
            a = c[0]
            pix[x, y] = (a, a, a, a)

    return _pil_to_pygame_rgba(pil_image)


def loadPNG(filename):
    return p.image.load(filename).convert_alpha()


def loadTexture(filename, alphaOnly=False, pngOnly=False, split=-1):
    if "_Alpha" in filename:
        alphaOnly = True
        pngOnly = False

    if alphaOnly:
        surface = createAlpha(filename)
    else:
        base = p.image.load(filename).convert_alpha()
        surface = p.Surface((base.get_width(), base.get_height()), p.SRCALPHA).convert_alpha()
        surface.fill((0, 0, 0, 0))

        # Normal blit. Do not use BLEND_RGBA_ADD for a simple copy.
        surface.blit(base, (0, 0))

        if not pngOnly:
            filename_segments = filename.split(".")
            filename_segments.insert(1, "_Alpha.")
            alpha_filename = "".join(filename_segments)

            if os.path.isfile(alpha_filename):
                surface = applyAlphaMask(surface, alpha_filename)

    if split != -1:
        return splitTexture(surface, split)

    return surface


class Textures:
    def __init__(self, parent):
        self.parent = parent
        self.p = parent.instance