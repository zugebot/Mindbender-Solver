# Jerrin Shirks

# native imports
import pygame as p
from PIL import Image
import os

# custom imports
from tools.point import Point


# https://nerdparadise.com/programming/pygameblitopacity
def blit_alpha(target, source, location, opacity):
    temp = p.Surface((source.get_width(), source.get_height())).convert_alpha()
    temp.fill((0, 0, 0, 0))
    temp.blit(source, (0, 0))
    temp.set_alpha(opacity)
    target.blit(temp, location.tuple())

def rotate(surface, angle, pivot=(0, 0), returnRect=True):
    rotated_image = p.transform.rotozoom(surface, -angle, 1)
    rect = rotated_image.get_rect(center=pivot)
    if returnRect:
        return rotated_image, rect
    else:
        return rotated_image

def splitTexture(texture, widthCount):
    size = Point((texture.get_width(), texture.get_height()))
    surfaces = [None] * widthCount
    width = size.x // widthCount
    for i in range(widthCount):
        surf = p.Surface((int(size.x / widthCount), size.y), p.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        surf.blit(texture, (0, 0), (width * i, 1, width, size.y), p.BLEND_RGBA_ADD)
        surfaces[i] = surf
    return surfaces


def createAlpha(filename):
    new_filename = "cached/" + filename.split("/")[-1]

    # check to see if the file has already been created
    if os.path.isfile(new_filename):
        return p.image.load(new_filename)

    # if the cached folder isn't already there make it
    if not os.path.isdir("../cached"):
        os.mkdir("../cached")

    old_filename = filename
    im = Image.open(old_filename)
    im = im.convert("RGBA")
    pix = im.load()
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            c = pix[x, y]
            pix[x, y] = (c[0], c[0], c[0], c[0])
    im.save(new_filename)

    mode = im.mode
    size = im.size
    data = im.tobytes()
    py_im = p.image.fromstring(data, size, mode)
    return py_im

def convertToAlpha(surface):

    raw_str = p.image.tostring(surface, "RGBA", False)
    pil_image = Image.frombytes("RGBA", surface.get_size(), raw_str)
    # pil_image = Image.fromstring("RGBA", (660, 660), pil_string_image)
    pix = pil_image.load()

    for x in range(pil_image.size[0]):
        for y in range(pil_image.size[1]):
            c = pix[x, y]
            pix[x, y] = (c[0], c[0], c[0], c[0])
    mode = pil_image.mode
    size = pil_image.size
    data = pil_image.tobytes()
    py_im = p.image.fromstring(data, size, mode)
    return py_im


def loadPNG(filename):
    base = p.image.load(filename).convert_alpha()
    return base


def loadTexture(filename, alphaOnly=False, pngOnly=False, split=-1):

    if "_Alpha" in filename:
        alphaOnly = True
        pngOnly = False

    if alphaOnly:
        base = createAlpha(filename)
    else:
        base = p.image.load(filename).convert_alpha()


    surface = p.Surface((base.get_width(), base.get_height()), p.SRCALPHA)
    surface.fill((0, 0, 0, 0))

    surface.blit(base, (0, 0), None, p.BLEND_RGBA_ADD)

    if not alphaOnly and not pngOnly:
        filename_segments = filename.split(".")
        filename_segments.insert(1, "_Alpha.")
        alpha_filename = "".join(filename_segments)
        base_alpha = createAlpha(alpha_filename)
        surface.blit(base_alpha, (0, 0), None, p.BLEND_RGBA_MULT)

    if split != -1:
        return splitTexture(surface, split)
    else:
        return surface

class Textures:
    def __init__(self, parent):
        self.parent = parent
        self.p = parent.instance
