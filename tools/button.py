# Jerrin Shirks

# native imports
import pygame as p
from tools.point import Point
from tools.texture_utils import rotate, convertToAlpha


class Button:
    def __init__(self,
                 game,
                 coords=Point(),
                 textures=None,
                 data=None,
                 clickable=True):

        self.game = game

        if isinstance(coords, tuple):
            coords = Point(coords)
        self.coords = coords

        self.data = {} if data is None else data
        self.fixData()
        if textures is not None:
            self.data["textures"]["default"] = textures[0]
            self.data["textures"]["hover"] = textures[1]
            self.data["textures"]["clickdown"] = textures[2]

        self.clickable = clickable

        self.clickedDownBeforeHover = False
        self.lastFrameTouching = False
        self.mouseDownStartedInside = False

        # pre-scale image sizes
        if "render" in self.data:
            if "scale" in self.data["render"]:
                scale = self.data["render"]["scale"]

                # multiply image size by number
                if "multiply" in scale:
                    newSize = Point(self.image.get_size()) * scale["multiply"]
                    self.data["textures"]["default"] = p.transform.scale(self.image, newSize.tuple())

                # set height to value, force width to stay in ratio
                elif "set-H" in scale:
                    oldSize = Point(self.data["textures"]["default"].get_size())
                    if scale["set-H"] == "screen-size":
                        scale = self.game.screenSize.y / oldSize.y
                    else:
                        scale = scale["set-H"] / oldSize.y
                    newSize = (oldSize * float(scale)).tuple()

                    self.data["textures"]["default"] = p.transform.scale(self.data["textures"]["default"], newSize)

        self.image = self.data["textures"]["default"]
        self.size = Point((self.image.get_width(), self.image.get_height()))
        self.rect = self.image.get_rect(center=(coords + self.size / 2).tuple())

        self.fixData()

    def setDefault(self):
        self.image = self.data["textures"]["default"]
        self.clickedDownBeforeHover = False
        self.lastFrameTouching = False
        self.mouseDownStartedInside = False

    def fixData(self):
        # attributes
        self.data.setdefault("draggable", False)

        # textures
        self.data.setdefault("textures", {})
        self.data["textures"].setdefault("default", None)
        self.data["textures"].setdefault("clickup", None)
        self.data["textures"].setdefault("clickdown", None)
        self.data["textures"].setdefault("hover", None)
        if "hitbox" not in self.data and self.clickable:
            self.data["hitbox"] = self.data["textures"]["default"]

        # rendering tags
        self.data.setdefault("render", {})
        self.data["render"].setdefault("type", None)
        self.data["render"].setdefault("clip-width", None)

        # rotating flags
        if "rot" in self.data["render"]:
            self.data["render"].setdefault("rot-frame", 0)
            self.data["render"].setdefault("rot-frame-count", 0)
            self.data["render"].setdefault("rot-clockwise", 1)

        # sounds
        self.data.setdefault("sounds", {})
        self.data["sounds"].setdefault("clickup", None)
        self.data["sounds"].setdefault("clickdown", None)
        self.data["sounds"].setdefault("hover", None)

        # functions tied to it
        self.data.setdefault("func", {})
        self.data["func"].setdefault("clickup", None)
        self.data["func"].setdefault("clickdown", None)
        self.data["func"].setdefault("hover", None)

    def _is_touching(self, mousePos):
        if not self.clickable:
            return False

        if not self.rect.collidepoint(*mousePos):
            return False

        pos_in_mask = mousePos[0] - self.rect.x, mousePos[1] - self.rect.y

        if isinstance(self.data["hitbox"], str):
            return self.data["hitbox"] == "all"

        return bool(self.data["hitbox"].get_at(pos_in_mask))

    def testMouseStates(self, mousePos, mouseState):
        if not self.clickable:
            return

        touching = self._is_touching(mousePos)
        pressed_visual = False

        if touching:
            self.game.cursorIndex = 2
            if not self.lastFrameTouching and self.data["sounds"]["hover"]:
                self.game.sounds.playSound(self.data["sounds"]["hover"])
            self.lastFrameTouching = True
        else:
            if self.lastFrameTouching:
                self.game.cursorIndex = 0
            self.lastFrameTouching = False

        if mouseState == 1:  # MOUSE_CLICK_DOWN
            self.mouseDownStartedInside = touching
            self.clickedDownBeforeHover = not touching

            if touching:
                if self.data["sounds"]["clickdown"]:
                    self.game.sounds.playSound(self.data["sounds"]["clickdown"])

                if self.data["func"]["clickdown"]:
                    dat = self.data["func"]["clickdown"]
                    if type(dat) == tuple:
                        dat[0](dat[1])
                    else:
                        dat()

        elif mouseState == -1:  # MOUSE_HELD_DOWN
            pressed_visual = self.mouseDownStartedInside
            if touching and not self.mouseDownStartedInside:
                self.clickedDownBeforeHover = True

        elif mouseState == 0:  # MOUSE_HOVER
            pass

        elif mouseState == -2:  # MOUSE_CLICK_UP
            should_activate = touching and self.mouseDownStartedInside

            if should_activate:
                if self.data["sounds"]["clickup"]:
                    self.game.sounds.playSound(self.data["sounds"]["clickup"])

                if self.data["func"]["clickup"]:
                    dat = self.data["func"]["clickup"]
                    if type(dat) == tuple:
                        dat[0](dat[1])
                    else:
                        dat()

            self.mouseDownStartedInside = False
            self.clickedDownBeforeHover = False
            pressed_visual = False

        if pressed_visual:
            if self.data["textures"]["clickdown"]:
                self.image = self.data["textures"]["clickdown"]
            elif touching and self.data["textures"]["hover"]:
                self.image = self.data["textures"]["hover"]
            else:
                self.image = self.data["textures"]["default"]
        else:
            if touching:
                if self.data["textures"]["hover"]:
                    self.image = self.data["textures"]["hover"]
                else:
                    self.image = self.data["textures"]["default"]
            else:
                self.image = self.data["textures"]["default"]

    def render(self, screen, image=None, default=False):
        if default:
            self.setDefault()

        if image is None:
            data = self.data
            image = self.image.copy()
        else:
            data = image.data
            image = image.image
        rect = self.rect

        if "render" in data:
            render = data["render"]

            # rotate an image around a point
            if "rot" in render:
                rot = render["rot"]
                angle = 360 * rot["frame"] / rot["frame-count"]
                data["render"]["apply"]["frame"] = (rot["frame"] + rot["clockwise"]) % rot["frame-count"]
                size = Point((image.get_width(), image.get_height()))
                image, rect = rotate(image, angle, pivot=(size / 2).tuple())

            # make the image move panoramically left or right.
            if "panorama" in render:
                panorama = render["panorama"]

                if "left" not in panorama:
                    if panorama["frame"] > panorama["frame-count"]:
                        data["render"]["panorama"]["frame"] = 0
                    else:
                        data["render"]["panorama"]["frame"] += 1
                else:
                    if panorama["frame"] == 0:
                        data["render"]["panorama"]["frame"] = panorama["frame-count"] - 1
                    else:
                        data["render"]["panorama"]["frame"] -= 1

                if panorama["frame"] != 0:
                    loc1 = (panorama["frame"] / panorama["frame-count"])
                    screen.blit(image, (image.get_width() * loc1, 0))
                    screen.blit(image, (image.get_width() * (loc1 - 1), 0))
                    return image

            if "convert" in render:
                if data["render"]["convert"] == "alpha":
                    image = convertToAlpha(image)

            if render["clip-width"] not in [None, 1]:
                newWidth = image.get_width() * data["render"]["clip-width"]
                cropped = p.Surface((newWidth, image.get_height()))
                cropped.blit(image, (0, 0), (0, 0, newWidth, image.get_height()))
                image = cropped

            if render["type"] == "ADD":
                screen.blit(image, rect, None, p.BLEND_RGBA_ADD)
            elif render["type"] == "MULT":
                screen.blit(image, rect, None, p.BLEND_RGBA_MULT)
            elif render["type"] is None:
                screen.blit(image, rect)

        return image