# Jerrin Shirks

# custom imports
from tools.sprite_object import spriteObject
from tools.point import Point
from tools.timer import Timer



class SpeedBar(spriteObject):
    def __init__(self, menu, on_alert=None, on_full=None):
        super().__init__(menu)

        self.on_alert = on_alert
        self.on_full = on_full

        # 5 seconds
        self.inc_frame = 0
        self.inc_frame_count = 300

        # 1 second
        self.dec_frame = 0
        self.dec_frame_count = 60

        self.pause_frame = 0
        self.pause_frame_count = 0

        self.percentage = 0
        self.increasing = True

        self.last_frame_value = 0


    def pause_timer(self, frames=30):
        pass

    def reduce_timer(self, frames=60):
        if self.increasing:
            self.inc_frame -= frames
            if self.inc_frame < 0:
                self.inc_frame = 0

    def increment(self):
        if self.increasing:
            if self.inc_frame < self.inc_frame_count:
                self.inc_frame += 1
                self.percentage = self.inc_frame / self.inc_frame_count

            elif self.inc_frame == self.inc_frame_count:
                self.increasing = False
                self.dec_frame = self.dec_frame_count
                self.percentage = 1

                # call on_alert
                if callable(self.on_full):
                    self.on_alert()

        else:
            if self.dec_frame == 0:
                self.increasing = True
                self.inc_frame = 0
                self.percentage = 0
            else:
                self.dec_frame -= 1
                self.percentage = self.dec_frame / self.dec_frame_count

        if self.increasing:
            if self.last_frame_value < 0.67 < self.percentage:
                self.game.sounds.SpeedWarning.play()
        self.last_frame_value = self.percentage

    def render(self, screen):

        self.increment()

        newWidth = self.percentage * self.textures.SpeedBarOverlay.get_width()
        screen.blit(self.textures.SpeedBarOverlay, (255, 557), (0, 0, newWidth, self.textures.SpeedBarOverlay.get_height()))
        screen.blit(self.textures.SpeedBar, (243, 553))








