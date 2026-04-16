# Jerrin Shirks

# native imports


# custom imports
from tools.menu import Menu
from tools.point import Point
from tools.image import Image
from tools.button import Button

from menus.mindbender import MindBender
from menus.away import AwayScreen


class MainMenu(Menu):
    def __init__(self, game):
        super().__init__(game)
        self.populate()

    def populate(self):
        self.textures = self.game.textures

        # background
        self.sprites.append(Image(image=self.textures.MainMenuBackground))

        # classic chuzzle button
        self.sprites.append(Button(game=self.game, coords=(40, 205),
                                   textures=self.textures.GameButton, data={
                "hitbox": self.textures.GameButton_Alpha[0],
                "sounds": self.buttonSound
            }))
        # classic chuzzle text
        self.sprites.append(Image((71, 263), self.textures.ClassicChuzzle_Menu))

        # speed chuzzle
        self.sprites.append(Button(game=self.game, coords=(195, 360),
                                   textures=self.textures.GameButton,
                                   data={
                                       "hitbox": self.textures.GameButton_Alpha[0],
                                       "sounds": self.buttonSound
                                   }))

        # speed chuzzle text
        self.sprites.append(Image((224, 421), self.textures.SpeedChuzzle_Menu))

        # game button 3/4
        self.sprites.append(Button(game=self.game, coords=(405, 360),
                                   textures=self.textures.GameButton,
                                   data={
                                       "hitbox": self.textures.GameButton_Alpha[0],
                                       "sounds": self.buttonSound,
                                       "func": {"clickup": (self.game.addMenu, AwayScreen)}
                                   }))
        # zen chuzzle text
        self.sprites.append(Image((435, 415), self.textures.ZenChuzzle_Menu))

        # mindbender button
        self.sprites.append(Button(game=self.game, coords=(560, 205),
                                   textures=self.textures.GameButton, data={
                "hitbox": self.textures.GameButton_Alpha[0],
                "sounds": self.buttonSound,
                "func": {"clickup": (self.game.addMenu, MindBender)}
            }))
        # mindbender text
        self.sprites.append(Image((590, 265), self.textures.Mindbender_Menu))

        # Options button 1/4
        self.sprites.append(Button(game=self.game, coords=(10, 420),
                                   textures=self.textures.OptionsButton, data={
                "hitbox": self.textures.OptionsButton_Alpha[0],
                "sounds": self.buttonSoundLow,
            }))
        # Highscores button 2/4
        self.sprites.append(Button(game=self.game, coords=(105, 513),
                                   textures=self.textures.HighScoresButton, data={
                "hitbox": self.textures.HighScoresButton_Alpha[0],
                "sounds": self.buttonSoundLow,
            }))

        # TrophyRoom button 3/4
        self.sprites.append(Button(game=self.game, coords=(615, 513),
                                   textures=self.textures.TrophyRoomButton, data={
                "hitbox": self.textures.TrophyRoomButton_Alpha[0],
                "sounds": self.buttonSoundLow,
            }))
        # Quit button 4/4
        self.sprites.append(Button(game=self.game, coords=(675, 420),
                                   textures=self.textures.QuitButton, data={
                "hitbox": self.textures.QuitButton_Alpha[0],
                "sounds": self.buttonSoundLow,
                "func": {"clickup": self.game.quit}
            }))
