# Jerrin Shirks

# native imports

# custom imports
from tools.game import Game

# https://www.pygame.org/docs/ref/mixer.html#pygame.mixer.Channel


class Sounds:
    def __init__(self, game: Game):
        self.game: Game = game
        self.p = game.instance
        self.mixer = self.p.mixer

        self.mixer.init()

        self.BadMove = self.loadSound("sounds/BadMove.ogg")
        self.BigBang = self.loadSound("sounds/BigBang.ogg")
        self.BigPop = self.loadSound("sounds/BigPop.ogg")
        self.BigScore = self.loadSound("sounds/BigScore.ogg")
        self.BoilChuzzles = self.loadSound("sounds/BoilChuzzles.ogg")
        self.bonuslife = self.loadSound("sounds/bonuslife.ogg")
        self.BonusTing = self.loadSound("sounds/BonusTing.ogg")
        self.Bottled = self.loadSound("sounds/Bottled.ogg")
        self.BreakLock = self.loadSound("sounds/BreakLock.ogg")
        self.bubble1 = self.loadSound("sounds/bubble1.ogg")
        self.bubble2 = self.loadSound("sounds/bubble2.ogg")
        self.CascadePop = self.loadSound("sounds/CascadePop.ogg")
        self.ChuzzleBump = self.loadSound("sounds/ChuzzleBump.ogg")
        self.Chuzzled = self.loadSound("sounds/Chuzzled.ogg")
        self.ChuzzledDoom = self.loadSound("sounds/ChuzzledDoom.ogg")
        self.Click = self.loadSound("sounds/Click.ogg")
        self.eyebonus = self.loadSound("sounds/eyebonus.ogg")
        self.EyeChirp = self.loadSound("sounds/EyeChirp.ogg")
        self.EyesEscape = self.loadSound("sounds/EyesEscape.ogg")
        self.Flash = self.loadSound("sounds/Flash.ogg")
        self.gameover = self.loadSound("sounds/gameover.ogg")
        self.gameover2 = self.loadSound("sounds/gameover2.ogg")
        self.GameSelect = self.loadSound("sounds/GameSelect.ogg")
        self.GetTrophy = self.loadSound("sounds/GetTrophy.ogg")
        self.Go = self.loadSound("sounds/Go.ogg")
        self.GotRainbow = self.loadSound("sounds/GotRainbow.ogg")
        self.HighScore = self.loadSound("sounds/HighScore.ogg")
        self.Hilite = self.loadSound("sounds/Hilite.ogg")
        self.HiliteLow = self.loadSound("sounds/HiliteLow.ogg")
        self.LevelUp = self.loadSound("sounds/LevelUp.ogg")
        self.LevelupPop = self.loadSound("sounds/LevelupPop.ogg")
        self.LockClank = self.loadSound("sounds/LockClank.ogg")
        self.Lockit = self.loadSound("sounds/Lockit.ogg")
        self.ManyChuzzles = self.loadSound("sounds/ManyChuzzles.ogg")
        self.Multipop = self.loadSound("sounds/Multipop.ogg")
        self.NoMoreMoves = self.loadSound("sounds/NoMoreMoves.ogg")
        self.PlugBottle = self.loadSound("sounds/PlugBottle.ogg")
        self.Pop = self.loadSound("sounds/Pop.ogg")
        self.Pop2 = self.loadSound("sounds/Pop2.ogg")
        self.Pop3 = self.loadSound("sounds/Pop3.ogg")
        self.Pop4 = self.loadSound("sounds/Pop4.ogg")
        self.Pop5 = self.loadSound("sounds/Pop5.ogg")
        self.popeyes = self.loadSound("sounds/popeyes.ogg")
        self.Ready = self.loadSound("sounds/Ready.ogg")
        self.ReleaseChuzzles = self.loadSound("sounds/ReleaseChuzzles.ogg")
        self.Scramble = self.loadSound("sounds/Scramble.ogg")
        self.selectpuzzle = self.loadSound("sounds/selectpuzzle.ogg")
        self.ShakeBottle = self.loadSound("sounds/ShakeBottle.ogg")
        self.Shuffle = self.loadSound("sounds/Shuffle.ogg")
        self.SpeedWarning = self.loadSound("sounds/SpeedWarning.ogg")
        self.SqueezeOut = self.loadSound("sounds/SqueezeOut.ogg")
        self.SwitchClick = self.loadSound("sounds/SwitchClick.ogg")
        self.Ting = self.loadSound("sounds/Ting.ogg")
        self.Transition = self.loadSound("sounds/Transition.ogg")
        self.TrophyEnlarge = self.loadSound("sounds/TrophyEnlarge.ogg")
        self.TrophyJoy = self.loadSound("sounds/TrophyJoy.ogg")
        self.TrophyShrink = self.loadSound("sounds/TrophyShrink.ogg")
        self.Zen = self.loadSound("sounds/Zen.ogg")

        # /Speaks
        self.cough = self.loadSound("sounds/Speaks/cough.ogg")
        self.dizzy = self.loadSound("sounds/Speaks/dizzy.ogg")
        self.handsoff = self.loadSound("sounds/Speaks/handsoff.ogg")
        self.hugesneeze = self.loadSound("sounds/Speaks/hugesneeze.ogg")
        self.hurk = self.loadSound("sounds/Speaks/hurk.ogg")
        self.oggleoggle = self.loadSound("sounds/Speaks/oggleoggle.ogg")
        self.pokechuzzle = self.loadSound("sounds/Speaks/pokechuzzle.ogg")
        self.poke_boredom = self.loadSound("sounds/Speaks/poke_boredom.ogg")
        self.PopShriek1 = self.loadSound("sounds/Speaks/PopShriek1.ogg")
        self.PopShriek2 = self.loadSound("sounds/Speaks/PopShriek2.ogg")
        self.PopShriek3 = self.loadSound("sounds/Speaks/PopShriek3.ogg")
        self.PopShriek4 = self.loadSound("sounds/Speaks/PopShriek4.ogg")
        self.PopShriek5 = self.loadSound("sounds/Speaks/PopShriek5.ogg")
        self.PopShriek6 = self.loadSound("sounds/Speaks/PopShriek6.ogg")
        self.smallsneeze = self.loadSound("sounds/Speaks/smallsneeze.ogg")
        self.SweetRelief = self.loadSound("sounds/Speaks/SweetRelief.ogg")

        # /Bonus
        self.GotJewel = self.loadSound("sounds/Bonus/GotJewel.ogg")
        self.JewelLand = self.loadSound("sounds/Bonus/JewelLand.ogg")

        self.applyVolumes()

    def clampVolume(self, volume):
        return max(0.0, min(1.0, float(volume)))

    def getFXVolume(self):
        return self.clampVolume(getattr(self.game.settings, "FXVolume", 1.0))

    def getMusicVolume(self):
        return self.clampVolume(getattr(self.game.settings, "musicVolume", self.getFXVolume()))

    def loadSound(self, path):
        sound = self.p.mixer.Sound(path)
        sound.set_volume(self.getFXVolume())
        return sound

    def applyVolumes(self):
        fx_volume = self.getFXVolume()
        music_volume = self.getMusicVolume()

        for value in self.__dict__.values():
            if isinstance(value, self.p.mixer.Sound):
                value.set_volume(fx_volume)

        self.mixer.music.set_volume(music_volume)

    def playSound(self, sound, volume=1.0):
        if sound is None:
            return None

        final_volume = self.getFXVolume() * self.clampVolume(volume)

        if final_volume <= 0.0:
            return None

        sound.set_volume(final_volume)
        return sound.play()