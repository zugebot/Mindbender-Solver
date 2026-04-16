# Jerrin Shirks

# native imports

# custom imports


# https://www.pygame.org/docs/ref/mixer.html#pygame.mixer.Channel


class Sounds:
    def __init__(self, parent):
        self.parent = parent
        self.p = parent.instance
        self.mixer = self.p.mixer
        self.mixer.music.set_volume(0.3)
        self.mixer.init()

        self.BadMove = self.p.mixer.Sound("sounds/BadMove.ogg")
        self.BigBang = self.p.mixer.Sound("sounds/BigBang.ogg")
        self.BigPop = self.p.mixer.Sound("sounds/BigPop.ogg")
        self.BigScore = self.p.mixer.Sound("sounds/BigScore.ogg")
        self.BoilChuzzles = self.p.mixer.Sound("sounds/BoilChuzzles.ogg")
        self.bonuslife = self.p.mixer.Sound("sounds/bonuslife.ogg")
        self.BonusTing = self.p.mixer.Sound("sounds/BonusTing.ogg")
        self.Bottled = self.p.mixer.Sound("sounds/Bottled.ogg")
        self.BreakLock = self.p.mixer.Sound("sounds/BreakLock.ogg")
        self.bubble1 = self.p.mixer.Sound("sounds/bubble1.ogg")
        self.bubble2 = self.p.mixer.Sound("sounds/bubble2.ogg")
        self.CascadePop = self.p.mixer.Sound("sounds/CascadePop.ogg")
        self.ChuzzleBump = self.p.mixer.Sound("sounds/ChuzzleBump.ogg")
        self.Chuzzled = self.p.mixer.Sound("sounds/Chuzzled.ogg")
        self.ChuzzledDoom = self.p.mixer.Sound("sounds/ChuzzledDoom.ogg")
        self.Click = self.p.mixer.Sound("sounds/Click.ogg")
        self.eyebonus = self.p.mixer.Sound("sounds/eyebonus.ogg")
        self.EyeChirp = self.p.mixer.Sound("sounds/EyeChirp.ogg")
        self.EyesEscape = self.p.mixer.Sound("sounds/EyesEscape.ogg")
        self.Flash = self.p.mixer.Sound("sounds/Flash.ogg")
        self.gameover = self.p.mixer.Sound("sounds/gameover.ogg")
        self.gameover2 = self.p.mixer.Sound("sounds/gameover2.ogg")
        self.GameSelect = self.p.mixer.Sound("sounds/GameSelect.ogg")
        self.GetTrophy = self.p.mixer.Sound("sounds/GetTrophy.ogg")
        self.Go = self.p.mixer.Sound("sounds/Go.ogg")
        self.GotRainbow = self.p.mixer.Sound("sounds/GotRainbow.ogg")
        self.HighScore = self.p.mixer.Sound("sounds/HighScore.ogg")
        self.Hilite = self.p.mixer.Sound("sounds/Hilite.ogg")
        self.HiliteLow = self.p.mixer.Sound("sounds/HiliteLow.ogg")
        self.LevelUp = self.p.mixer.Sound("sounds/LevelUp.ogg")
        self.LevelupPop = self.p.mixer.Sound("sounds/LevelupPop.ogg")
        self.LockClank = self.p.mixer.Sound("sounds/LockClank.ogg")
        self.Lockit = self.p.mixer.Sound("sounds/Lockit.ogg")
        self.ManyChuzzles = self.p.mixer.Sound("sounds/ManyChuzzles.ogg")
        self.Multipop = self.p.mixer.Sound("sounds/Multipop.ogg")
        self.NoMoreMoves = self.p.mixer.Sound("sounds/NoMoreMoves.ogg")
        self.PlugBottle = self.p.mixer.Sound("sounds/PlugBottle.ogg")
        self.Pop = self.p.mixer.Sound("sounds/Pop.ogg")
        self.Pop2 = self.p.mixer.Sound("sounds/Pop2.ogg")
        self.Pop3 = self.p.mixer.Sound("sounds/Pop3.ogg")
        self.Pop4 = self.p.mixer.Sound("sounds/Pop4.ogg")
        self.Pop5 = self.p.mixer.Sound("sounds/Pop5.ogg")
        self.popeyes = self.p.mixer.Sound("sounds/popeyes.ogg")
        self.Ready = self.p.mixer.Sound("sounds/Ready.ogg")
        self.ReleaseChuzzles = self.p.mixer.Sound("sounds/ReleaseChuzzles.ogg")
        self.Scramble = self.p.mixer.Sound("sounds/Scramble.ogg")
        self.selectpuzzle = self.p.mixer.Sound("sounds/selectpuzzle.ogg")
        self.ShakeBottle = self.p.mixer.Sound("sounds/ShakeBottle.ogg")
        self.Shuffle = self.p.mixer.Sound("sounds/Shuffle.ogg")
        self.SpeedWarning = self.p.mixer.Sound("sounds/SpeedWarning.ogg")
        self.SqueezeOut = self.p.mixer.Sound("sounds/SqueezeOut.ogg")
        self.SwitchClick = self.p.mixer.Sound("sounds/SwitchClick.ogg")
        self.Ting = self.p.mixer.Sound("sounds/Ting.ogg")
        self.Transition = self.p.mixer.Sound("sounds/Transition.ogg")
        self.TrophyEnlarge = self.p.mixer.Sound("sounds/TrophyEnlarge.ogg")
        self.TrophyJoy = self.p.mixer.Sound("sounds/TrophyJoy.ogg")
        self.TrophyShrink = self.p.mixer.Sound("sounds/TrophyShrink.ogg")
        self.Zen = self.p.mixer.Sound("sounds/Zen.ogg")

        # /Speaks
        self.cough = self.p.mixer.Sound("sounds/Speaks/cough.ogg")
        self.dizzy = self.p.mixer.Sound("sounds/Speaks/dizzy.ogg")
        self.handsoff = self.p.mixer.Sound("sounds/Speaks/handsoff.ogg")
        self.hugesneeze = self.p.mixer.Sound("sounds/Speaks/hugesneeze.ogg")
        self.hurk = self.p.mixer.Sound("sounds/Speaks/hurk.ogg")
        self.oggleoggle = self.p.mixer.Sound("sounds/Speaks/oggleoggle.ogg")
        self.pokechuzzle = self.p.mixer.Sound("sounds/Speaks/pokechuzzle.ogg")
        self.poke_boredom = self.p.mixer.Sound("sounds/Speaks/poke_boredom.ogg")
        self.PopShriek1 = self.p.mixer.Sound("sounds/Speaks/PopShriek1.ogg")
        self.PopShriek2 = self.p.mixer.Sound("sounds/Speaks/PopShriek2.ogg")
        self.PopShriek3 = self.p.mixer.Sound("sounds/Speaks/PopShriek3.ogg")
        self.PopShriek4 = self.p.mixer.Sound("sounds/Speaks/PopShriek4.ogg")
        self.PopShriek5 = self.p.mixer.Sound("sounds/Speaks/PopShriek5.ogg")
        self.PopShriek6 = self.p.mixer.Sound("sounds/Speaks/PopShriek6.ogg")
        self.smallsneeze = self.p.mixer.Sound("sounds/Speaks/smallsneeze.ogg")
        self.SweetRelief = self.p.mixer.Sound("sounds/Speaks/SweetRelief.ogg")

        # /Bonus
        self.GotJewel = self.p.mixer.Sound("sounds/Bonus/GotJewel.ogg")
        self.JewelLand = self.p.mixer.Sound("sounds/Bonus/JewelLand.ogg")


    def playSound(self, sound):
        if sound is not None:
            sound.set_volume(0.1)
            sound.play()
        



