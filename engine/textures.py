# Jerrin Shirks

# native imports


# custom imports
from tools.game import Game
from tools.texture_utils import *



class Textures:
    def __init__(self, game: Game):
        self.game: Game = game
        self.p = game.instance

        self.ChuzzleBase = [None] * 9
        self.ChuzzleHuge = [None] * 9

        self.bkg1 = self.loadTexture("textures/Backdrops/bkg1.jpg", pngOnly=True)
        self.bkg2 = self.loadTexture("textures/Backdrops/bkg2.jpg", pngOnly=True)
        self.bkg3 = self.loadTexture("textures/Backdrops/bkg3.jpg", pngOnly=True)
        self.bkg4 = self.loadTexture("textures/Backdrops/bkg4.jpg", pngOnly=True)
        self.bkg5 = self.loadTexture("textures/Backdrops/bkg5.jpg", pngOnly=True)
        self.bkg6 = self.loadTexture("textures/Backdrops/bkg6.jpg", pngOnly=True)
        self.bkg7 = self.loadTexture("textures/Backdrops/bkg7.jpg", pngOnly=True)
        self.bkg8 = self.loadTexture("textures/Backdrops/bkg8.jpg", pngOnly=True)
        self.bkg9 = self.loadTexture("textures/Backdrops/bkg9.jpg", pngOnly=True)
        self.AngryEyes = self.loadTexture("textures/AngryEyes.png", pngOnly=True)
        self.AreYouSure = self.loadTexture("textures/AreYouSure.png")
        self.Beauty = self.loadTexture("textures/Beauty.png")
        self.Border = self.loadTexture("textures/Border.png")
        self.Bottle = self.loadTexture("textures/Bottle.png")
        self.BottleFill = self.loadTexture("textures/BottleFill_Alpha.png", alphaOnly=True)
        self.Burst = self.loadTexture("textures/Burst_Alpha.png", alphaOnly=True)
        self.BuyChuzzle = self.loadTexture("textures/BuyChuzzle.png")
        self.Candy = self.loadTexture("textures/Candy_Alpha.png", alphaOnly=True)
        self.CasualButton = self.loadTexture("textures/CasualButton.png", split=3)
        self.chaBoom = self.loadTexture("textures/chaBoom.png")
        self.changeProfile = self.loadTexture("textures/changeProfile_Alpha.png", alphaOnly=True)
        self.ChuzzleCheeks = self.loadTexture("textures/ChuzzleCheeks.png")
        self.ChuzzleMouths = self.loadTexture("textures/ChuzzleMouths.png", split=16)
        self.ChuzzOutline = self.loadTexture("textures/ChuzzOutline_Alpha.png", alphaOnly=True)
        self.ClassicChuzzle_Menu = self.loadTexture("textures/ClassicChuzzle_Menu.png")
        self.ClickHereNextPuzzle = self.loadTexture("textures/ClickHereNextPuzzle.png", split=3)
        self.ClickHereNextPuzzle_Alpha = self.loadTexture("textures/ClickHereNextPuzzle_Alpha.png", alphaOnly=True, split=3)
        self.ClickHereToPlay = self.loadTexture("textures/ClickHereToPlay.png", split=3)
        self.ClickHereToPlayAgain = self.loadTexture("textures/ClickHereToPlayAgain.png", split=3)
        self.ClickHereToPlayAgain_Alpha = self.loadTexture("textures/ClickHereToPlayAgain_Alpha.png", alphaOnly=True, split=3)
        self.ClickToContinue = self.loadTexture("textures/ClickToContinue.png")
        self.ColorblindSymbols = self.loadTexture("textures/ColorblindSymbols.png", split=8)
        self.ColorblindSymbolsSmall = self.loadTexture("textures/ColorblindSymbolsSmall.png", split=7)
        self.Combo = self.loadTexture("textures/Combo.png")
        self.ContinueGameButton = self.loadTexture("textures/ContinueGameButton.png", pngOnly=True, split=3)
        self.Cork = self.loadTexture("textures/Cork.png")
        self.Cursors = self.loadTexture("textures/Cursors.png", pngOnly=True, split=2) # might need to finetune this one
        self.customCursors = self.loadTexture("textures/CustomCursors.png", split=5)
        self.DialogBox = self.loadTexture("textures/DialogBox.png")
        self.Difficulty = self.loadTexture("textures/Difficulty.png")
        self.DontShowThisAgain = self.loadTexture("textures/DontShowThisAgain.png", split=3)
        self.DragFrom = self.loadTexture("textures/DragFrom.png")
        self.DragRainbowV = self.loadTexture("textures/DragRainbow.png")
        self.DragRainbowH = rotate(self.DragRainbowV, 90, returnRect=False)
        self.DragTo = self.loadTexture("textures/DragTo.png")
        self.ExpertButton = self.loadTexture("textures/ExpertButton.png", split=3)
        self.Eyes = self.loadTexture("textures/Eyes.png", split=4)
        self.FizzyStar = self.loadTexture("textures/FizzyStar_Alpha.png", alphaOnly=True)
        self.FuzzAlpha = self.loadTexture("textures/Fuzz_Alpha.png", alphaOnly=True)
        self.GameButton = self.loadTexture("textures/GameButton.png", split=3)
        self.GameButton_Alpha = self.loadTexture("textures/GameButton_Alpha.png", split=3, alphaOnly=True)
        self.GameCredits = self.loadTexture("textures/GameCredits.png")
        self.GameModeTitle = self.loadTexture("textures/GameModeTitle.png", split=5, split_vertical=True)
        self.GameOptions = self.loadTexture("textures/GameOptions.png")
        self.GameOver = self.loadTexture("textures/GameOver.png")
        self.GameOverCircle = self.loadTexture("textures/GameOverCircle.png")
        self.GameSaved = self.loadTexture("textures/GameSaved.png")
        self.GM_MainMenu = self.loadTexture("textures/GM_MainMenu.png", split=3)
        self.GM_Options = self.loadTexture("textures/GM_Options.png", split=3)
        self.GM_PlayAgain = self.loadTexture("textures/GM_PlayAgain.png", split=3)
        self.GM_ReturnToGame = self.loadTexture("textures/GM_ReturnToGame.png", split=3)
        self.Go = self.loadTexture("textures/Go.png")
        self.HandIcon = self.loadTexture("textures/HandIcon.png")
        self.HighScore_Alpha = self.loadTexture("textures/HighScore_Alpha.png", alphaOnly=True)
        self.HighScoreHeader_Alpha = self.loadTexture("textures/HighScoreHeader.png")
        self.HighScoresButton = self.loadTexture("textures/HighScoresButton.png", split=3)
        self.HighScoresButton_Alpha = self.loadTexture("textures/HighScoresButton_Alpha.png", split=3, alphaOnly=True)
        self.HighScoreScreenBottom = self.loadTexture("textures/HighScoreScreenBottom.png", pngOnly=True)
        self.HighScoreScreenTop = self.loadTexture("textures/HighScoreScreenTop.png")
        self.HintArrow_Alpha = self.loadTexture("textures/HintArrow_Alpha.png", alphaOnly=True, split=4)
        self.HintButton = self.loadTexture("textures/HintButton.png", split=3)
        self.HintButton_Halo_Alpha = self.loadTexture("textures/HintButton_Halo_Alpha.png", alphaOnly=True)
        self.HintCircle_Alpha = self.loadTexture("textures/HintCircle_Alpha.png", alphaOnly=True)
        self.HintCircleBig_Alpha = self.loadTexture("textures/HintCircleBig_Alpha.png", alphaOnly=True)
        self.HowToPlayBG = self.loadTexture("textures/HowToPlayBG.png", pngOnly=True)
        self.Instructions = self.loadTexture("textures/Instructions.png")
        # figure out how to render this
        self.LevelFont_Color = self.loadTexture("textures/LevelFont_Color.png", split=10)
        self.LevelFont_Outline_Alpha = self.loadTexture("textures/LevelFont_Outline_Alpha.png", alphaOnly=True, split=10)
        self.LevelStar = self.loadTexture("textures/LevelStar.png")
        self.LevelupBonus = self.loadTexture("textures/LevelupBonus.png")
        self.LevelUpDisplay = self.loadTexture("textures/LevelUpDisplay.png")
        self.LevelupDoors = self.loadTexture("textures/LevelupDoors.png", pngOnly=True)
        self.LevelupLight_Alpha = self.loadTexture("textures/LevelupLight_Alpha.png", alphaOnly=True)
        self.Lock = self.loadTexture("textures/Lock.png")
        self.LockGlint_Alpha = self.loadTexture("textures/LockGlint_Alpha.png", alphaOnly=True)
        self.MainMenuBackground = self.loadTexture("textures/MainMenuBackground.png", pngOnly=True)
        self.MainMenuButton = self.loadTexture("textures/MainMenuButton.png", split=3)
        self.MainMenuButton_Alpha = self.loadTexture("textures/MainMenuButton_Alpha.png", split=3, alphaOnly=True)
        self.MainMenuContext = self.loadTexture("textures/MainMenuContext.png", split=4)
        self.MainMenuWelcome = self.loadTexture("textures/MainMenuWelcome.png")
        self.MainMenuWelcomeBack = self.loadTexture("textures/MainMenuWelcomeBack.png")
        self.Marquee = self.loadTexture("textures/Marquee.png")
        self.Matched = self.loadTexture("textures/Matched.png")
        self.MatchThis = self.loadTexture("textures/MatchThis.png", split=5)
        self.Menu = self.loadTexture("textures/Menu.png")
        self.MessageDot = self.loadTexture("textures/MessageDot.png")
        self.MindbBenderMini_Select_Base = self.loadTexture("textures/MindbBenderMini_Select.png", pngOnly=True)
        self.MindBenderMini_Select_Alpha = self.loadTexture("textures/MindBenderMini_Select_Alpha.png", alphaOnly=True)
        self.MindbBenderMini_Select = self.MindbBenderMini_Select_Base.copy()
        self.MindbBenderMini_Select.blit(self.MindBenderMini_Select_Alpha, (0, 0), None, p.BLEND_RGBA_MULT)
        self.Mindbender_Menu = self.loadTexture("textures/Mindbender_Menu.png")
        self.MindBenderBigStar = self.loadTexture("textures/MindBenderBigStar.png")
        self.MindBenderGadget = self.loadTexture("textures/MindBenderGadget.png")
        self.MindbenderGlint1 = self.loadTexture("textures/MindbenderGlint1.png")
        self.MindbenderGlint2 = self.loadTexture("textures/MindbenderGlint2.png")
        self.MindbenderLevelup_Alpha = self.loadTexture("textures/MindbenderLevelup_Alpha.png", alphaOnly=True)
        self.MindBenderMarquee = self.loadTexture("textures/MindBenderMarquee.png")
        self.MindBenderMini = self.loadTexture("textures/MindBenderMini.png", split=3)
        self.MindBenderMini_Alpha = self.loadTexture("textures/MindBenderMini_Alpha.png", split=3, alphaOnly=True)
        self.MindBenderMini_Check = self.loadTexture("textures/MindBenderMini_Check.png")
        self.MindBenderMini_Select_Alpha = self.loadTexture("textures/MindBenderMini_Select_Alpha.png", alphaOnly=True)
        self.MindBenderSmallStarLeft = self.loadTexture("textures/MindBenderSmallStarLeft.png", split=3)
        self.MindBenderSmallStarLeft_Alpha = self.loadTexture("textures/MindBenderSmallStarLeft_Alpha.png", split=3, alphaOnly=True)
        self.MindBenderSmallStarRight = self.loadTexture("textures/MindBenderSmallStarRight.png", split=3)
        self.MindBenderSmallStarRight_Alpha = self.loadTexture("textures/MindBenderSmallStarRight_Alpha.png", split=3, alphaOnly=True)
        self.MindBenderSolvedSmall = self.loadTexture("textures/MindBenderSolvedSmall.png")
        self.MiniChuzzles = self.loadTexture("textures/MiniChuzzles.png", split=9)
        self.MiniEyes = self.loadTexture("textures/MiniEyes.png", split=4)
        self.NakedChuzzle = self.loadTexture("textures/NakedChuzzle.png")
        self.NewGameButton = self.loadTexture("textures/NewGameButton.png", split=3)
        self.NewProfile = self.loadTexture("textures/NewProfile.png")
        self.NoButton = self.loadTexture("textures/NoButton.png", pngOnly=True, split=3)
        self.OKButton = self.loadTexture("textures/OKButton.png", split=3)
        self.OKButtonAlpha = self.loadTexture("textures/OKButton_Alpha.png", alphaOnly=True, split=3)
        self.Oops = self.loadTexture("textures/Oops.png")
        self.OptionCheckBox = self.loadTexture("textures/OptionCheckBox.png", split=2)
        self.OptionDisplayBox_Alpha = self.loadTexture("textures/OptionDisplayBox_Alpha.png", alphaOnly=True)
        self.Options_BuyIt = self.loadTexture("textures/Options_BuyIt.png", split=3)
        self.Options_Credits = self.loadTexture("textures/Options_Credits.png", split=3)
        self.Options_Done = self.loadTexture("textures/Options_Done.png", split=3)
        self.Options_Help = self.loadTexture("textures/Options_Help.png", split=3)
        self.Options_Update = self.loadTexture("textures/Options_Update.png", split=3)
        self.OptionsButton = self.loadTexture("textures/OptionsButton.png", split=3)
        self.OptionsButton_Alpha = self.loadTexture("textures/OptionsButton_Alpha.png", alphaOnly=True, split=3)
        self.OptionSliderBar = self.loadTexture("textures/OptionSliderBar.png")
        self.ProfileBox_Alpha = self.loadTexture("textures/ProfileBox_Alpha.png", alphaOnly=True)
        self.Profiles = self.loadTexture("textures/Profiles.png")
        self.Profiles_Accept = self.loadTexture("textures/Profiles_Accept.png", split=3)
        self.Profiles_Cancel = self.loadTexture("textures/Profiles_Cancel.png", split=3)
        self.Profiles_Delete = self.loadTexture("textures/Profiles_Delete.png", split=3)
        self.Profiles_New = self.loadTexture("textures/Profiles_New.png", split=3)
        self.Pupils = self.loadTexture("textures/Pupils.png", split=6)
        self.QuitButton = self.loadTexture("textures/QuitButton.png", split=3)
        self.QuitButton_Alpha = self.loadTexture("textures/QuitButton_Alpha.png", alphaOnly=True, split=3)
        self.Rainbow = self.loadTexture("textures/Rainbow.png", pngOnly=True)
        self.RainbowBonus = self.loadTexture("textures/RainbowBonus.png")
        self.RainbowChaser = self.loadTexture("textures/RainbowChaser.png", pngOnly=True)
        # used in other function
        self.RainbowChuzzle = self.loadTexture("textures/RainbowChuzzle.png", split=60)
        self.RainbowChuzzle_Alpha = self.loadTexture("textures/RainbowChuzzle_Alpha.png", split=60, alphaOnly=True)
        self.RainbowSplode = self.loadTexture("textures/RainbowSplode.png")
        self.Ready = self.loadTexture("textures/Ready.png")
        self.ReallyQuit = self.loadTexture("textures/ReallyQuit.png")
        self.ReleaseRainbow = self.loadTexture("textures/ReleaseRainbow.png")
        self.RetroCircle_Alpha = self.loadTexture("textures/RetroCircle_Alpha.png", alphaOnly=True)
        self.RosyCheeks_Alpha = self.loadTexture("textures/RosyCheeks_Alpha.png", alphaOnly=True)
        self.SadSadBlink = self.loadTexture("textures/SadSadBlink.png")
        self.SadSadChuzzle = self.loadTexture("textures/SadSadChuzzle.png")
        self.SadSadTears = self.loadTexture("textures/SadSadTears.png")
        self.ScoreUp = self.loadTexture("textures/ScoreUp.png", split=10)
        self.Scrambles = self.loadTexture("textures/Scrambles.png")
        self.ScrambleStar = self.loadTexture("textures/ScrambleStar.png")
        self.Scrambulator_Arrow = self.loadTexture("textures/Scrambulator_Arrow.png", split=5)
        self.Scrambulator_Button = self.loadTexture("textures/Scrambulator_Button.png", split=3)
        self.Scrambulator_Font = self.loadTexture("textures/Scrambulator_Font.png", split=5)
        self.ScrambulatorButtonBG = self.loadTexture("textures/ScrambulatorButtonBG.png")
        self.ScrambulatorDot = self.loadTexture("textures/ScrambulatorDot.png")
        self.ScrollDown = self.loadTexture("textures/ScrollDown.png", split=3)
        self.ScrollUp = self.loadTexture("textures/ScrollUp.png", split=3)
        self.Shockwave = self.loadTexture("textures/Shockwave.png")
        self.SleepyEyes = self.loadTexture("textures/SleepyEyes.png")
        self.SleepyZ = self.loadTexture("textures/SleepyZ.png")
        self.SliderThumb = self.loadTexture("textures/SliderThumb.png")
        self.SodaPop_Alpha = self.loadTexture("textures/SodaPop_Alpha.png", alphaOnly=True)
        self.SpeedBar = self.loadTexture("textures/SpeedBar.png")
        self.SpeedBarOverlay = self.loadTexture("textures/SpeedBarOverlay.png", pngOnly=True)
        self.SpeedChuzzle_Menu = self.loadTexture("textures/SpeedChuzzle_Menu.png")
        self.SpeedWarning_Alpha = self.loadTexture("textures/SpeedWarning_Alpha.png", alphaOnly=True)
        self.Spiral = self.loadTexture("textures/Spiral.png", pngOnly=True)
        self.Star_Alpha = self.loadTexture("textures/Star_Alpha.png", alphaOnly=True)
        self.title = self.loadTexture("textures/title.png", pngOnly=True)
        self.Title_Chuzzle = self.loadTexture("textures/Title_Chuzzle.png")
        self.TitleScreen_Loadbar2 = self.loadTexture("textures/TitleScreen_Loadbar2.png")
        self.TitleScreen_Loading = self.loadTexture("textures/TitleScreen_Loading.png")
        self.TitleScreen_Play = self.loadTexture("textures/TitleScreen_Play.png")
        self.TransitionStar1 = self.loadTexture("textures/TransitionStar1.png")
        self.TransitionStar2 = self.loadTexture("textures/TransitionStar2.png")
        self.TrophyRoomBG = self.loadTexture("textures/TrophyRoomBG.png", pngOnly=True)
        self.TrophyRoomButton = self.loadTexture("textures/TrophyRoomButton.png", split=3)
        self.TrophyRoomButton_Alpha = self.loadTexture("textures/TrophyRoomButton.png", alphaOnly=True, split=3)
        self.Update = self.loadTexture("textures/Update.png")
        self.UpdateArrow = self.loadTexture("textures/UpdateArrow.png", pngOnly=True)
        self.Wince = self.loadTexture("textures/Wince.png")
        self.WonderGlare = self.loadTexture("textures/WonderGlare.png")
        self.YesButton = self.loadTexture("textures/YesButton.png", split=3)
        self.ZenChuzzle_Menu = self.loadTexture("textures/ZenChuzzle_Menu.png")
        self.ZenLight = self.loadTexture("textures/ZenLight.png")

        # (255, 255, 0),  # yellow
        self.colorsMain = {
            "0": {"color": (178, 8, 8)},  # red
            "1": {"color": (8, 174, 8)},  # green
            "2": {"color": (8, 8, 255)},  # blue
            "3": {"color": (255, 105, 2)},  # orange
            "4": {
                "texture": {
                    "base": "textures/Chuzzle-Yellow.png",
                    "alpha": "cached/Chuzzle-Base_Alpha.png"
                },
                "fat_color": (255, 255, 0)
            },
            # yellow
            "5": {"color": (207, 7, 207)},  # purple
            "6": {"color": (255, 255, 255)},  # white
            "7": {"color": (87, 197, 221)},  # cyan
            "8": {"texture": {"base": "textures/RainbowChuzzle.png", "alpha": "cached/RainbowChuzzle_Alpha.png"}},
            # rainbow
            "9": {"name": "cinnamon", "color": (98, 42, 15)}
        }
        self.colorsMainBlind = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 6,
            "9": 6
        }

        self.colorsMainBlindSmall = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": -1,
            "7": 6,
            "8": None,
            "9": None
        }

        self.loadChuzzleTextures()
        self.loadChuzzleTexturesFat()

    def loadTexture(self, filename, alphaOnly=False, pngOnly=False, split=-1, split_vertical=False):

        if "_Alpha" in filename:
            alphaOnly = True
            pngOnly = False

        if alphaOnly:
            base = createAlpha(filename)
        else:
            base = self.p.image.load(filename).convert_alpha()

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
            if not split_vertical:
                return splitTexture(surface, split)

            # vertical split
            piece_h = surface.get_height() // split
            pieces = []
            for i in range(split):
                piece = p.Surface((surface.get_width(), piece_h), p.SRCALPHA)
                piece.fill((0, 0, 0, 0))
                piece.blit(
                    surface,
                    (0, 0),
                    (0, i * piece_h, surface.get_width(), piece_h),
                    p.BLEND_RGBA_ADD
                )
                pieces.append(piece)
            return pieces

        return surface


    def loadChuzzleTextures(self):

        createAlpha("textures/Chuzzle-Base_Alpha.png")
        createAlpha("textures/Chuzzle-Hilite_Alpha.png")

        BaseImage = p.image.load("textures/Chuzzle-Base.png").convert_alpha()
        AlphaImage = p.image.load("cached/Chuzzle-Base_Alpha.png").convert_alpha()
        HiliteImage = p.image.load("cached/Chuzzle-Hilite_Alpha.png").convert_alpha()

        surface = p.Surface((4860, 84), p.SRCALPHA)
        surface.fill((0, 0, 0, 255))

        for color in range(9):
            colorObj = self.colorsMain[str(color)]

            print(colorObj)
            base = None
            if "color" in colorObj.keys():
                base = BaseImage.copy()
                color1 = surface.copy()
                color1.fill(colorObj["color"])

                color2 = surface.copy()
                color2.fill((60,60,60,0))
                color2.blit(HiliteImage, (0, 0), None, p.BLEND_RGBA_MULT)

                base.blit(color1, (0, 0), None, p.BLEND_RGBA_MULT)
                base.blit(color2, (0, 0), None, p.BLEND_RGB_ADD)
                base.blit(AlphaImage, (0, 0), None, p.BLEND_RGBA_MULT)

            elif "texture" in colorObj.keys():
                base = surface.copy()
                baseImage = p.image.load(colorObj["texture"]["base"]).convert_alpha()
                alphaImage = p.image.load(colorObj["texture"]["alpha"]).convert_alpha()
                base.blit(baseImage, (0, 0), None, p.BLEND_RGBA_ADD)
                base.blit(alphaImage, (0, 0), None, p.BLEND_RGBA_MULT)

            # This creates all the chuzzle textures
            surfaces = []
            for i in range(60):
                surf = p.Surface((81, 81), p.SRCALPHA)
                surf.fill((0, 0, 0, 0))
                surfaces.append(surf)
                surfaces[-1].blit(base, (0, 0), (81 * i, 2, 80, 80), p.BLEND_RGBA_ADD)

                # resize the texture
                new = 480 / self.game.settings.boardSize + 1
                if self.game.settings.boardSize != 6:
                    oldSize = Point(surfaces[-1].get_size())
                    newSize = (oldSize * new / 81).tuple()
                    surfaces[-1] = p.transform.scale(surfaces[-1], newSize)

            self.ChuzzleBase[color] = surfaces

    def loadChuzzleTexturesFat(self):

        createAlpha("textures/ChuzzleHuge-Base_Alpha.png")
        createAlpha("textures/ChuzzleHuge-Hilite_Alpha.png")

        BaseImage = p.image.load("textures/ChuzzleHuge-Base.png").convert_alpha()
        AlphaImage = p.image.load("cached/ChuzzleHuge-Base_Alpha.png").convert_alpha()
        HiliteImage = p.image.load("cached/ChuzzleHuge-Hilite_Alpha.png").convert_alpha()

        surface = p.Surface((5370, 183), p.SRCALPHA)
        surface.fill((0, 0, 0, 255))

        for color in range(9):
            colorObj = self.colorsMain[str(color)]

            print(colorObj)
            base = None

            # Fat-specific solid color override.
            # This is required for yellow because the normal yellow texture strip
            # does not map correctly onto the huge/fat sheet layout.
            if "fat_color" in colorObj:
                base = BaseImage.copy()

                color1 = surface.copy()
                color1.fill(colorObj["fat_color"])

                color2 = surface.copy()
                color2.fill((60, 60, 60, 60))
                color2.blit(HiliteImage, (0, 0), None, p.BLEND_RGBA_MULT)

                base.blit(color1, (0, 0), None, p.BLEND_RGBA_MULT)
                base.blit(color2, (0, 0), None, p.BLEND_RGB_ADD)
                base.blit(AlphaImage, (0, 0), None, p.BLEND_RGBA_MULT)

            elif "color" in colorObj:
                base = BaseImage.copy()

                color1 = surface.copy()
                color1.fill(colorObj["color"])

                color2 = surface.copy()
                color2.fill((60, 60, 60, 60))
                color2.blit(HiliteImage, (0, 0), None, p.BLEND_RGBA_MULT)

                base.blit(color1, (0, 0), None, p.BLEND_RGBA_MULT)
                base.blit(color2, (0, 0), None, p.BLEND_RGB_ADD)
                base.blit(AlphaImage, (0, 0), None, p.BLEND_RGBA_MULT)

            elif "texture" in colorObj:
                base = surface.copy()
                baseImage = p.image.load(colorObj["texture"]["base"]).convert_alpha()
                alphaImage = p.image.load(colorObj["texture"]["alpha"]).convert_alpha()
                base.blit(baseImage, (0, 0), None, p.BLEND_RGBA_ADD)
                base.blit(alphaImage, (0, 0), None, p.BLEND_RGBA_MULT)

            surfaces = []
            for i in range(30):
                surf = p.Surface((179, 179), p.SRCALPHA)
                surf.fill((0, 0, 0, 0))
                surfaces.append(surf)
                surfaces[-1].blit(base, (0, 0), (179 * i, 0, 179, 179), p.BLEND_RGBA_ADD)

                # resize the texture
                new = 480 / self.game.settings.boardSize * 2 + 18
                if self.game.settings.boardSize != 6:
                    oldSize = Point(surfaces[-1].get_size())
                    newSize = (oldSize * new / 179).tuple()
                    surfaces[-1] = p.transform.scale(surfaces[-1], newSize)

            self.ChuzzleHuge[color] = surfaces






