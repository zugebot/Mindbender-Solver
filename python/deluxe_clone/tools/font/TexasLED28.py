# Jerrin Shirks

# custom imports
from tools.font import Font


class TexasLED28(Font):
    def __init__(self):
        super().__init__()
        self.filename = "textures/fonts/TexasLED28_Alpha.gif"

        self.CharList = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

        self.WidthList = (21, 16, 21, 21, 21, 21, 21, 20, 21, 21, 21)

        self.RectList = ((0, 0, 18, 29), (19, 0, 11, 29), (31, 0, 18, 29), (50, 0, 18, 29),
                         (69, 0, 18, 29), (88, 0, 18, 29), (107, 0, 18, 29), (126, 0, 18, 29),
                         (145, 0, 18, 29), (164, 0, 18, 29), (183, 0, 18, 29))

        self.OffsetList = ((3, 0), (7, 0), (3, 0), (3, 0), (3, 0), (3, 0), (3, 0), (3, 0),
                           (3, 0), (3, 0), (3, 0))

        self.prepare()


"""
CreateLayer          Main;
LayerSetImage        Main 'TexasLED28';
LayerSetAscent       Main 29;
LayerSetCharWidths   Main CharList WidthList;
LayerSetCharWidths   Main (' ') (11);
LayerSetImageMap     Main CharList RectList;
LayerSetCharOffsets  Main CharList OffsetList;
LayerSetPointSize    Main 28;

SetDefaultPointSize  28;
"""
