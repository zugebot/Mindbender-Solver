# Jerrin Shirks

# custom imports
from tools.font import Font


class HarlowSolidItalic28Bold(Font):
    def __init__(self):
        super().__init__()
        self.filename = "textures/fonts/HarlowSolidItalic28Bold_Alpha.gif"

        self.CharList = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        self.WidthList = (18, 18, 18, 18, 18, 18, 18, 18, 18, 18)

        self.RectList = ((0, 0, 19, 47), (20, 0, 13, 47), (34, 0, 18, 47), (53, 0, 18, 47),
                         (72, 0, 19, 47), (92, 0, 19, 47), (112, 0, 16, 47), (129, 0, 17, 47),
                         (147, 0, 18, 47), (166, 0, 17, 47))

        self.OffsetList = ((2, 0), (6, 0), (2, 0), (3, 0), (2, 0), (3, 0), (3, 0), (4, 0),
                           (3, 0), (4, 0))

        self.prepare()


"""
CreateLayer          Main;
LayerSetImage        Main 'HarlowSolidItalic28Bold';
LayerSetAscent       Main 33;
LayerSetCharWidths   Main CharList WidthList;
LayerSetCharWidths   Main (' ') (20);
LayerSetImageMap     Main CharList RectList;
LayerSetCharOffsets  Main CharList OffsetList;
LayerSetPointSize    Main 28;

SetDefaultPointSize  28;
"""
