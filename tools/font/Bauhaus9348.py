# Jerrin Shirks

# custom imports
from tools.font import Font


class Bauhaus9348(Font):
    def __init__(self):
        super().__init__()
        self.filename = "textures/fonts/Bauhaus9348_Alpha.gif"

        self.CharList = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',')

        self.WidthList = (36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 24)

        self.RectList = ((0, 0, 31, 72), (32, 0, 13, 72), (46, 0, 31, 72), (78, 0, 24, 72),
                         (103, 0, 32, 72), (136, 0, 26, 72), (163, 0, 33, 72), (197, 0, 29, 72),
                         (227, 0, 31, 72), (259, 0, 33, 72), (293, 0, 15, 72))

        self.OffsetList = ((3, 0), (12, 0), (3, 0), (6, 0), (2, 0), (5, 0), (2, 0), (4, 0),
                           (3, 0), (2, 0), (5, 0))

        self.prepare()


"""
CreateLayer          Main;
LayerSetImage        Main 'Bauhaus9348';
LayerSetAscent       Main 57;
LayerSetCharWidths   Main CharList WidthList;
LayerSetCharWidths   Main (' ') (32);
LayerSetImageMap     Main CharList RectList;
LayerSetCharOffsets  Main CharList OffsetList;
LayerSetPointSize    Main 48;

SetDefaultPointSize  48;
"""
