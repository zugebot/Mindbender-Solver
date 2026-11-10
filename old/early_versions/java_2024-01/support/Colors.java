package support;
import java.util.*;


public final class Colors {

    public static final String Red = "\033[31m";
    public static final String Orange ="\033[38:5:208m";
    public static final String Yellow = "\033[93m";
    public static final String Green = "\033[32m";
    public static final String Cyan = "\033[96m";
    public static final String CyanBold = "\033[96;1m";
    public static final String Blue = "\033[34m";
    public static final String Magenta = "\033[35m";
    public static final String White = "\033[37m";
    public static final String Black = "\033[30m";
    public static final String Reset = "\033[0;0m";

    public static final List<String> colors = Arrays.asList(
            Red,
            Orange,
            Yellow,
            Green,
            Cyan,
            Blue,
            Magenta,
            White,
            Black,
            Reset);

    public static String getColor(byte index) {
        if (index < 9) {
            return colors.get(index);
        }
        return colors.get(9);
    }
}