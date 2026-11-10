package support;
import java.util.*;



public class Colors {

    public static String Red = "\033[31m";
    public static String Orange ="\033[38:5:208m";
    public static String Yellow = "\033[93m";
    public static String Green = "\033[32m";
    public static String Cyan = "\033[96m";
    public static String CyanBold = "\033[96;1m";
    public static String Blue = "\033[34m";
    public static String Magenta = "\033[35m";
    public static String White = "\033[37m";
    public static String Black = "\033[30m";
    public static String Reset = "\033[0;0m";

    public static List<String> colors = Arrays.asList(
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
        } else {
            return colors.get(9);
        }
        
    }
}