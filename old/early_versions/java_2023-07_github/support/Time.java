package support;

public class Time {

    public static String time(double start) {
        return String.format("%.3f",
                (System.currentTimeMillis() - start) / 1000);
    }
}
