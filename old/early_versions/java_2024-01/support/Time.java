package support;

public final class Time {

    public static String time(double start) {
        return String.format("%.3f",
                (System.nanoTime() - start) / 1_000_000_000);
    }

}
