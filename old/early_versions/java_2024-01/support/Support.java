package support;

public class Support {

    public static void printArray(byte[] array) {
        for (byte b : array) {
            System.out.print(b + " ");
        }
        System.out.println();
    }

    public static void printArray(byte[][] array) {
        for (int y = 0; y < array.length; y++) {
            System.out.print("Y=" + y + " | ");
            for (int x = 0; x < array[y].length; x++) {
                System.out.print(array[y][x] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }

}
