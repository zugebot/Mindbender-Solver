import java.util.ArrayList;
import java.util.List;

public class Permutations {
    public static void main(String[] args) {
        int[] counts = {6, 6, 1, 0, 0, 0, 0, 0};

        long startTime = System.nanoTime();
        List<byte[]> result = generatePermutations(counts, 6);
        long endTime = System.nanoTime();
        long durationWithList = endTime - startTime;
        System.out.println("Time taken using List<>: " + durationWithList + " nanoseconds");

        for (byte[] perm : result) {
            for (byte b : perm) {
                System.out.print(b);
            }
            System.out.println();
        }
    }

    public static List<byte[]> generatePermutations(int[] counts, int size) {
        List<byte[]> result = new ArrayList<>();
        backtrack(counts, size, new byte[size], 0, result);
        return result;
    }

    private static void backtrack(int[] counts, int remaining, byte[] current, int index, List<byte[]> result) {
        if (remaining == 0) {
            result.add(current.clone());
            return;
        }

        for (int i = 0; i < counts.length; i++) {
            if (counts[i] > 0) {
                counts[i]--;
                current[index] = (byte) i;
                backtrack(counts, remaining - 1, current, index + 1, result);
                counts[i]++;
            }
        }
    }
}