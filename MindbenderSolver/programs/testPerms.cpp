#include "MindbenderSolver/code/perms_gen.hpp"

#include "MindbenderSolver/utils/processor.hpp"


int main() {

    const std::vector<u8> theVector = {0, 1, 2, 3, 0, 0, 4, 5, 0};
    // this is the set of pieces of the vector to permute
    const std::vector<PermGenPair> pairs = {
        {1, 3}, // index 1, length 3
        {6, 2}, // index 6, length 2
    };

    PermGen<u8> gen;
    // this function creates a list of all permutations
    // given a pair (index, length) of an input list
    // and permutations of all perms passed
    gen.allPermutations(theVector, pairs);
    gen.printVectors();

    int x;
    std::cin >> x;
}
