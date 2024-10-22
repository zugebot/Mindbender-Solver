#pragma once

#include <iomanip>
#include <iostream>
#include <vector>

#include "MindbenderSolver/utils/processor.hpp"
#include "MindbenderSolver/include.hpp"


struct PermGenPair {
    u32 start;
    u32 end;

    explicit PermGenPair() : start(0), end(0) {}

    PermGenPair(C u32 theStart, C u32 theLength) {
        start = theStart;
        end = theStart + theLength;
    }
};



class PermGen {
    std::vector<PermGenPair> myPairs;
    std::vector<u8> myToBePermuted;

public:
    std::vector<std::vector<u8>> myOutput{};

    void allPermutations(
       C std::vector<u8>& toBePermuted,
       C std::vector<PermGenPair>& thePairs) {
        myToBePermuted = toBePermuted;
        myPairs = thePairs;
        myOutput.clear();
        allPermutations(thePairs[0].start, 0);
    }

    void printVectors() {
        std::cout << "\n";

        for (int index = 0; index < myOutput.size(); ++index) {
            std::vector<u8>& theVector = myOutput[index];

            int i = 0;
            std::cout << std::setw(3) << index + 1 << ": { ";
            for (; i < theVector.size(); i++) {
                if (theVector[i] == 0) {
                    std::cout << ". ";
                    continue;
                }
                std::cout << static_cast<u32>(theVector[i]) << " ";
            }
            std::cout << "}\n";
        }

    }

private:

    void allPermutations(C u32 nextIndex, C u32 pairIndex) {
        if (nextIndex == myPairs[pairIndex].end) {
            if (pairIndex == myPairs.size() - 1) {
                myOutput.emplace_back(myToBePermuted);
            } else {
                allPermutations(myPairs[pairIndex + 1].start, pairIndex + 1);
            }
            return;
        }

        for (u32 i = nextIndex; i < myPairs[pairIndex].end; i++) {
            // swap items
            C u8 x1 = myToBePermuted[i];
            myToBePermuted[i] = myToBePermuted[nextIndex];
            myToBePermuted[nextIndex] = x1;
            // recursion
            allPermutations(nextIndex + 1, pairIndex);
            // swap items
            C u8 x2 = myToBePermuted[i];
            myToBePermuted[i] = myToBePermuted[nextIndex];
            myToBePermuted[nextIndex] = x2;
        }
    }
};


static std::vector<u8> createMemoryPermutations(
    C std::vector<u8>& theMemory) {
    u32 index = 0, length = 1;
    std::vector<PermGenPair> pairs;

    char isRow = allActStructList[0].name[0];

    for (u32 i = 1; i < static_cast<u32>(theMemory.size()); i++) {
        C u8 action = theMemory[i];
        C char thisDir = allActStructList[action].name[0];
        if (thisDir == isRow) {
            length++;
        } else {
            if (length != 1) {
                pairs.push_back({index, length});
            }
            index = i;
            length = 1;
            isRow = thisDir;
        }
        i++;
    }

    PermGen<u8> gen;
    gen.allPermutations(theMemory, pairs);

    return gen.myOutput;
}


/*
given a std::vector<Memory>
iterate over it, and make sure all actions are sorted
sort the memory array
remove duplicates
iterate over this final list with createMemoryPermutations
 */
