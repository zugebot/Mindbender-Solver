// JVec_Test.cpp
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <algorithm>



#include "MindbenderSolver/utils/jvec.hpp"


void TestConstruction() {
    printf("TestConstruction: ");
    JVec<int> vec(10);
    assert(vec.size() == 0);
    assert(vec.capacity() == 10);
    assert(vec.data() != nullptr);
    printf("Passed\n");
}

void TestSizeAndCapacity() {
    printf("TestSizeAndCapacity: ");
    JVec<int> vec(5);
    assert(vec.size() == 0);
    assert(vec.capacity() == 5);
    vec.resize(3);
    assert(vec.size() == 3);
    assert(vec.capacity() == 5);
    vec.resize(5);
    assert(vec.size() == 5);
    assert(vec.capacity() == 5);
    printf("Passed\n");
}

void TestElementAccess() {
    printf("TestElementAccess: ");
    JVec<int> vec(5);
    vec.resize(3);
    vec[0] = 10;
    vec[1] = 20;
    vec[2] = 30;
    assert(vec[0] == 10);
    assert(vec[1] == 20);
    assert(vec[2] == 30);
    printf("Passed\n");
}

void TestResizing() {
    printf("TestResizing: ");
    JVec<int> vec(2);
    vec.resize(2);
    vec[0] = 1;
    vec[1] = 2;
    assert(vec.size() == 2);
    assert(vec.capacity() == 2);
    vec.resize(4);
    assert(vec.size() == 4);
    assert(vec.capacity() == 4);
    vec[2] = 3;
    vec[3] = 4;
    assert(vec[2] == 3);
    assert(vec[3] == 4);
    vec.resize(1);
    assert(vec.size() == 1);
    printf("Passed\n");
}

void TestReserve() {
    printf("TestReserve: ");
    JVec<int> vec(2);
    vec.resize(2);
    vec[0] = 100;
    vec[1] = 200;
    vec.reserve(5);
    assert(vec.capacity() == 5);
    assert(vec.size() == 2);
    assert(vec[0] == 100);
    assert(vec[1] == 200);
    printf("Passed\n");
}

void TestClear() {
    printf("TestClear: ");
    JVec<int> vec(3);
    vec.resize(3);
    vec[0] = 7;
    vec[1] = 8;
    vec[2] = 9;
    assert(vec.size() == 3);
    vec.clear();
    assert(vec.size() == 0);
    printf("Passed\n");
}

void TestSwap() {
    printf("TestSwap: ");
    JVec<int> vec1(3);
    vec1.resize(3);
    vec1[0] = 1;
    vec1[1] = 2;
    vec1[2] = 3;

    JVec<int> vec2(2);
    vec2.resize(2);
    vec2[0] = 4;
    vec2[1] = 5;

    vec1.swap(vec2);

    // After swap
    assert(vec1.size() == 2);
    assert(vec1.capacity() == 2);
    assert(vec1[0] == 4);
    assert(vec1[1] == 5);

    assert(vec2.size() == 3);
    assert(vec2.capacity() == 3);
    assert(vec2[0] == 1);
    assert(vec2[1] == 2);
    assert(vec2[2] == 3);
    printf("Passed\n");
}

void TestIteration() {
    printf("TestIteration: ");
    JVec<int> vec(5);
    vec.resize(5);
    for (uint64_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<int>(i * 10);
    }

    int expected = 0;
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        assert(*it == expected);
        expected += 10;
    }
    printf("Passed\n");
}

void TestOverResize() {
    printf("TestOverResize: ");
    JVec<int> vec(2);
    vec.resize(5); // Increase size beyond initial capacity
    assert(vec.size() == 5);
    assert(vec.capacity() == 5);
    for (uint64_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<int>(i + 1);
    }
    for (uint64_t i = 0; i < vec.size(); ++i) {
        assert(vec[i] == static_cast<int>(i + 1));
    }
    printf("Passed\n");
}

void TestReserveLessThanCapacity() {
    printf("TestReserveLessThanCapacity: ");
    JVec<int> vec(5);
    vec.reserve(3); // Should not reduce capacity
    assert(vec.capacity() == 5);
    printf("Passed\n");
}

void TestResizeWithZero() {
    printf("TestResizeWithZero: ");
    JVec<int> vec(5);
    vec.resize(0);
    assert(vec.size() == 0);
    assert(vec.capacity() == 5);
    printf("Passed\n");
}

// --- Main Function to Run Tests ---

int main() {
    printf("Running JVec Unit Tests...\n\n");

    TestConstruction();
    TestSizeAndCapacity();
    TestElementAccess();
    TestResizing();
    TestReserve();
    TestClear();
    TestSwap();
    TestIteration();
    TestOverResize();
    TestReserveLessThanCapacity();
    TestResizeWithZero();

    printf("\nAll tests passed successfully!\n");
    return 0;
}