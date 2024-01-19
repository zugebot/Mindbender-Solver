
#include <cstdio>
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int sum;
        std::vector<int>::iterator x, y;
        auto end = nums.end();
        auto begin = nums.begin();

        for (x = nums.begin(); x != end; x++) {
            for (y = x + 1; y != end; y++) {
                sum = *x + *y;

                if (sum == target) {
                    goto END;
                }
            }
        }
        END:
        return {(int)(x - begin), (int)(y - begin)};
    }
};

int main() {
    Solution sol;
    std::vector<int> vec = {3, 2, 4};
    auto answer = sol.twoSum(vec, 6);
    std::cout << answer[0] << " " << answer[1] << std::endl;


}