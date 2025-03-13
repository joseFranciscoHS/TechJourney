"""
Given an integer array nums, find a subarray that has the largest product, and return the product.

The test cases are generated so that the answer will fit in a 32-bit integer.

 

Example 1:

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
Example 2:

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
 

Constraints:

1 <= nums.length <= 2 * 104
-10 <= nums[i] <= 10
The product of any subarray of nums is guaranteed to fit in a 32-bit integer.
"""

from typing import List


class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        # we will keep track of the max and min of the products
        max_prod = nums[0]
        current_max = nums[0]
        current_min = nums[0]
        for i in range(1, len(nums)):
            # if the current number is negative we need to swap the max and min
            if nums[i] < 0:
                current_max, current_min = current_min, current_max
            # update the max and min
            # take the max of the current product or the current number
            current_max = max(current_max * nums[i], nums[i])
            current_min = min(current_min * nums[i], nums[i])

            # keep the max of products
            max_prod = max(max_prod, current_max)

        return max_prod


print(Solution().maxProduct([2, 3, -2, 4]))
