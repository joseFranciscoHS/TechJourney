"""
Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

 

Example 1:

Input: nums = [3,2,3]
Output: 3
Example 2:

Input: nums = [2,2,1,1,1,2,2]
Output: 2

"""

from typing import List, Optional


class Solution:
    def majorityElement(self, nums: List[int]) -> Optional[int]:
        count = 0
        majority = None

        for n in nums:
            if count == 0:
                majority = n

            if n == majority:
                count += 1
            else:
                count -= 1

        return majority
