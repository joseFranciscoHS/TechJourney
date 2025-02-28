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
