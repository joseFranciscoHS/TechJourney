from typing import List


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        last_change = 0
        for i in range(1, len(nums)):
            if nums[last_change] != nums[i]:
                last_change += 1
                nums[last_change] = nums[i]

        return last_change + 1


Solution().removeDuplicates([0, 0, 1, 1, 1, 2, 2, 3, 3, 4])
