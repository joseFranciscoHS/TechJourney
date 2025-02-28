from math import inf
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        new_low = int(10**4)
        for price in prices:
            if price < new_low:
                new_low = price

            tmp_profit = price - new_low
            if tmp_profit > profit:
                profit = tmp_profit

        return profit


print(Solution().maxProfit([7, 1, 5, 3, 6, 4]))
