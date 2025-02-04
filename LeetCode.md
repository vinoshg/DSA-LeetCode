## Binary Search - O(logn) and O(1)
1. Binary Search 
- Question: Find a target in a Sorted Array or range
- Solution: 1. Take start and end pointers, while loop start <= end calculate mid and compare the target with mid to reduce the range update start and end
2. Search Insert Position / Ceiling of a Number (start) and Floor of a Number (end)
- Solution: 1. Binary search with return start (Ceiling) / end (Floor) instead of -1
3. Find Smallest Letter Greater Than Target
- Solution: 1. Binary search with return array[start % array.length] instead of -1
4. Find First and Last Position of Element in Sorted Array
- Solution: 1. Binary search with target and target-1 or target+1
5. Position of an Element in Infinite Sorted Array
- Solution: 1. Binary search by fixing the range (Exponentially increasing) start=0, end=1 and while target>array[end] then temp=end, end = end + (end-start+1)*2, start=temp
6. Peak Index in a Mountain Array
- Solution: 1. Binary search with array[mid] < array[mid+1] then start= mid+1 else end=mid-1. Finally, return start instead of -1
7. Find in Mountain Array
- Solution: 1. Find Peak index and apply 2 times Binary search with the range of (0,peak) and (peak+1, array.length-1)
8. Find the Pivot element in the Rotated Sorted Array
- Solution: 1. Binary search with 4 cases - 1. mid < end && arr[mid] > arr[mid + 1] = mid 2. mid > start && arr[mid - 1] > arr[mid] = mid-1 
                                            3. arr[mid] <= arr[start] - end=mid-1 else start=mid+1 after while loop return -1
9. Search in Rotated Sorted Array
- Solution: 1. Get Pivot element, target==pivot and apply Binary search 2 times (0, pivot-1) and (pivot+1, array.length-1)
10. Rotation Count in Rotated Sorted Array =  (Pivot + 1)
11. Split Array Largest Sum / Book allocation / Capacity To Ship Packages Within D Days
- Solution: 1. Binary search by taking start=max element, end=sum of all elements and sum=0,pieces=1 if sum+num > mid then sum=num,pieces++ else sum+=num after loop pieces>m - start=mid+1 else end=mid-1. After while loop return start instead of -1 (while loop inside another for loop)

## Cyclic Sort - O(n) and O(n)
1. Cyclic Sort 
- Question: Range (0,n) or (1,n) - Find the target element in an unsorted array
- Solution: 1. while loop i<array.length index=array[i]-1, if array[i]!=array[index] then swap else i++
2. Sort an Array - Sorting Algorithm - O(n*n) and O(1)
- Solution: 1. Bubble Sort - Nested loops i=0 to array.length and j=0 to array.length-i, if array[j]>array[j+1] swap and end ith loop flag break; 2. Insertion Sort - Nested loops i=0 to array.length and j=i+1;j>0;j--, if array[j]<array[j-1] swap else break; 3. Selection Sort - For loop i=0 to array.length last=array.length-i-1 getMaxIndex= range(0, last) swap(last,maxIndex)
3. Missing Number / Find all numbers disappeared in an array
- Solution: 1. Apply Cyclic Sort and then For loop if array[i]!=i+1 got and(i+1)
4. Find the Duplicate Number / Find all Duplicates in an array
- Solution: 1. Apply Cyclic Sort and then For loop if array[i]!=i+1 got ans(array[i])
5. Set Mismatch
- Solution: 1. Apply Cyclic Sort and then For loop if array[i]!=i+1 got ans(array[i], i+1)
6. First missing Positive
- Solution: 1. Apply Cyclic Sort and then For loop if array[i]!=i+1 got ans(i+1) if not found return array.length-1

## Prefix Sum - O(n)
1. Prefix Sum Array
- Solution: 1. For loop i=1 to array.length array[i] = array[i-1] + array[i] 
2. Range Sum Query
- Solution: 1. Build Prefix Sum Array and return prefixSum[right] - prefixSum[left-1]. Handle if left==0 then return prefixSum[right]
3. Number of Subarrays whose sum is k
- Solution: 1. PrefixSum with HashMap for if map.containsKey(prefixSum-k) after map.put(prefixSum, map.getOrDefault(prefixSum,0) + 1)          2. Nested loops

## Kadane's Algorithm - O(n) & O(1)
1. [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/description/) - O(n) & O(1) 
- Question: Finding the maximum subarray sum in an array of integers (Array has -ve elements as well)
- Solution: 1. Kadane's Algorithm 2. Nested loop
- So, the algorithm works by iterating through the array. At each step, we decide whether to add the current element to the existing subarray or start a new subarray from the current element. We take the maximum of these two choices. Then, we compare this with the overall maximum sum found so far.
```
class Solution {
    public int maxSubArray(int[] nums) {
        int currentSum = nums[0];
        int maxSum = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            currentSum = Math.max(nums[i], currentSum + nums[i]);
            if (currentSum > maxSum) {
                maxSum = currentSum;
            }
        }
        
        return maxSum;
    }
}
```
## Top K elements - O(nlogk) and O(k)
- Question: K largest/smallest, K most/least frequent, Top k elements
1. Top K largest elements
- Solution: 1. Min heap(Priority Queue) add at max k elements and remaining poll 2. Sort Array and take elements - O(nlogn)
2. K most frequent elements i.e. large in frequency- O(nlogk) & O(n+k)
- Solution: 1. Min heap having HashMap - PriorityQueue<Map.Entry<Integer, Integer>> minHeap = new PriorityQueue<>((a,b) -> a.getValue() - b.getValue())
3. K closest points to the origin
- Solution: 1. Max heap - PriorityQueue<int[]> maxHeap = new PriorityQueue<>((a,b) -> getDistance(b) - getDistance(a)) and getDistance is point[0]^2+point[1]^2

## Monotonic Stack - O(n)
- Question: 1. Next greater/smaller, previous greater/smaller
1. Next greater element for each number in an array
- Solution: 1. Monotonic decreasing stack i.e. greater element index at top -For loop and  while !stack.isEmpty() && array[i]>array[stack.peek()] then index= stack.pop(); result[index]=array[i] afer while stack.push(i); 2. Nested loops - O(n^2)
- For Smaller element : change while !stack.isEmpty() && array[i]<array[stack.peek()]
2. Given a list of daily temperatures, determine how many days you've to wait for the next warmer day
- Solution: 1. Like Next greater element - result[index]=i - index

## Sliding Window - O(n) & O(1)
- Fixed Sliding window (Find Subarray/Substring of a fixed length) 2. Dynamic Sliding window (Longest/Shortest Subarray or Substring that satisfies the condition) - (e.g., max sum, longest substring with unique characters)
1. [Maximum Sum of Subarrays With Length K / Maximum sum of a subarray of size k](https://leetcode.com/problems/maximum-average-subarray-i/description/) - O(n) & O(1)
- Question: Find the maximum sum of any contiguous subarray of length k / Max avg sum of Subarray (avg=max/array.length) - Fixed Sliding window
- Solution: 1. Sliding Window 2. Nested loops - O(n^2) & O(1)
- Compute the initial sum of the first k elements.
- Then slide the window:
- Initialize max_sum to this initial sum.
- Then for each i in k to n-1:
- sum = sum - nums[i -k] + nums[i]
- if sum > max_sum, update max_sum.
- At the end, return max_sum.
```
class Solution {
    public long maximumSubarraySum(int[] nums, int k) {
        int n = nums.length;
        if (n < k) {
            return 0;
        }
        
        long currentSum = 0;
        for (int i = 0; i < k; i++) {
            currentSum += nums[i];
        }
        
        long maxSum = currentSum;
        
        for (int i = k; i < n; i++) {
            currentSum += nums[i] - nums[i - k];
            if (currentSum > maxSum) {
                maxSum = currentSum;
            }
        }
        
        return maxSum;
    }
}
```
2. [Maximum Sum of Distinct Subarrays With Length K](https://leetcode.com/problems/maximum-sum-of-distinct-subarrays-with-length-k/description/) - O(n) & O(1)
- Question: Find the maximum sum of any contiguous subarray of length k where all elements in the subarray are distinct
- Solution: 1. Sliding Window 2. Nested loops - O(n^2) & O(1)
- Use a sliding window approach with a frequency map to track the counts of elements in the current window.
- Expand the window by adding elements to the right.
- If adding an element causes duplicates, move the left pointer to the right until all elements in the window are unique.
- Once the window's size is exactly k and all elements are unique, calculate the sum and update the maximum sum if necessary.
```
class Solution {
    public long maximumSubarraySum(int[] nums, int k) {
        int n = nums.length;
        if (n < k) {
            return 0;
        }
        
        HashMap<Integer, Integer> freq = new HashMap<>();
        long currentSum = 0;
        long maxSum = 0;
        int left = 0;
        
        for (int right = 0; right < n; right++) {
            int num = nums[right];
            currentSum += num;
            freq.put(num, freq.getOrDefault(num, 0) + 1);
            
            // Ensure the window has exactly k elements and all are unique
            while (right - left + 1 > k || freq.get(num) > 1) {
                int leftNum = nums[left];
                currentSum -= leftNum;
                freq.put(leftNum, freq.get(leftNum) - 1);
                if (freq.get(leftNum) == 0) {
                    freq.remove(leftNum);
                }
                left++;
            }
            
            // Check if the current window is of size k and all elements are unique
            if (right - left + 1 == k && freq.size() == k) {
                maxSum = Math.max(maxSum, currentSum);
            }
        }
        
        return maxSum;
    }
}
```
3. [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/) - O(n) & O(min(m,n)) m is the size of the character set (e.g., ASCII)
- Solution: 1. Dynamic Sliding Window with Two Pointers and HashMap / Freq array (Less number of char and only lower case, array size 128) 2. Sliding Window with Two Pointers and HashSet
- Sliding window usually involves two pointers, left and right. The right pointer expands the window as we iterate through the string, and the left pointer adjusts when a duplicate is found
- Initialize a hash map to store the last index of each character.
- Initialize left = 0 and max_length = 0.
- Iterate with right from 0 to s.length - 1:
a. If the current character is in the map and its last index >= left, then update left to be max(left, last index + 1).
b. Update the last index of the current character to right.
c. Update max_length with the current window size (right - left + 1).
```
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> charIndexMap = new HashMap<>();
        int maxLength = 0;
        int left = 0;
        
        for (int right = 0; right < s.length(); right++) {
            char currentChar = s.charAt(right);
            
            if (charIndexMap.containsKey(currentChar) && charIndexMap.get(currentChar) >= left) {
                left = charIndexMap.get(currentChar) + 1;
            }
            
            charIndexMap.put(currentChar, right);
            maxLength = Math.max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
}
```
## Two Pointers - O(n) and O(1) 
- Applicable to Linear Data Structure Arrays, String, LinkedList - Converging pointers (Two pointers start at 0th and array.length-1 and converge together) | Parallel pointers (Right pointer is used to get new info and the left is used to track. Both start from 0/1) | Trigger based pointers (left pointer move only after right pointer reaches particular position) | Expand Around Center
1. [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/description/) - O(n) & O(1)
- Question: To determine if a given string is a palindrome after ignoring non-alphanumeric characters and case differences - Applicable for both Subarray/Substrings
- Solution: 1. Two-pointer approach
- Convert the entire string to lowercase (or uppercase, but lowercase is probably easier). Remove all non-alphanumeric characters. So, keep only a-z, 0-9. Check if the processed string is a palindrome.
- Initialize left pointer at 0 and right pointer at s.length() - 1.
- While left < right:
- Move left pointer to the right until it points to an alphanumeric character.
- Move right pointer to the left until it points to an alphanumeric character.
- Compare the characters at left and right (case-insensitive).
- If they are not equal, return false.
- Otherwise, move both pointers inward.
```
class Solution {
    public boolean isPalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;
        
        while (left < right) {
            // Move left pointer to the next alphanumeric character
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                left++;
            }
            // Move right pointer to the previous alphanumeric character
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            }
            // Compare characters case-insensitively
            if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right))) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
}
```
3. [Move Zeroes](https://leetcode.com/problems/move-zeroes/description/) - O(n) & O(1)
- Question: Moving all zeros to the end of an array while maintaining the order of non-zero elements
- Solution: 1. Trigger-based pointers  2. Take result array fill with 0 and traverse the array, add non-zero element to it - O(n) & O(n)
- Initialize a variable, say 'insertPos' to 0. This will track where the next non-zero should be placed.
- Loop through each element in the array:
a. If the current element is not zero, swap it with the element at 'insertPos' and increment 'insertPos'.
```
class Solution {
    public void moveZeroes(int[] nums) {
        int insertPos = 0;
        for (int num : nums) {
            if (num != 0) {
                nums[insertPos++] = num;
            }
        }
        while (insertPos < nums.length) {
            nums[insertPos++] = 0;
        }
    }
}
```
3. [Container With Most Water](https://leetcode.com/problems/container-with-most-water/description/) - O(n) & O(1)
- Question: Finding the maximum area of water that can be contained between two vertical lines in an array
- Solution: 1. Two-pointer technique 2. Nested loops with find all possible area (i=0 to array.length and j=i+1 to array.length with area) - O(n^2)
- The idea is to start with the widest possible container, which is from the first and last elements. Then, move the pointers inward to try and find a taller line that could result in a larger area.
- Initialize left pointer at 0 and right pointer at the end of the array. Calculate the area and keep track of the maximum. Then, compare the heights at the two pointers. Move the pointer with the shorter height towards the center. Repeat until the pointers meet.
- Wait, why move the shorter one? Because the area is limited by the shorter height. If we move the taller one inward, the width decreases, and the height might not increase, so it's better to move the shorter one in hopes of finding a taller line that could give a larger area.
```
class Solution {
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int maxArea = 0;
        
        while (left < right) {
            int currentHeight = Math.min(height[left], height[right]);
            int currentWidth = right - left;
            maxArea = Math.max(maxArea, currentHeight * currentWidth);
            
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        
        return maxArea;
    }
}
```
4. [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/) - O(n) & O(1)
- Question: Removing duplicates from a sorted array in-place and returning the new length
- Solution: 1. Trigger based two pointers - One pointer to keep track of the position where the next unique element should be placed, and another to traverse the array.
- Let's say I have a variable called 'k' that starts at 1. Because the first element is always unique.
- Then, I loop through the array starting from the second element (index 1). For each element, I compare it with the previous one.
- If they are different, that means it's a new unique element. So I place it at nums[k] and increment k. This way, 'k' will always point to the next position where a unique element should be placed.
```
class Solution {
    public int removeDuplicates(int[] nums) {
        if (nums.length == 0) return 0;
        int k = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[k - 1]) {
                nums[k] = nums[i];
                k++;
            }
        }
        return k;
    }
}
```
5. [Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/) - O(n) & O(1)
- Question: Finding two numbers in a sorted array that add up to a specific target
- Solution: 1. Two pointers(Converging pointers) and two-pointer approach works well with sorted arrays.
- If I have a left pointer starting at the beginning (index 0) and a right pointer at the end (index numbers.length - 1), then I can check the sum of the values at these two pointers. - If the sum is equal to the target, then I found the answer. If the sum is less than the target, I need a larger sum, so I move the left pointer to the right.
- If the sum is larger, then I move the right pointer to the left. That makes sense because moving the left pointer increases the sum, and moving the right decreases it.
- Wait, but the problem requires 1-based indices for the return. So whatever indices I find, I need to add 1 to each before returning. Oh right, the problem statement says the indices are 1-based. So if left is i and right is j in 0-based, then the answer is [i+1, j+1].
```
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        int left = 0;
        int right = numbers.length - 1;
        while (left < right) {
            int sum = numbers[left] + numbers[right];
            if (sum == target) {
                return new int[]{left + 1, right + 1};
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
        return new int[]{-1, -1}; // As per problem statement, this line is unreachable
    }
}
```
6. [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/description/) - O(n^2) & O(1)
- Solution: 1. Expand Around Center Approach 2. The longest one could be of any length, so I have to check all possible substrings. But checking every possible substring would be O(n^3)  
- Initialize variables to keep track of the start and end indices of the longest palindrome found so far.
- Iterate through each character in the string.
- For each character, expand around it assuming it's the center of an odd-length palindrome.
- Also, expand around the current character and the next one for even-length palindromes.
- After each expansion, check if the current palindrome is longer than the previous longest. If yes, update the start and end indices.
- At the end, return the substring from start to end+1.
- Wait, but how do I handle the expansion? Let's think of a helper function that takes left and right indices and expands as long as the characters at those indices are equal. It will return the length of the palindrome found. 
```
class Solution {
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) return "";
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int[] len1 = expandAroundCenter(s, i, i);    // Odd length
            int[] len2 = expandAroundCenter(s, i, i + 1); // Even length
            if (len1[1] - len1[0] > end - start) {
                start = len1[0];
                end = len1[1];
            }
            if (len2[1] - len2[0] > end - start) {
                start = len2[0];
                end = len2[1];
            }
        }
        return s.substring(start, end + 1);
    }

    private int[] expandAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return new int[]{left + 1, right - 1}; //Last char not matched, exit the loop +1 and -1 to the pointers
    }
}
```
## Overlapping Intervals - O(nlogn) & O(n)
- Question: Overlapping intervals, scheduling conflicts - Approach: Sort intervals and merge overlapping ones
1. [Merge Intervals](https://leetcode.com/problems/merge-intervals/description/) - O(nlogn) & O(n)
- Question: Merging overlapping intervals - intervals[][] (2D Array)
- Solution: 1. Overlapping Intervals - Dealing with intervals, a common approach is to sort them
- Check if the intervals array is empty. If yes, return empty.
- Sort the intervals based on the start time (the first element of each interval).
- Initialize a list (or something) to hold the merged intervals. Start by adding the first interval.
- Iterate from the second interval to the end:
a. Get the last merged interval from the list.
b. Compare current interval's start with the end of the last merged interval.
c. If current interval's start <= last merged interval's end: they overlap. Merge them by updating the end of the last merged interval to the max of both ends.
d. If not, add the current interval to the merged list.
- Convert the merged list to an array and return.
- But how to sort the intervals in Java? Since each interval is an int[2], we can sort the array using a custom comparator. In Java, for a 2D array, Arrays.sort can take a Comparator. So for example, using Arrays.sort(intervals, (a, b) -> a[0] - b[0]). This sorts the intervals based on the start time.
```
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals.length == 0) {
            return new int[0][];
        }
        
        // Sort intervals based on their start times
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        
        List<int[]> merged = new ArrayList<>();
        merged.add(intervals[0]);
        
        for (int i = 1; i < intervals.length; i++) {
            int[] current = intervals[i];
            int[] last = merged.get(merged.size() - 1);
            
            if (current[0] <= last[1]) {
                // Merge the intervals by updating the end of the last interval
                last[1] = Math.max(last[1], current[1]);
            } else {
                // Add the current interval as it doesn't overlap
                merged.add(current);
            }
        }
        
        return merged.toArray(new int[merged.size()][]);
    }
}
```
2. [Insert Interval](https://leetcode.com/problems/insert-interval/) - O(n) if array's sorted & O(n)
- Solution: 1. Overlapping Intervals
- When dealing with intervals, sorting is often a good first step. 1. Find where the new interval starts in relation to the existing intervals 2. Merge any overlapping intervals with the new one 3. Add the remaining intervals that don't overlap after the merged part.
- Create a list to collect the result.
- Iterate through each interval in intervals:
a. If the current interval ends before the newInterval starts, add to result.
b. Else if the current interval starts after the newInterval ends, add the newInterval (if not added), then add current and remaining intervals.
c. Else, merge the current interval into the newInterval by updating newInterval's start and end.
- After processing all intervals, add the newInterval if it hasn't been added yet.
```
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> result = new ArrayList<>();
        int i = 0;
        int n = intervals.length;
        
        // Add all intervals ending before newInterval starts
        while (i < n && intervals[i][1] < newInterval[0]) {
            result.add(intervals[i]);
            i++;
        }
        
        // Merge overlapping intervals
        int newStart = newInterval[0];
        int newEnd = newInterval[1];
        while (i < n && intervals[i][0] <= newEnd) {
            newStart = Math.min(newStart, intervals[i][0]);
            newEnd = Math.max(newEnd, intervals[i][1]);
            i++;
        }
        result.add(new int[]{newStart, newEnd});
        
        // Add remaining intervals
        while (i < n) { //while(i<n && intervals[i][0] > newEnd) 
            result.add(intervals[i]);
            i++;
        }
        
        return result.toArray(new int[result.size()][]);
    }
}
```
## Greedy Algorithm
1. Longest Consecutive Sequence
- Solution: 1. HashSet fill with array element. For num if (!set.contains(num - 1)) currentNum=num while (set.contains(currentNum + 1)

## Dynamic Programming - Iteration(Two variable & dp[] array) and Memoization(Recursion)
- Maximise/Minimise/Fewest of certain value or number of ways
- Question: Optimize recursive problems with overlapping subproblems - 0/1 Knapsack | Unbounded Knapsack | Longest Common Subsequence (LCS) | Fibonacci sequence pattern
- 0/1 Knapsack(Subset selection with constraints) | Unbounded Knapsack(Unlimited item usage) | LCS(String alignment problems)
1. Longest Common Subsequence - O(mn)
- Solution: 1. dp[m+1][n+1] Nested loops i,j=1 if (text1.charAt(i - 1) == text2.charAt(j - 1)) dp[i][j] = dp[i - 1][j - 1] + 1; else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]); after return dp[m][n];
2. [Target Sum](https://leetcode.com/problems/target-sum/description/) - O(n*S) & O(S) S is the required subset sum 
- Question: Finding the number of ways to assign '+' and '-' signs to elements of an array such that their sum equals a given target
- Solution: 1. Dynamic programming - 0/1 Knapsack (Subsets problem)  2. Recursion (p,up)
- Similar to partitioning the array into two subsets where the difference between their sums is equal to the target. Because if I have a subset with sum S1 and another with sum S2, then S1 - S2 = target. But since S1 + S2 is the total sum of the array. Let totalSum be the sum of all elements. Then S1 - (totalSum - S1) = target. That simplifies to 2*S1 = target + totalSum. So S1 = (target + totalSum)/2. Therefore, the problem reduces to finding the number of subsets that sum up to S1, where S1 is (target + totalSum)/2. But wait, this makes sense only if target + totalSum is even and non-negative. Otherwise, there's no solution. So first, I need to check if (target + totalSum) is even and that (target + totalSum) >= 0. If not, return 0.
- Calculate the total sum of the array.
- Check if (target + totalSum) is even and non-negative. If not, return 0.
- Compute S1 = (target + totalSum)/2.
- Find the number of subsets of nums that add up to S1. This is a classic subset sum problem which can be solved using dynamic programming.
```
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        int totalSum = 0;
        for (int num : nums) {
            totalSum += num;
        }
        
        // Check if the target is achievable
        if ((totalSum + target) % 2 != 0 || (totalSum + target) < 0) { //S1 has to be even and non-negative value otherwise return 0
            return 0;
        }
        
        int requiredSum = (totalSum + target) / 2;
        int[] dp = new int[requiredSum + 1];
        dp[0] = 1; // Base case: one way to get sum 0 (using no elements)
        
        for (int num : nums) {
            for (int j = requiredSum; j >= num; j--) { //j=S1 to num
                dp[j] += dp[j - num];
            }
        }
        
        return dp[requiredSum];
    }
}
```
3. [Coin Change - Unbounded Knapsack](https://leetcode.com/problems/coin-change/description/) - O(n * amount) & O(amount)
- Question: Finding the minimum number of coins (having infinite coins) needed to make up a given amount
- Solution: 1. Dynamic programming - Unbounded Knapsack(similar to loop through each coin and then through the amounts)  2. Recursion
- Initialize a DP array of size amount+1, filled with a value larger than the maximum possible (like amount+1), except dp[0] = 0.
- For each amount from 1 to amount:
a. For each coin in coins:
i. If the coin value <= current amount, check if dp[current amount - coin value] +1 is less than current dp[current amount]. If so, update.
- After processing all, if dp[amount] is still larger than amount, return -1, else return dp[amount].
```
class Solution {
    public int coinChange(int[] coins, int amount) {
        if (amount == 0) return 0;
        
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) { 
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1); //i is amount and comparing with previous amount by - coin
                }
            }
        }
        
        return dp[amount] > amount ? -1 : dp[amount]; //coins [2], amount 1. Then dp[1] remains 2, which is larger than 1. So return -1.
    }
}
```
4. [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/description/) - O(n) & O(1)
- Question: Finding the number of distinct ways to climb n stairs where you can take 1 or 2 steps at a time (Fibonacci sequence pattern)
- Solution: 1. Dynamic programming problem similar to the Fibonacci sequence.
- If n is 1, return 1.
- If n is 2, return 2.
- For n > 2, iterate up to n, calculating each step as the sum of the two previous steps.
```
class Solution {
    public int climbStairs(int n) {
        if (n == 1) return 1;
        if (n == 2) return 2;
        
        int first = 1, second = 2;
        for (int i = 3; i <= n; i++) {
            int third = first + second;
            first = second;
            second = third;
        }
        return second;
    }
}
```
5. [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/description/) - O(nlogn) & O(n)
- Question: Find the length of the longest subsequence in an array that is strictly increasing. A subsequence means the elements don't have to be consecutive, but they have to be in order.
- Solution: 1. Dynamic programming approach combined with binary search 2. Dynamic programming approach For each element, the idea is to check all previous elements and see if the current element can extend their subsequences. But that would be O(n^2) 
- Initialize an empty list.
- For each number in the array:
a.If the number is greater than the last element in the list, append it.
b. Else, find the index of the first element in the list >= current number, replace it with current number.
- The length of the list is the answer.
```
class Solution {
    public int lengthOfLIS(int[] nums) {
        ArrayList<Integer> tails = new ArrayList<>();
        for (int num : nums) {
            if (tails.isEmpty() || num > tails.get(tails.size() - 1)) {
                tails.add(num);
            } else {
                int index = Collections.binarySearch(tails, num);
                if (index < 0) { // Collections.binarySearch returns the index if present, else returns (-1 -insertion point). So the insertion point is (-(index +1)).
                    index = -(index + 1);
                }
                tails.set(index, num);
            }
        }
        return tails.size();
    }
}
```
## Linked List
- Fast and Slow pointers - O(n) & O(1) | Reversal of Linked List using Three pointers - O(n) & O(1)
1. Linked List Cycle
- Solution: 1. Fast and Slow pointers while(fast != null && fast.next != null) if(fast == slow) return true after while return false;     
     2. HashSet - while(current != null) if(set.contains(current)) return true; set.add(current) current=current.next; - O(n) & O(n)
2. Middle of the Linked List
- Solution: 1. Fast and Slow pointers, return slow; - Single parse 2. Find the length of the LL count=0 and Apply For loop again i=0 to count/2, return current - Two times for loop - O(n) & O(1)
3. Happy Number
- Question: Given num 14 = 1^2+4^2 = 17 ... If it ends with '1' it means Happy, else the loop again 17 is not a happy number
- Solution: 1. Fast and Slow pointers - slow=n,fast=getNext(n) while(fast!=1 && slow != fast) slow,fast increment after while return fast==1
  2. HashSet - while(n>0) findSquare and check in the if set.contains(square) return true otherwise set.add(squre)  after while return false
4. Reverse a Linked List
- Solution: 1. Three-pointers - prev=null,current=head while(current!=null) next=current.next, current.next=prev, prev=current, current=next after while loop return prev 2. Copy all the elements to an array and reverse it. Update LL according to the array - O(n) & O(n)
    
## Trie - O(L) and O(N*L)
1. Implement Trie (Prefix Tree) - insert, search, startsWithPrefix
- Node[] children; boolean eow;
2. Word Break 
- Question: Break the word into multiple parts and check whether all parts are present in the dictionary
- Solution: 1. Reverse DP with dp[i] = dp[i + w.length()] 2. Trie insert, search and recursion helper for dividing string 
3. Count Unique Substrings / Count Trie Nodes
- Solution: 1. HashSet with Nested loops to get a substring 2. Trie insert and count nodes
4. Longest word with all prefixes / Longest Word in Dictionary
- Solution: 1. HashSet contains its prefix of (length - 1) 2. Trie inserts and builds a string with it's all prefixes present in the dictionary along with recursion and backtracking

## Graph - Time:O(V+E) Space:O(V)
1. Connected and Disconnected Graph - BFS (Using Queue) and DFS (Using Recursion) along with the visited boolean array
2. All Paths From Source to Target
- Solution: 1. DFS with Backtracking (As per the question nodes are to be visited multiple times, so the visited array is not required and backtracking will handle this list.remove(list.size() - 1))
3. Cycle Detection - Directed Graph
- Solution: 1. DFS recursion with the boolean stack and visited array
4. Cycle Detection - Undirected Graph
- Solution: 1. DFS recursion with parent element and visited array
5. Topological Sort - DAG O(V+E)
- Question: Directed Acyclic Graph(DAG) with no cycle (Not for non-DAGs). It's a linear order of vertices u->v (Dependency)
- Solution: 1. DFS recursion with Stack
6. Dijkstra’s Algorithm (Shortest Distance) - O(E+ElogV)
- Question: Applicable only for a positive weight and its Greedy Algorithm
- Solution: 1. BFS with Priority Queue as Pair(node, distance) and distance array initialize with Integer.MAX for all nodes except the source with relaxation condition dist[u] + edge.weight < dist[v] then dist[v] = dist[u] + edge.weight (u-current and v-neighbor node)
7. Bellman Ford Algorithm (Shortest Distance) - O(E.V) More than Dijktra's 
- Question: The shortest distance for -ve weights and without a cycle. DP Algorithm - Perform relaxation in the loop V-1 times.
  Detect -ve weight cycle by running nested i and j loop code again
- Solution: 1. It's just a V-1 times loop and another i and j loop for relaxation with Dist array. (NO DFS/BFS/Visited array)
8. Minimum Spanning Tree / Minimum Weight Spanning Tree (MST)
- Question: Sub-Graph with all vertices with minimum possible total edge weight without any cycle. All vertices are connected. A graph has multiple spanning trees, but it has an MST (Undirected Weighted Graph)
9. Prim's Algorithm (MST) - O(ElogE)
- Question: Calculate minimum cost
- Solution: 1. BFS with Priority Queue as Pair(node, cost) is Non-MST Set and Visited Array is MST Set.
10. Kosaraju's Algorithm (SCC / Directed Unweighted Graph) - O(V+E)
- Question: Strongly Connected Component - We can reach every vertex of the component from every other vertex in that component
- Solution: 1. Reverse DFS i.e. a) Get a node in a Stack(Topological Sort) b) Transpose the Graph (Reverse the edge direction) c) Do DFS according to stack nodes on the transpose graph
11. Bridge in Graph (Tarjan's Algorithm) - Undirected Connected Graph (UCG) - O(V+E)
- Question: A bridge is an edge whose deletion increases the graph's number of components
- Solution: 1. DFS, Parent element, dt[V] and low[V]. If not visited then dt[current] < low[neighbor]
12. Articulation Point in Graph (Tarjan's Algorithm) - Undirected Connected Graph (UCG)
- Question: Vertex in an undirected connected graph, if removing it disconnects the graph
- Ancestor: Node A, discovered before the current node in DFS, is the ancestor of the curr. To track the ancestor we're using discovery time dt and lowest dt array
- Solution: 1. a) Starting node with parent=-1 & disconnected children's > 1 b) u-v then v is unvisited i.e it's not ancestor (No Back edge) then u is ap or u is the starting point of the cycle => a) par=-1 & children>1 b) par!=-1 & dt[curr] < low[neigh]
Here are 3 cases of current node neighbor 1. Parent (Ignore) 2. Visited (Ancestor) - low[curr] = Math.min(low[curr], dt[neigh]) 3.Not Visited (Child) - low[curr] = Math.min(low[curr], low[neigh]) 2. Naive approach - O(V.(V+E)) - More
13. [Number of Islands](https://leetcode.com/problems/number-of-islands/description/) - O(nm) and O(nm)
- Solution: 1. Depth-First Search (DFS) approach 2. BFS
- The idea is to traverse each cell in the grid and, when encountering a land cell ('1'), explore all connected land cells horizontally and vertically, marking them as visited to avoid counting them again.
- Iterate through each cell in the grid.
- When a cell with '1' is found, increment the island count.
- Perform DFS to mark all connected '1's as visited, so they aren't counted again.
- But how do I mark them as visited? Since modifying the input grid is allowed, I can change visited '1's to '0's to avoid revisiting. That way, no extra space is needed for a visited matrix.
```
class Solution {
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        
        int numIslands = 0;
        int rows = grid.length;
        int cols = grid[0].length;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == '1') {
                    numIslands++;
                    dfs(grid, i, j);
                }
            }
        }
        
        return numIslands;
    }
    
    private void dfs(char[][] grid, int i, int j) {
        int rows = grid.length;
        int cols = grid[0].length;
        
        if (i < 0 || j < 0 || i >= rows || j >= cols || grid[i][j] == '0') {
            return;
        }
        
        grid[i][j] = '0'; // Mark as visited
        dfs(grid, i + 1, j); // Down
        dfs(grid, i - 1, j); // Up
        dfs(grid, i, j + 1); // Right
        dfs(grid, i, j - 1); // Left
    }
}
```
14. [Clone Graph](https://leetcode.com/problems/clone-graph/description/) - O(n) & O(n)
- Solution: 1. Breadth-First Search (Queue) with HashMap (Visited Node) 2. DFS(Recusrion / Stack) with HashMap
- If the input node is null, return null.
- Create a hash map to track original nodes to their clones.
- Use BFS or DFS to traverse the original graph.
- For each node, create a clone if not already present.
- For each neighbor of the current node, clone them and add to the clone's neighbors list.
- Let me outline the BFS approach: Check if the input node is null. If yes, return null.
- Initialize the hash map. Let's call it visited.
- Create the clone of the input node. Add it to visited with original node as key. Create a queue and add the original node to it.
- While the queue is not empty: Dequeue a node (current). Iterate over all neighbors of current. For each neighbor: If neighbor is not in visited: Create a clone of neighbor and Add to visited.
- Enqueue the neighbor (original) to the queue. Add the clone of the neighbor (from visited) to the cloned current node's neighbors list.
- Return the clone of the input node.
```
//BFS
class Solution {
    public Node cloneGraph(Node node) {
        if (node == null) {
            return null;
        }
        
        Map<Node, Node> visited = new HashMap<>();
        Queue<Node> queue = new LinkedList<>();
        
        Node clonedNode = new Node(node.val);
        visited.put(node, clonedNode);
        queue.add(node);
        
        while (!queue.isEmpty()) {
            Node current = queue.poll();
            
            for (Node neighbor : current.neighbors) {
                if (!visited.containsKey(neighbor)) {
                    visited.put(neighbor, new Node(neighbor.val));
                    queue.add(neighbor);
                }
                visited.get(current).neighbors.add(visited.get(neighbor));
            }
        }
        
        return clonedNode;
    }
}
//DFS
public Node cloneGraph(Node node) {

    if (node == null) return null;

    Map < Node, Node > map = new HashMap < > ();

    return dfs(node, map);

}

private Node dfs(Node node, Map < Node, Node > map) {

    if (map.containsKey(node)) {

        return map.get(node);

    }

    Node clone = new Node(node.val);

    map.put(node, clone);

    for (Node neighbor: node.neighbors) {

        clone.neighbors.add(dfs(neighbor, map));

    }

    return clone;

}
```
