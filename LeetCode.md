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
9. [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/description/) - O(long) & O(1)
- Solution: 1. Modified binary search approach 2. Get Pivot element, target==pivot and apply Binary search 2 times (0, pivot-1) and (pivot+1, array.length-1)
- The usual approach for rotated arrays is binary search. But how exactly do I adjust binary search here? Because the array isn't fully sorted, but it has two sorted parts.
- Let me outline the steps. The idea is to determine which half of the array is sorted and then check if the target is within that sorted half. If not, look in the other half.
- Find the mid element.
- Compare the start and mid elements to see which half is sorted.
- If nums[start] <= nums[mid], then the left half is sorted.
- Else, the right half is sorted.
- Once I know which half is sorted, check if the target is within that sorted half's range.
- If it is, adjust the search to that half.
- If not, adjust to the other half.
- Wait, let me think. For example, if the left half is sorted (nums[start] <= nums[mid]), then the target must be between nums[start] and nums[mid] for it to be in the left half. Otherwise, it's in the right half. Similarly, if the right half is sorted, check if the target is between mid and end.
- But wait, the array is rotated, so one of the two halves is guaranteed to be sorted. So each step, we can eliminate half of the possibilities based on where the target might lie.
```
class Solution {
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target) {
                return mid;
            }
            
            // Check if the left half is sorted
            if (nums[left] <= nums[mid]) {
                // Left half is sorted
                if (target >= nums[left] && target < nums[mid]) {
                    // Target is in the left half
                    right = mid - 1;
                } else {
                    // Target is in the right half
                    left = mid + 1;
                }
            } else {
                // Right half is sorted
                if (target > nums[mid] && target <= nums[right]) {
                    // Target is in the right half
                    left = mid + 1;
                } else {
                    // Target is in the left half
                    right = mid - 1;
                }
            }
        }
        
        return -1;
    }
}
```
10. Rotation Count in Rotated Sorted Array =  (Pivot + 1)
11. Split Array Largest Sum / Book allocation / Capacity To Ship Packages Within D Days
- Solution: 1. Binary search by taking start=max element, end=sum of all elements and sum=0,pieces=1 if sum+num > mid then sum=num,pieces++ else sum+=num after loop pieces>m - start=mid+1 else end=mid-1. After while loop return start instead of -1 (while loop inside another for loop)
12. [Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/) - O(long) & O(1)
- Solution: 1. Binary search approach 2. Linear search would work but that's O(n)
- Wait, the array is sorted but rotated. So the array has two parts, both sorted, and the minimum is the point where the second part starts
- So the plan is to find the pivot point where the rotation happened. Once I find that, the element at that index is the minimum.
- So during binary search, I can compare the middle element with the rightmost element. Because in a sorted and rotated array, the right half will be the part that's smaller if there's a rotation.
- Initialize left = 0, right = nums.length - 1.
- While left < right:
- Calculate mid = left + (right - left) / 2.
- Compare nums[mid] with nums[right].
- If nums[mid] > nums[right], the minimum must be in the right half, so set left = mid + 1.
- Else, the minimum is in the left half (including mid), so set right = mid.
- Once the loop ends, left == right, which is the index of the minimum.
```
class Solution {
    public int findMin(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] > nums[right]) { //That's where minimum element located, if array's rotated [3,4,5,1,2]
                left = mid + 1;
            } else {
                right = mid; //Else, the minimum is in the left half (including mid), so set right = mid
            }
        }
        
        return nums[left];
    }
}
```

## Cyclic Sort - O(n) and O(1)
1. Cyclic Sort 
- Question: Range (0,n) or (1,n) - Find the target element in an unsorted array
- Solution: 1. while loop i<array.length index=array[i]-1, if array[i]!=array[index] then swap else i++
2. Sort an Array - Sorting Algorithm - O(n*n) and O(1)
- Solution: 1. Bubble Sort - Nested loops i=0 to array.length and j=0 to array.length-i, if array[j]>array[j+1] swap and end ith loop flag break; 2. Insertion Sort - Nested loops i=0 to array.length and j=i+1;j>0;j--, if array[j]<array[j-1] swap else break; 3. Selection Sort - For loop i=0 to array.length last=array.length-i-1 getMaxIndex= range(0, last) swap(last,maxIndex)
3. [Missing Number / Find all numbers disappeared in an array](https://leetcode.com/problems/missing-number/description/) - O(n) & O(1)
- Solution: 1. Apply Cyclic Sort and then For loop if array[i]!=i+1 got and(i+1)
4. Find the Duplicate Number / Find all Duplicates in an array
- Solution: 1. Apply Cyclic Sort and then For loop if array[i]!=i+1 got ans(array[i])
5. Set Mismatch
- Solution: 1. Apply Cyclic Sort and then For loop if array[i]!=i+1 got ans(array[i], i+1)
6. First missing Positive
- Solution: 1. Apply Cyclic Sort and then For loop if array[i]!=i+1 got ans(i+1) if not found return array.length-1

## Bitwise 
1. [Missing Number](https://leetcode.com/problems/missing-number/description/) - O(n) & O(1)
- Question: Determine the missing number in an array containing distinct integers in the range [0, n], where n is the length of the array.
- Solution: 1. XOR Approach  2. Sum formula (Sum of the first n numbers is n*(n+1)/2. Wait, but here the numbers are from 0 to n. So the sum should be (n*(n+1))/2. Then subtract the sum of the array from that, and the result is the missing number) - O(n) & O(1) 3. Cyclic Sort
- Because XOR of a number with itself cancels out. So if we XOR all numbers from 0 to n, and then XOR with all elements in the array, the result will be the missing number.
- XOR of a number with itself is 0.
- XOR of a number with 0 is the number itself.
```
class Solution {
    public int missingNumber(int[] nums) {
        int missing = nums.length;
        for (int i = 0; i < nums.length; i++) {
            missing ^= i ^ nums[i];
        }
        return missing;
    }
}
```
2. [Reverse Bits](https://leetcode.com/problems/reverse-bits/description/) - O(1) & O(1)
- Question: Reverse the bits of a given 32-bit unsigned integer.
- Solution: 1. Take each bit from the input integer, starting from the least significant bit (LSB), and place it in the reversed position in the resulting integer.
- Let me think. So the idea is to take each bit from the input, starting from the least significant bit (rightmost), and place it in the reversed position in the result.
- For example, the first bit we take from the input (n & 1) will be the least significant bit of the result. But wait, no. Wait, when we reverse, the least significant bit becomes the most significant bit. So the first bit we extract (LSB of n) should be shifted to the 31st position (since Java's int is 32 bits, the leftmost bit is position 31).
- So here's the plan: Initialize a result variable as 0. Then, for each of the 32 bits:
- Shift the result to the left by 1 bit to make room for the next bit.
- Add the least significant bit of n to the result.
- Shift n to the right by 1 bit to process the next bit.
- Wait, but since Java uses signed right shifts for integers, shifting a negative number (n) to the right would fill with 1s. But since we need to treat n as unsigned, according to the problem statement. Oh right, the comment says "treat n as an unsigned value". So when we shift n, we should do an unsigned right shift.
- Wait, but in Java, the operator for unsigned right shift is >>>. So for each bit, we need to take the last bit of n and then shift n using >>> to ensure that the sign bit doesn't cause filling with 1s.
- Wait, but if we are shifting n to the right each time, then after 32 shifts, it'll become zero. So maybe we can loop 32 times, each time taking the least significant bit and appending it to the result, but shifting the result left each time.
- Let me outline the steps:
- Initialize result to 0.
- Loop i from 0 to 31:
- Shift result left by 1 (so that the next bit can be added to the least significant position)
- If the current least significant bit of n is 1, add 1 to result. Otherwise, add 0.
- Shift n right by 1 bit (using unsigned shift, so that even if the original n is negative, it's treated as unsigned, so new bits added on the left are 0s).
- Wait, but how do I get the least significant bit of n each time? Well, (n & 1) will give that. Then, after extracting, we shift n right by 1, but using >>> because we want to treat n as unsigned. So n = n >>> 1.
```
public class Solution {
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        int result = 0;
        for (int i = 0; i < 32; i++) {
            result = (result << 1) | (n & 1);
            n = n >>> 1;
        }
        return result;
    }
}
```
3. [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/description/) - O(k) & O(1) k is the number of 1 bits in the binary representation
- Question: Count the number of '1' bits in the binary representation of a given integer. This is known as the Hamming weight.
- Solution: 1. Two's complement form
- Let me think. One approach is to check each bit one by one. Like, iterate 32 times (since int is 32 bits in Java), and for each bit position, check if that bit is set. Then add to the count if it is.
- Alternatively, there's a trick using n & (n-1) to turn off the rightmost 1 bit. So every time you do n = n & (n-1), you remove the rightmost 1. Then you can count how many times you can do this until n becomes zero. That might be more efficient if there are fewer 1s.
- Which method is better? Well, for a 32-bit number, the first method always loops 32 times, but the second method loops as many times as the number of 1s. So the second is probably better in average cases, but for Java, maybe even 32 loops are acceptable.
- So let's think about both approaches.
- First approach: loop 32 times, check each bit.
- Initialize count to 0.
- For i from 0 to 31:
- if (n & (1 << i)) != 0, then increment count.
- But wait, in Java, shifting 1 to the left by 31 positions would give a negative number, because 1 << 31 is 2^31 which is larger than Integer.MAX_VALUE (which is 2^31 -1). So when you shift left, the sign bit (the 32nd bit) becomes 1, which makes it a negative number. So when you do a bitwise AND with n (which could be negative), how does that work?
- Alternatively, maybe we can use a mask and shift the bits of n instead. Like, check the least significant bit each time and then shift right. But shifting right in Java for integers is signed, so if n is negative, shifting right would fill with 1s. Wait, so if I do a right shift, but use a logical shift (unsigned) instead of arithmetic shift (signed), then maybe that's better. But Java's '>>' is arithmetic shift for signed integers. There's also '>>>' which is the logical right shift, filling with zeros regardless of the sign.
```
public class Solution {
    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            n &= (n - 1);
            count++;
        }
        return count;
    }
}
```

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
- Fixed Sliding window (Find Subarray/Substring of a fixed length) 2. Dynamic Sliding window (Longest/Shortest Subarray or Substring that satisfies the condition) - (e.g., max sum, longest substring with unique characters) - I remember that sliding window techniques are useful for substring problems. 
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
4. [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/description/) - O(n) & O(1)
- Question: Given a string s and an integer k, I can choose any k characters in the string and replace them with any other uppercase English letter. My goal is to find the length of the longest substring containing all repeating letters after performing those replacements.
- Solution: 1. Sliding Window
- I remember that sliding window techniques are useful for substring problems.
- The key idea is to find the longest window where the number of characters that need to be replaced is <=k. Because if in a window, the maximum frequency of a single character is maxFreq, then the number of replacements needed is window length - maxFreq. So as long as this value is <=k, the window is valid.
- So, the plan is to maintain a sliding window with left and right pointers. Expand the window by moving the right pointer, keep track of the maximum frequency of any character in the current window. If the window length minus the max frequency exceeds k, we need to move the left pointer to shrink the window until it's valid again.
- Wait, but how do we track the max frequency efficiently? Because every time the window changes, the frequencies might change. So maybe we can use a frequency array or hashmap to count the occurrences of each character in the current window. Then, as we expand the window, we update the counts and check if the current window is valid.
- Let's outline the steps:
- Initialize a frequency array (size 26 for uppercase letters) to count the characters in the current window.
- Use left and right pointers to represent the current window. Both start at 0.
- maxCount keeps track of the maximum frequency of any character in the current window.
- The result is the maximum window size found during the process.
- As we iterate with the right pointer:
- Increment the count of s.charAt(right)
- Update maxCount if the current character's count is higher than previous max
- Check if (window length - maxCount) >k. If yes, we need to move left forward, decrementing the count of the character at left, and increment left.
- Update the result with the maximum window size (right - left +1)
```
class Solution {
    public int characterReplacement(String s, int k) {
        int[] count = new int[26]; // Frequency count of each character in the current window
        int left = 0; // Left pointer of the sliding window
        int maxCount = 0; // Maximum frequency of a single character in the current window
        int maxLength = 0; // Result to store the maximum valid window length
        
        for (int right = 0; right < s.length(); right++) {
            char c = s.charAt(right);
            count[c - 'A']++;
            maxCount = Math.max(maxCount, count[c - 'A']); // Update max frequency
            
            // If the current window size minus the max frequency exceeds k, shrink the window
            while (right - left + 1 - maxCount > k) {
                count[s.charAt(left) - 'A']--; // Decrement the count of the leftmost character
                left++; // Move the left pointer to the right
            }
            
            // Update the maximum valid window length
            maxLength = Math.max(maxLength, right - left + 1);
        }
        
        return maxLength;
    }
}
```
5. [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/description/) - O(n) & O(1)
- Question: Find the smallest substring in a given string s that contains all the characters of another string t, including their frequencies.
- Solution: 1. Sliding window technique with two pointers and an array to track character counts.
- First, I remember that this is a sliding window problem. The sliding window technique is useful here because we're dealing with a substring and need to efficiently check if all characters in t are included with the right frequencies.
- Let me outline the steps.
- **Count the characters in t**: I need to know how many of each character are required. For example, if t is "AABC", then we need two A's, one B, and one C. So a frequency map for t's characters is necessary.
- **Sliding window approach**: Use two pointers, left and right, to represent the current window in s. Expand the window by moving the right pointer, and when all required characters are included (with sufficient counts), try to contract the window from the left to find the minimum length.
- **Keep track of required characters**: As we move the right pointer, decrement the count of characters from the frequency map. When a character's count in the map reaches zero, it means we've met the required count for that character. Keep a counter to track how many characters from t have been fully included in the window. When this counter equals the number of unique characters in t, the window is valid.
- **Contract the window**: Once a valid window is found, move the left pointer to see if we can minimize the window size while still maintaining validity. Each time the left pointer moves, increment the count in the frequency map again. If a character's count goes above zero, decrement the valid counter, indicating that the window is no longer valid, and we need to expand again.
- Wait, how do I manage the frequency counts? Let's see. For example, if t has two A's, and the window has three A's, then even if we remove one A (by moving left), the count would still be two, which is sufficient. So the frequency count can go negative, which indicates that we have more than enough of that character. Only when the count is exactly zero do we increment the valid counter.
- So the steps would be:
- Create a frequency map for t.
- Initialize left and right pointers at 0, and variables to track the minimum window start and length.
- Use a counter (required) which is the number of unique characters in t. When this counter reaches zero, all required characters are included in the window.
- Expand the window by moving right. For each character at right, if it's in the frequency map, decrement its count. If after decrementing, the count is zero, decrement the required counter.
- When required counter is zero, start moving left to try to minimize the window. For each character at left, if it's in the frequency map, increment its count. If the count becomes positive, increment the required counter, and break the contraction. Update the minimum window if the current window is smaller.
- Repeat until right reaches the end of s.
- Let me think about edge cases. For example, s is shorter than t. Then return empty. Also, characters in s that are not in t can be skipped. Also, multiple valid windows, need to track the smallest.
```
public class Solution {
    public String minWindow(String s, String t) {
        if (s == null || t == null || s.length() == 0 || t.length() == 0) {
            return "";
        }
        
        int[] map = new int[128]; //size array for ASCII characters
        for (char c : t.toCharArray()) {
            map[c]++;
        }
        
        int required = 0;
        for (int i = 0; i < map.length; i++) {
            if (map[i] > 0) required++;
        }
        
        int left = 0, right = 0, minLen = Integer.MAX_VALUE, start = 0;
        int count = 0;
        int[] current = new int[128];
        
        while (right < s.length()) {
            char c = s.charAt(right);
            if (map[c] > 0) {
                current[c]++;
                if (current[c] == map[c]) {
                    count++;
                }
            }
            right++;
            
            while (count == required) { //Minimize the window by moving left
                if (right - left < minLen) {
                    minLen = right - left;
                    start = left;
                }
                char leftChar = s.charAt(left);
                if (map[leftChar] > 0) { //Checking for minimizing the window
                    current[leftChar]--; //Doing left--
                    if (current[leftChar] < map[leftChar]) {
                        count--;
                    }
                }
                left++;
            }
        }
        
        return minLen == Integer.MAX_VALUE ? "" : s.substring(start, start + minLen);
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
2. [Move Zeroes](https://leetcode.com/problems/move-zeroes/description/) - O(n) & O(1)
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
7. [Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/description/) - O(n^2) & O(1)
- Question: Count all the palindromic substrings in a given string
- Solution: 1. Expand Around Centers (expanding around potential centers of palindromes) 2. Brute force approach would be to check every possible substring and see if it's a palindrome. But that's O(n^3) time because there are O(n^2) substrings and checking each takes O(n) time
- Initialize a count variable to 0.
- For each index i in the string:
a. Expand around i as the center (odd length palindromes).
b. Expand around i and i+1 as the center (even length palindromes).
- For each expansion, check if the left and right characters are equal. If yes, increment count and move left and right pointers outward.
- Return the total count.
```
class Solution {
    public int countSubstrings(String s) {
        int count = 0;
        for (int i = 0; i < s.length(); i++) {
            count += expand(s, i, i);    // Odd length palindromes
            count += expand(s, i, i + 1); // Even length palindromes
        }
        return count;
    }
    
    private int expand(String s, int left, int right) {
        int currentCount = 0;
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            currentCount++;
            left--;
            right++;
        }
        return currentCount;
    }
}
```
8. [3Sum](https://leetcode.com/problems/3sum/description/) - O(n^2) & O(1)
- Question: Find all unique triplets in an array that sum up to zero
- Solution: 1. Two-Pointer Technique 2. Brute force approach would check every possible triplet, but that's O(n^3) time
- Wait, maybe I can sort the array first. Sorting helps in handling duplicates and makes it easier to use two pointers for two-sum like problems. Right. Because once the array is sorted, I can fix one element and then find pairs in the remaining elements that sum to the negative of the fixed element. That way, their sum would be zero.
- So the plan is: sort the array. Then iterate through each element as the first element of the triplet. For each element, use two pointers (left and right) to find pairs in the remaining part of the array that sum to -nums[i]. But also need to avoid duplicates.
- Wait, but how to avoid duplicates. Like, if there are multiple same elements, how do I skip over them once I've processed the first occurrence.
- So during the iteration, for each i, if nums[i] == nums[i-1], we skip i. But wait, only when i > 0, to avoid index out of bounds. That way, the first occurrence is processed, and duplicates are skipped.
- Then, for the two pointers part: after fixing i, we set left to i+1, right to end of array. Then calculate the sum. If the sum is less than target (which is -nums[i]), then we need to increase left. If sum is greater, decrease right. If equal, add to the result.
- Once a valid triplet is found, we add it to the list. Then, we need to skip all duplicates for left and right. So, after adding, we increment left until nums[left] != nums[left+1] and left < right. Or perhaps, while left < right and nums[left] == nums[left+1], we move left. Similarly for right. But perhaps a better way is, once we have a triplet, we move left to the next unique value and right to the previous unique value.
- left starts at i+1, right at end.
- When sum is zero:
- add [nums[i], nums[left], nums[right]] to the result.
- then, increment left until nums[left] is different than current left value (so, while left < right and nums[left] == nums[left+1], left++)
- then, decrement right until nums[right] is different than current right (while left < right and nums[right] == nums[right-1], right--)
- then, move left once more (left++) and right once more (right--), because the current left and right were part of the triplet just added.
- Wait, but after that, the next iteration will check the new left and right.
- So the steps when sum == target:
a. add the triplet.
b. skip duplicates for left: while left < right and nums[left] == nums[left+1], left++. Then left++ once to move to the next new value.
c. skip duplicates for right: while left < right and nums[right] == nums[right-1], right--. Then right-- once.
```
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length < 3) return res;
        Arrays.sort(nums);
        int n = nums.length;
        
        for (int i = 0; i < n - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue; // Skip duplicate elements for i
            
            int target = -nums[i];
            int left = i + 1;
            int right = n - 1;
            
            while (left < right) {
                int sum = nums[left] + nums[right];
                if (sum < target) {
                    left++;
                } else if (sum > target) {
                    right--;
                } else {
                    List<Integer> triplet = Arrays.asList(nums[i], nums[left], nums[right]);
                    res.add(triplet);
                    // Skip duplicates for left
                    while (left < right && nums[left] == triplet.get(1)) left++;
                    // Skip duplicates for right
                    while (left < right && nums[right] == triplet.get(2)) right--;
                }
            }
        }
        return res;
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
3. [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/description/) -  O(nlogn) & O(1)
- Question: Determine the minimum number of intervals to remove so that the remaining intervals do not overlap.
- Solution: 1. Greedy algorithm approach
- So the problem is similar to scheduling intervals without overlaps, right? Like, we want as many intervals as possible that don't overlap, and the minimum to remove would be the total minus that maximum number. So the key is to find the maximum number of non-overlapping intervals. Then subtract that from the total number to get the answer.
- Wait, right. So the problem reduces to finding the largest possible set of non-overlapping intervals, which is the same as the interval scheduling problem. The classic greedy algorithm for that is to sort the intervals based on their end times and then pick the earliest ending one each time, ensuring the next one starts after the previous ends.
- Let me recall. For example, if we sort intervals by end, then select the first interval. Then select the next one whose start is >= the end of the previous. Repeat until all are processed. That should give the maximum possible non-overlapping intervals.
- So the steps would be:
- Sort the intervals based on their end times.
- Initialize a count of 1 (since we select the first interval), and set the current end to the end of the first interval.
- Iterate through the sorted intervals:
a. If the current interval's start is >= current end, increment count and update current end to this interval's end.
b. Else, skip it (as it overlaps).
- The answer is total intervals minus the count.
- Wait, but how to handle the sorting when the input is an array of arrays in Java? Oh right, we can sort using a custom comparator. For example, in Java, we can use Arrays.sort with a comparator that compares the end times.
```
class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length == 0) return 0;
        
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[1], b[1]));
        
        int count = 1;
        int currentEnd = intervals[0][1];
        
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] >= currentEnd) {
                count++;
                currentEnd = intervals[i][1];
            }
        }
        
        return intervals.length - count;
    }
}
```

## Backtracking
1. [Combination Sum](https://leetcode.com/problems/combination-sum/description/) - O(N^(T/M + 1)) & O(T/M)  N is the number of candidates, T is the target, and M is the minimum value among the candidates. Space complexity due to the recursion stack depth
- Question: Find all unique combinations in a given list of distinct integers where the numbers sum up to a specified target. Each number in the list can be used multiple times.
- Solution: 1. Backtracking
- Hmm, so how do I approach this? Well, this feels like a backtracking problem. Because we need to explore all possible combinations and backtrack when a path doesn't lead to a solution.
- Wait, for example, if the candidates are sorted, and the current number is larger than the remaining target, then adding it would exceed, so we can break out of the loop. That's a possible optimization.
- So to avoid duplicates, once we choose to move to the next candidate (say index i+1), we can't go back to candidates at index i. So for example, if we have candidates sorted, and in the current combination, we have started using candidate at index i, then the next step can only consider candidates from index i onwards. That way, we avoid permutations of the same numbers.
- So the backtrack function will have parameters: start index, current combination, remaining target. The steps would be:
- Sort the candidates array (to allow pruning when candidates[i] > remaining target).
- Initialize a list to hold the result.
- Define a backtrack function that takes start index, current combination, remaining target.
- For each candidate from start index to end:
a. If the candidate's value is greater than the remaining target, break the loop (since sorted, no further candidates can be used).
b. Add the candidate to the current combination.
c. Subtract candidate's value from remaining target.
d. If remaining target is zero, add the current combination to the result.
e. Else, recursively call the backtrack function with the same start index (since we can reuse), and the new remaining target.
f. Remove the last candidate from the current combination (backtrack).
```
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(candidates); // Sort the candidates to enable pruning
        backtrack(candidates, target, 0, new ArrayList<>(), result);
        return result;
    }
    
    private void backtrack(int[] candidates, int remaining, int start, List<Integer> current, List<List<Integer>> result) {
        if (remaining == 0) {
            result.add(new ArrayList<>(current)); // Add a copy of the current combination to the result
            return;
        }
        
        for (int i = start; i < candidates.length; i++) {
            if (candidates[i] > remaining) {
                break; // Prune the remaining candidates as they are larger than the remaining target
            }
            current.add(candidates[i]); // Include the current candidate
            backtrack(candidates, remaining - candidates[i], i, current, result); // Recur with the same start index to allow reuse
            current.remove(current.size() - 1); // Backtrack by removing the last added candidate - When it's break. So backtrack, remove last one's candidate > target.
        }
    }
}
```

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
6. [Word Break](https://leetcode.com/problems/word-break/description/) - O(n * maxLen) & O(n), n is the length of the string and maxLen is the maximum word length in the dictionary
- Question: Determine if a given string s can be segmented into a sequence of one or more words from a provided dictionary wordDict. The same word from the dictionary can be reused multiple times in the segmentation.
- Solution: 1. Dynamic programming 2. Brute force way would be to try every possible split. Like, for each position in the string, check if the substring up to that position is in the dictionary, and then recursively check the remaining part. But that's probably going to be very inefficient for longer strings because of overlapping subproblems
- Dynamic programming approach: Maybe I can create a boolean array dp where dp[i] indicates whether the substring s[0..i-1] can be segmented into dictionary words. Then, for each position i, I check all possible j < i, and if dp[j] is true and the substring from j to i is in the dictionary, then set dp[i] to true. The base case would be dp[0] = true, meaning the empty string is trivially segmented.
- Let me outline the steps. Initialize a dp array of size s.length() + 1. Set dp[0] to true. Then iterate through the string from 1 to n (length of s). For each i, check all j from 0 to i-1. If dp[j] is true and the substring s[j..i-1] is present in the wordDict, then set dp[i] to true and break the loop for that i. Finally, return dp[n].
- Alternatively, we can convert the wordDict into a set for O(1) lookups. That way, checking if a substring is present becomes O(1). So converting the list to a set first would help. Then the time complexity becomes O(n^2) for the loops, plus O(m) for converting the list to a set, which is acceptable.
- Another optimization: instead of checking all j for each i, maybe we can limit j to i - maxLength, where maxLength is the maximum word length in the dictionary. Because if the substring from j to i is longer than the longest word in the dictionary, it's impossible for it to be in the dictionary. So for each i, we can check j from i - maxLength to i-1, but making sure that j doesn't go below 0. This can reduce the number of checks per i.
- So first, find the maximum word length in wordDict. Then, for each i, j starts from max(0, i - maxLength) to i-1. That could save some time.
- So the steps would be:
- Convert wordDict into a set for O(1) lookups.
- Find the maximum word length in wordDict. If wordDict is empty, return false.
- Initialize dp array of size n+1 (n is s.length()), set dp[0] = true.
- For each i from 1 to n:
a. For j from i-1 down to max(0, i - maxLength):
b. If dp[j] is true and s.substring(j, i) is in the set, then set dp[i] to true and break.
- Return dp[n].
```
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        if (s.isEmpty()) {
            return false;
        }
        Set<String> wordSet = new HashSet<>(wordDict);
        if (wordSet.isEmpty()) {
            return false;
        }
        
        int maxLen = 0;
        for (String word : wordSet) {
            maxLen = Math.max(maxLen, word.length());
        }
        
        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;
        
        for (int i = 1; i <= n; i++) {
            int start = Math.max(0, i - maxLen);
            for (int j = i - 1; j >= start; j--) {
                if (dp[j] && wordSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        
        return dp[n];
    }
}
```
7. [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/description/) - O(n) & O(1)
- Question: Finding the subarray with the largest product in an integer array
- Solution: 1. Dynamic programming similar to the maximum subarray sum
- So the idea is to keep track of both the maximum and minimum products at each step. For each number in the array, calculate the new possible max and min by considering the current number multiplied by the previous max or min, or just the current number itself (in case the previous products are zero or the current number is better on its own).
- Initialize variables to keep track of current max and min products. Also, a variable to track the overall maximum product found so far.
- Iterate through each number in the array.
- For each number, calculate the temp_max (since the current max will be updated before min, but min depends on the previous max, so we need a temporary variable to store the old max).
- The new current max is the maximum of (temp_max * current number, current min * current number, current number).
- Similarly, the new current min is the minimum of (temp_max * current number, current min * current number, current number).
- Update the overall max with the current max if it's larger.
- Wait, why do we need to consider all three possibilities (prev max * current, prev min * current, current)? Because the current number could be negative, and multiplying a negative (prev min) by a negative could give a larger product. Also, if the current number is higher than any product, like if previous product was zero and current is positive, we start fresh.
```
class Solution {
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        
        int currentMax = nums[0];
        int currentMin = nums[0];
        int overallMax = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            int tempMax = currentMax * nums[i];
            int tempMin = currentMin * nums[i];
            
            currentMax = Math.max(Math.max(tempMax, tempMin), nums[i]);
            currentMin = Math.min(Math.min(tempMax, tempMin), nums[i]);
            
            overallMax = Math.max(overallMax, currentMax);
        }
        
        return overallMax;
    }
}
```
8. [Unique Paths](https://leetcode.com/problems/unique-paths/description/) - O(mn) & O(n)
- Determine the number of unique paths a robot can take from the top-left corner to the bottom-right corner of an m x n grid, where the robot can only move right or down.
- Solution: 1. Dynamic programming 2. Combination of (m+n-2) choose (m-1) or (n-1), since combinations are symmetric.
- In this case, the robot has to make a total of (m-1) down moves and (n-1) right moves. Because to go from top-left to bottom-right, you need to go down (m-1) times and right (n-1) times. The total steps are (m-1 + n-1) = m+n-2 steps. The number of unique paths would be the number of ways to choose which of these steps are down (or right), right?
- So the general solution is to compute (m+n-2) choose min(m-1, n-1). Because combinations can be calculated using factorials, but directly computing factorials for large numbers might be computationally expensive and could lead to integer overflow issues. But maybe for the problem constraints, it's manageable.
- Wait, but what are the constraints? The problem says m and n can be up to 100. So m+n-2 could be up to 198, and choosing 99. The value of C(198,99) is a huge number. But in Java, using integers might not work because the numbers could exceed the maximum integer value. So perhaps we need to use a dynamic programming approach instead of combinatorial math to avoid big numbers.
- Alternatively, maybe the problem expects the DP approach. Let's think about that. For a grid, the number of unique paths to a cell (i,j) is the sum of the paths to the cell above it (i-1,j) and the cell to the left (i,j-1). The base case is the first row and first column, which have only 1 path each.
- So for a DP table, we can create a 2D array where dp[i][j] represents the number of paths to reach (i,j). Then, dp[i][j] = dp[i-1][j] + dp[i][j-1]. The base case is dp[0][j] = 1 for all j and dp[i][0] =1 for all i.
- But since m and n can be up to 100, the 2D array approach would have a time and space complexity of O(m*n), which should be manageable. But maybe we can optimize space, since each row only depends on the previous row or the current row.
- Alternatively, since each cell only needs the value from the top and left, we can use a 1D array. Let's say we have a dp array of size n, where dp[j] represents the number of paths to reach the current row's jth column. For each row, starting from the first, we update the dp array. For the first row, all dp[j] are 1. For subsequent rows, for each column j >0, dp[j] += dp[j-1]. Wait, maybe that's possible.
- So the algorithm would be:
- Initialize dp array of size n, filled with 1s.
- Then, for each row from 1 to m-1:
- for each column from 1 to n-1:
- dp[j] = dp[j] + dp[j-1]
- At the end, dp[n-1] is the answer.
- Yes, that makes sense. Because for each new row, the first column is always 1 (since you can only come from above, which is 1 way). Then each subsequent column adds the current dp[j] (which is the value from the previous row, i.e., above) and dp[j-1] (which is the value from the left in the current row).
```
class Solution {
    public int uniquePaths(int m, int n) {
        int[] dp = new int[n];
        // Initialize the dp array with 1s for the first row
        for (int j = 0; j < n; j++) {
            dp[j] = 1;
        }
        // Process each subsequent row
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] += dp[j - 1];
            }
        }
        return dp[n - 1];
    }
}
```
9. [House Robber](https://leetcode.com/problems/house-robber/description/) - O(n) & O(1)
- Question: Determine the maximum amount of money a robber can steal from houses arranged in a line without alerting the police by robbing two adjacent houses.
- Solution: 1. DP with 2 variables 2. DP with 1D array
- The key insight is that for each house, the robber has two choices: either rob the current house and add its value to the maximum amount obtained from houses two steps back, or skip the current house and retain the maximum amount obtained from the previous house. By iteratively updating two variables that track the maximum amounts from the previous house and the house two steps back, we can efficiently compute the solution without using additional space for an array.
```
public class Solution {
    public int rob(int[] nums) {
        int prev = 0, curr = 0;
        for (int num : nums) {
            int temp = curr;
            curr = Math.max(curr, prev + num); //skip or rob
            prev = temp;
        }
        return curr;
    }
}
```

## Ad-hoc 
- Problems that don’t fit into a standard category and require a unique approach like Swapping, Sorting, HashMap, Manipulation
1. [Rotate Image](https://leetcode.com/problems/rotate-image/description/) - O(n^2) & O(1)
- Question: Rotating an n x n matrix by 90 degrees clockwise in-place
- Solution: 1. Two-step approach that involves transposing the matrix and then reversing each row.
-  So for a 90-degree clockwise rotation, we can transpose the matrix and then reverse each row. That's a neat trick. So the steps would be:
- Transpose the matrix (swap matrix[i][j] with matrix[j][i] for all i < j)
- Reverse each row of the transposed matrix.
```
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        
        // Step 1: Transpose the matrix
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) { //[0][1] = [1][0]
                // Swap elements at (i, j) and (j, i)
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        
        // Step 2: Reverse each row
        for (int i = 0; i < n; i++) {
            int left = 0;
            int right = n - 1;
            while (left < right) {
                // Swap elements in row i from left and right ends
                int temp = matrix[i][left];
                matrix[i][left] = matrix[i][right];
                matrix[i][right] = temp;
                left++;
                right--;
            }
        }
    }
}
```
2. [Group Anagrams](https://leetcode.com/problems/group-anagrams/description/) - O(n * klogk) & O(n * k) n is the number of strings and k is the maximum length of a string
- Question: Need to group anagrams from a given list of strings. So anagrams are words that have the same characters but in different orders. For example, "eat" and "tea" are anagrams. The goal is to group all such anagrams together in a list.
- Solution: 1. Using a hash map where the keys are sorted versions of the strings, and the values are lists of original strings that are anagrams of each other. 2. Alternatively, maybe count the frequency of each character in the string and use that as a key. For example, for "aab", the key could be something like "a2b1". That way, we avoid sorting. But building the frequency count might take O(k) time per string, and then converting it into a string key. Comparing the two approaches: sorting vs. frequency count.
- Hmm, how do I determine if two strings are anagrams? One way is to sort the characters in each string. If two sorted strings are equal, they are anagrams. For example, sorting "eat" gives "aet", and sorting "tea" also gives "aet". So using the sorted version as a key in a hash map makes sense.
- Right, so the plan is:
- Iterate through each string in the input array.
- For each string, sort its characters to create a key.
- Use a hash map where the key is the sorted string, and the value is a list of original strings that are anagrams.
- After processing all strings, collect all the values from the hash map into the result list.
- Let's outline the steps:
- Create a hash map (maybe a HashMap in Java) where the key is the sorted string, and the value is a list of strings that are anagrams.
- For each string in strs:
- Convert it to a char array.
- Sort the char array.
- Convert it back to a string to use as the key.
- Add the original string to the corresponding list in the map.
- Finally, collect all the values of the map into the result list.
```
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String s : strs) {
            char[] chars = s.toCharArray();
            Arrays.sort(chars);
            String key = new String(chars);
            if (!map.containsKey(key)) {
                map.put(key, new ArrayList<>());
            }
            map.get(key).add(s);
        }
        return new ArrayList<>(map.values());
    }
}
```
3. [Spiral Matrix](https://leetcode.com/problems/spiral-matrix/description/) - O(nm) & O(1)
- Question: Traverse a matrix in spiral order, starting from the top-left corner and moving right, then down, then left, and then up, repeating this process while shrinking the boundaries of the matrix until all elements are covered.
- Solution: 1. Spiral order traversal of a matrix - Four pointers top, bottom, left and right
- Hmm, how does the spiral order work? Like, you go right, then down, then left, then up, and repeat, but each time the boundaries shrink.
- So maybe I need to keep track of the current boundaries: top row, bottom row, left column, right column. Then loop through the layers, moving in each direction and adding elements to the result list. But how to handle the layers and directions properly?
- Let me think. For example, for a 3x3 matrix, the spiral order would start at top left, move right to the end, then down, then left, then up. Then the next layer is the center element. But how to manage when there's only one row or column left.
- First, I need to initialize the boundaries. Let's say top starts at 0, bottom at matrix.length-1, left at 0, right at matrix[0].length-1. Then, while top <= bottom and left <= right, we process the current layer.
- For each layer:
- Traverse from left to right along the top row. Then increment top.
- Traverse from top to bottom along the right column. Then decrement right.
- If top <= bottom (to handle cases where there's a single row), traverse from right to left along the bottom row. Then decrement bottom.
- If left <= right (to handle single column), traverse from bottom to top along the left column. Then increment left.
- So, the algorithm steps would be:
- Initialize top=0, bottom=matrix.length-1, left=0, right=matrix[0].length-1.
- Create a result list.
- While top <= bottom and left <= right:
- For i from left to right: add matrix[top][i] to result. Then top++.
- For i from top to bottom: add matrix[i][right] to result. Then right--.
- Check if top <= bottom. If yes, loop i from right to left: add matrix[bottom][i]. Then bottom--.
- Check if left <= right. If yes, loop i from bottom to top: add matrix[i][left]. Then left++.
- Yes. So the steps are:
- Go right along the top row, then increment top.
- Go down the right column, then decrement right.
- If top <= bottom, go left along the bottom row, then decrement bottom.
- If left <= right, go up along the left column, then increment left.
- Repeat until the top > bottom or left > right.
```
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        if (matrix == null || matrix.length == 0) return res;

        int top = 0;
        int bottom = matrix.length - 1;
        int left = 0;
        int right = matrix[0].length - 1;

        while (top <= bottom && left <= right) {
            // Traverse from left to right on the top row
            for (int i = left; i <= right; i++) {
                res.add(matrix[top][i]);
            }
            top++;

            // Traverse from top to bottom on the right column
            for (int i = top; i <= bottom; i++) {
                res.add(matrix[i][right]);
            }
            right--;

            // Traverse from right to left on the bottom row, if there are rows left
            if (top <= bottom) {
                for (int i = right; i >= left; i--) {
                    res.add(matrix[bottom][i]);
                }
                bottom--;
            }

            // Traverse from bottom to top on the left column, if there are columns left
            if (left <= right) {
                for (int i = bottom; i >= top; i--) {
                    res.add(matrix[i][left]);
                }
                left++;
            }
        }

        return res;
    }
}
```
4. [Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/description/) - O(mn) & O(1)
- Question: Need to set the entire row and column of a matrix to zero if any element in that row or column is zero.
- Solution: 1. In-Place Marking
- The key idea is to use the first row and first column of the matrix itself to mark which rows and columns need to be zeroed. This avoids using additional space for storing flags. Here are the steps:
- Check for Zeros in First Row and Column: Determine if the first row or column originally contains any zeros and store this information.
- Mark Zeros Using First Row and Column: Iterate through the matrix starting from the second row and column. If an element is zero, mark its corresponding position in the first row and column.
- Set Rows and Columns to Zero: Use the marks in the first row and column to set the appropriate rows and columns to zero, excluding the first row and column initially.
- Handle First Row and Column: Finally, zero out the first row and/or column if they originally contained any zeros.
```
public class Solution {
    public void setZeroes(int[][] matrix) {
        boolean firstRowZero = false;
        boolean firstColZero = false;
        int m = matrix.length;
        if (m == 0) return;
        int n = matrix[0].length;
        
        // Check if first row has zero
        for (int j = 0; j < n; j++) {
            if (matrix[0][j] == 0) {
                firstRowZero = true;
                break;
            }
        }
        
        // Check if first column has zero
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) {
                firstColZero = true;
                break;
            }
        }
        
        // Mark zeros using first row and column
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        
        // Set rows to zero based on first column
        for (int i = 1; i < m; i++) {
            if (matrix[i][0] == 0) {
                for (int j = 1; j < n; j++) {
                    matrix[i][j] = 0;
                }
            }
        }
        
        // Set columns to zero based on first row
        for (int j = 1; j < n; j++) {
            if (matrix[0][j] == 0) {
                for (int i = 1; i < m; i++) {
                    matrix[i][j] = 0;
                }
            }
        }
        
        // Handle first row
        if (firstRowZero) {
            for (int j = 0; j < n; j++) {
                matrix[0][j] = 0;
            }
        }
        
        // Handle first column
        if (firstColZero) {
            for (int i = 0; i < m; i++) {
                matrix[i][0] = 0;
            }
        }
    }
}
```

## Greedy Algorithm
1. Longest Consecutive Sequence
- Solution: 1. HashSet fill with array element. For num if (!set.contains(num - 1)) currentNum=num while (set.contains(currentNum + 1)

2. [Jump Game](https://leetcode.com/problems/jump-game/description/) - O(n) & O(1)
- Question: Determine if we can jump from the start of an array to its end. The array elements represent the maximum jump length from that position.
- Solution: 1. Greedy Algorithm 2. Brute force approach would be to check all possible jumps, but that's going to be O(2^n) time
- Wait, maybe a greedy approach. Because I don't need to track the exact path, just whether it's possible. So perhaps I can keep track of the farthest position I can reach. Let's see: starting from the first index, the maximum reach is 0 + nums[0]. Then for each subsequent index, if I can reach that index (i.e., it's within the current maximum reach), then I update the maximum reach to be the max of current max and i + nums[i]. If at any point the max reach is >= last index, return true. If I finish the loop and the max reach didn't reach the end, return false.
- So the steps are:
- Initialize max_reach to 0.
- Iterate through each index i from 0 to n-1:
a. If i > max_reach, return false (can't reach this index)
b. Update max_reach to max(max_reach, i + nums[i])
c. If max_reach >= last index (n-1), return true early.
- After loop, check if max_reach >= last index. If yes, return true, else false.
- Wait, but in step 2a, why check if i > max_reach? Because if the current index is beyond the maximum we can reach, then we can't proceed further. 
```
public class Solution {
    public boolean canJump(int[] nums) {
        int maxReach = 0;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (i > maxReach) { //maxReach < i can't reach
                return false;
            }
            maxReach = Math.max(maxReach, i + nums[i]);
            if (maxReach >= n - 1) { //return early. we can skip this step as well
                return true;
            }
        }
        return maxReach >= n - 1; //return true; able to reach the end
    }
}
```

## Linked List
- Fast and Slow pointers - O(n) & O(1) | Reversal of Linked List using Three pointers - O(n) & O(1)
1. [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/description/) - O(n) & O(1)
- Question: Determine if a linked list has a cycle
- Solution: 1. Two-pointer method, also known as Floyd's Tortoise and Hare algorithm (Slow and Fast pointers) 2. HashSet - while(current != null) if(set.contains(current)) return true; set.add(current) current=current.next; - O(n) & O(n)
- The idea is to have a slow pointer and a fast pointer. If there's a cycle, the fast pointer will eventually catch up to the slow one. If there's no cycle, the fast pointer will reach the end of the list.
- Wait, how do I implement that? Let's see. Start both pointers at the head. Then, in each iteration, the slow pointer moves one step, and the fast pointer moves two steps. If they ever meet, there's a cycle. If the fast pointer hits null, then there's no cycle.
- Initialize both slow and fast pointers to head.
- While fast is not null and fast.next is not null:
a. Move slow by one.
b. Move fast by two.
c. If slow and fast are the same node, return true (cycle exists).
- If the loop exits without returning true, return false.
```
public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null) return false;
        
        ListNode slow = head;
        ListNode fast = head;
        
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            
            if (slow == fast) {
                return true;
            }
        }
        
        return false;
    }
}
```
2. Middle of the Linked List
- Solution: 1. Fast and Slow pointers, return slow; - Single parse 2. Find the length of the LL count=0 and Apply For loop again i=0 to count/2, return current - Two times for loop - O(n) & O(1)
3. Happy Number
- Question: Given num 14 = 1^2+4^2 = 17 ... If it ends with '1' it means Happy, else the loop again 17 is not a happy number
- Solution: 1. Fast and Slow pointers - slow=n,fast=getNext(n) while(fast!=1 && slow != fast) slow,fast increment after while return fast==1
  2. HashSet - while(n>0) findSquare and check in the if set.contains(square) return true otherwise set.add(squre)  after while return false
4. [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/description/) - O(n) & O(1)
- Question: 
- Solution: 1. Three-pointers - prev=null,current=head while(current!=null) next=current.next, current.next=prev, prev=current, current=next after while loop return prev 2. Copy all the elements to an array and reverse it. Update LL according to the array - O(n) & O(n)
5. [Reorder List](https://leetcode.com/problems/reorder-list/description/) -O(n) & (1)
- Question: Reorder a linked list such that the nodes are arranged in a specific way: the first node is followed by the last node, then the second node is followed by the second last, and so on
- Solution: 1. Splitting the list into two halves, reversing the second half, and then merging the two halves alternately. 2. Copy to the new array, modify and update the list
- Find the middle of the linked list. The middle can be found using the slow and fast pointers. Once the fast pointer reaches the end, the slow pointer is at the middle.
- Split the list into two halves. So after finding the middle, we need to split. Let's say the middle is at slow. Then the second half starts at slow.next. But wait, we need to break the link between the first and second halves. So after finding the middle, the first half's end (slow) should have its next set to null. Then the second half is slow.next, which is the head of the second part.
- Reverse the second half. How to reverse a linked list? We can do that iteratively. Initialize prev as null, current as the head of the second half. Then for each step, save the next node, set current.next to prev, then move prev and current forward. Once done, prev is the new head of the reversed list.
- Merge the two lists alternately. So take one node from the first half, then one from the reversed second half, and so on.
- Find the middle node using slow and fast pointers.
- Split the list into two halves.
- Reverse the second half.
- Merge the two halves by alternating nodes.
```
class Solution {
    public void reorderList(ListNode head) {
        if(head == null || head.next == null) {
            return;
        }

        ListNode mid = middleNode(head);
        ListNode headSecond = reverseList(mid);
        ListNode headFirst = head;

        while(headFirst != null && headSecond != null) {
            ListNode temp = headFirst.next;
            headFirst.next = headSecond;
            headFirst = temp;

            temp = headSecond.next;
            headSecond.next = headFirst;
            headSecond = temp;
        }

        if(headFirst != null) { // Pointing tail to null
            headFirst.next = null;
        }
    }

    public ListNode middleNode(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;

        while(fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    public ListNode reverseList(ListNode head) {
        if(head == null) {
            return head;
        }
        ListNode prev = null;
        ListNode present = head;
        ListNode next = present.next;

        while(present != null) {
            present.next = prev;

            prev = present;
            present = next;
            if(next != null) {
                next = next.next;
            }
        }
        return prev;
    }
}

```
6. [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/) - O(n) & O(1)
- Question: Remove the nth node from the end of a linked list
- Solution: 1. Fast and Slow pointers 2. Find the length of the list. Then compute the position to delete, which is (length - n) - Iterate through the list to find the length. Then compute the position, then traverse again to that position and delete.
- The idea is to have a fast pointer that is n steps ahead of the slow pointer. When the fast pointer reaches the end, the slow pointer will be at the node before the one to delete.
- Let's say we have a dummy node pointing to the head. Then, we have two pointers, slow and fast. We move fast n steps ahead. Then, we move both until fast.next is null. Then, slow.next is the node to delete.
- Create a dummy node pointing to head. This helps handle cases where the head needs to be deleted(We're using slow to remove the node)
- Initialize slow and fast pointers at dummy.
- Move fast n steps ahead. So after that, fast is at node n steps ahead.
- Then, move both slow and fast until fast reaches the last node (fast.next is null). At this point, slow is pointing to the node before the one to be deleted.
- Then, set slow.next = slow.next.next.
```
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0); //Create a dummy node that points to the head of the list. This helps handle edge cases, such as removing the head node.
        dummy.next = head;
        ListNode slow = dummy;
        ListNode fast = dummy;
        
        // Move fast pointer n steps ahead
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        
        // Move both pointers until fast reaches the last node
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        
        // Remove the nth node from the end
        slow.next = slow.next.next;
        
        return dummy.next;
    }
}
```
7. [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/description/) - O(n+m) & O(1)
- Solution: 1. Two-pointer approach along with a dummy node to simplify the merging process.
- First, I remember that when merging two sorted lists, you typically compare the current nodes of each list and choose the smaller one to add to the merged list. So, maybe I can use a dummy node to simplify the process. The dummy node will help avoid dealing with edge cases where the merged list is empty initially.
- Initialize a dummy node and a current pointer pointing to the dummy.
- While both list1 and list2 are not null:
a. Compare the values of list1 and list2.
b. Attach the smaller node to current.next.
c. Move the current pointer and the selected list's pointer forward.
- Once one list is exhausted, attach the remaining nodes from the other list.
- Return dummy.next as the merged list.
```
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode();
        ListNode current = dummy;
        
        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                current.next = list1;
                list1 = list1.next;
            } else {
                current.next = list2;
                list2 = list2.next;
            }
            current = current.next;
        }
        
        // Attach the remaining elements of list1 or list2
        current.next = (list1 != null) ? list1 : list2;
        
        return dummy.next;
    }
}
```
8. [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/description/) O(nlogk) & O(k) heap stores at most k nodes and each heap operation takes O(logk) time
- Solution: 1. Use a priority queue (heap) 2. Merging two sorted lists can be done efficiently using a two-pointer approach. But when there are k lists, doing this pairwise might not be the most efficient. If I merge two lists at a time, like merging the first two, then merging the result with the third, and so on, that could work. But the time complexity might be high. For example, if each merge is O(n), and there are k lists, then the total time could be O(kN)
- The idea would be to take the smallest element from each of the k lists and add them to the heap. Then, extract the smallest element from the heap, add it to the merged list, and then add the next element from the list that the extracted node came from. This way, we always pick the next smallest element available across all lists.
- Create a priority queue that orders nodes by their value.
- Add all the non-null head nodes from the lists array into the priority queue.
- Create a dummy node to serve as the starting point of the merged list.
- Use a pointer to build the merged list by repeatedly extracting the smallest node from the queue.
- After extracting a node, add the next node from its original list to the queue (if it exists).
- Continue until the queue is empty.
- Return the next of the dummy node as the merged list.
- But wait, in Java, the PriorityQueue requires a Comparator. Since ListNode doesn't implement Comparable, I need to provide a comparator that compares the values of the nodes. So, the comparator would be something like (a, b) -> a.val - b.val. But I need to handle nulls? No, because we are only adding non-null nodes to the queue.
```
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        
        PriorityQueue<ListNode> heap = new PriorityQueue<>((a,b) -> a.val - b.val);
        
        
        for (ListNode list : lists) {
            if (list != null) {
                heap.offer(list);
            }
        }
        
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        
        while (!heap.isEmpty()) {
            ListNode node = heap.poll();
            current.next = node;
            current = current.next;
            
            if (node.next != null) {
                heap.offer(node.next);
            }
        }
        
        return dummy.next;
    }
}
```

## Stack
1. [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/description/) - O(n) & O(n)
- Question: Determine if a string of parentheses is valid
- Solution: 1. Stack data structures
- Every time I encounter an opening bracket, I push it onto the stack. And when I find a closing bracket, I check if the top of the stack is the corresponding opening bracket. If not, it's invalid.
- Initialize a stack.
- Iterate through each character in the string.
- If the character is an opening bracket (like '(', '{', '['), push it onto the stack.
- If it's a closing bracket, check if the stack is empty. If it is, return false because there's nothing to match with.
- Pop the top element from the stack and check if it's the matching opening bracket. If not, return false.
- At the end, if the stack is empty, return true; else, return false (because there are unmatched opening brackets left).
- But wait, how to map the closing brackets to their corresponding opening ones? Maybe use a hashmap or a simple switch case. Like, when we see a ')', we check if the last opened was '('.
- Alternatively, we can use a map where the key is the closing bracket and the value is the corresponding opening bracket. That way, when we get a closing bracket, we can check if the popped element matches the value.
- Let me think about the data structures. So, for Java, I can use a Stack<Character> or maybe Deque as a stack.
```
class Solution {
    public boolean isValid(String s) {
        Deque<Character> stack = new ArrayDeque<>();
        Map<Character, Character> bracketMap = new HashMap<>();
        bracketMap.put(')', '(');
        bracketMap.put('}', '{');
        bracketMap.put(']', '[');
        
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (bracketMap.containsKey(c)) {
                // Closing bracket
                if (stack.isEmpty() || stack.pop() != bracketMap.get(c)) {
                    return false;
                }
            } else {
                // Opening bracket
                stack.push(c);
            }
        }
        
        return stack.isEmpty();
    }
}
```

## Heap (Priority Queue)
1. [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/description/) - O(log n) & O(n)
- Question: Find the median from a data stream. Hmm, the median is the middle value when the numbers are ordered. If there's an odd number of elements, it's the middle one. If even, the average of the two middle ones.
- Solution: 1. Using two heaps 2. If I just keep a list and add numbers, then when finding the median, I have to sort it every time. But that would be O(n log n) for each insertion
- I remember that using two heaps can help here. Oh right! One max-heap to store the lower half of the numbers and a min-heap for the upper half. That way, the top of the max-heap is the largest of the lower half, and the top of the min-heap is the smallest of the upper half. The median would depend on the sizes of the heaps.
- Let me think. The max-heap (let's call it 'low') will hold the first half, and the min-heap ('high') the second. When the total count is even, both heaps have the same size, and the median is the average of their tops. If odd, the low heap has one more element, so its top is the median.
- So when adding a number, I need to balance the heaps. Let's see. When adding a new number, first add it to the low heap. Then, to make sure that the max of low is <= the min of high, I should move the max of low to high. Wait, but maybe it's better to first add to one heap and then balance. Wait, here's the standard approach:
- Maintain that the size of 'low' is either equal to or one more than 'high'.
- When inserting a new number:
a. Add it to the 'low' heap (max-heap), then pop the max and add it to the 'high' heap. This ensures that the 'high' gets the larger elements.
b. Then, if the 'high' heap has more elements than 'low', move the smallest from 'high' to 'low'. This balances the sizes.
- All numbers in 'low' are <= all numbers in 'high'. So whenever a new number comes in, we can add it to the appropriate heap and then rebalance.
- Add the number to the max-heap (low). Then, take the max from low and add it to the min-heap (high). Now, if high has more elements than low, take the min from high and add back to low. This way, low can be equal or one larger than high.
- Steps are :
- Add to max-heap (low)
- Pop the max from low, add to min-heap (high)
- If high's size is larger than low's, pop the min from high and add to low.
- When the total count is even, both heaps have the same size, and the median is the average of their tops. If odd, the low heap has one more element, so its top is the median.
```
class MedianFinder {
    private PriorityQueue<Integer> low;  // Max-heap for the lower half
    private PriorityQueue<Integer> high; // Min-heap for the upper half

    public MedianFinder() {
        low = new PriorityQueue<>(Collections.reverseOrder());
        high = new PriorityQueue<>();
    }
    
    public void addNum(int num) {
        low.offer(num);
        high.offer(low.poll());
        if (high.size() > low.size()) {
            low.offer(high.poll());
        }
    }
    
    public double findMedian() {
        if (low.size() > high.size()) { //odd
            return low.peek();
        } else {
            return (low.peek() + high.peek()) / 2.0;
        }
    }
}
```

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

## Tree
1. [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/) - O(n) & O(n)
- Solution: 1. Traversal like pre-order
- So maybe I can do a pre-order traversal, and when I encounter a null, I add something like "N" to the string. Each node's value is separated by a delimiter, like a comma. That way, when deserializing, I can split the string into a list of values and reconstruct the tree.
- So for serialize function, start with the root. If the node is null, append "N" and return. Otherwise, append the node's value, then recursively serialize the left and right children. That should build the string in pre-order.
- Now for deserializing, I need to take the string, split it into an array or queue of values. Then, using the same pre-order approach, take the first element. If it's "N", return null. Otherwise, create a node with that value, then recursively deserialize the left and right children, in that order.
- For serialize method takes a TreeNode root. If root is null, maybe return "N". Otherwise, pre-order traversal. Using a helper function that builds a string, maybe via a StringBuilder for efficiency.
- For deserialize, split the string by commas into a list. Then use a helper function that takes a list (as a queue) and builds the tree. Each time, pop the first element, if it's "N", return null. Else create node, then recursively build left and right.
```
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        serializeHelper(root, sb);
        if (sb.length() > 0) {
            sb.setLength(sb.length() - 1); // Remove trailing comma
        }
        return sb.toString();
    }

    private void serializeHelper(TreeNode node, StringBuilder sb) {
        if (node == null) {
            sb.append("N,");
        } else {
            sb.append(node.val).append(",");
            serializeHelper(node.left, sb);
            serializeHelper(node.right, sb);
        }
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.isEmpty()) {
            return null;
        }
        Queue<String> nodes = new LinkedList<>(Arrays.asList(data.split(",")));
        return deserializeHelper(nodes);
    }

    private TreeNode deserializeHelper(Queue<String> nodes) {
        String val = nodes.poll();
        if (val.equals("N")) {
            return null;
        }
        TreeNode node = new TreeNode(Integer.parseInt(val));
        node.left = deserializeHelper(nodes);
        node.right = deserializeHelper(nodes);
        return node;
    }
}
```
2. [Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/description/) - O(nm) & O(n+m)
- Question: Determine if a given binary tree (subRoot) is a subtree of another binary tree (root). A subtree is defined as a tree that starts at some node in the main tree and has exactly the same structure and node values as the given subtree.
- Solution: 1. DFS Pre order traversal
- First, I remember that a subtree means that the structure and node values must match exactly starting from some node in the main tree. So, the main idea is to check every node in the root tree to see if any of them is the start of a subtree that exactly matches subRoot.
- Wait, right. So the steps probably are: For each node in the root tree, check if the subtree starting at that node is identical to subRoot. If any of them is, return true. Otherwise, false.
- But how do I traverse the root tree and check each node's subtree? Oh right, a helper function to check if two trees are identical. And then in the main function, traverse the root tree, and for each node, call the helper to check if the subtree matches.
- So the helper function would be something like isSameTree(TreeNode a, TreeNode b). If both are null, return true. If one is null and the other isn't, return false. Then check if the values are equal, and recursively check left and right subtrees.
- Then, in the isSubtree function, we need to traverse the root tree. So maybe a recursive approach here as well. Like, check if the current node's subtree matches subRoot. If not, check the left and right children recursively. So the base case would be if root is null, then return false (since subRoot can't be a subtree of an empty tree unless subRoot is also null, but maybe we need to handle that edge case).
```
class Solution {
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null) {
            return subRoot == null; //If both are null, return true. If one is null and the other isn't, return false.
        }
        if (isSameTree(root, subRoot)) {
            return true;
        }
        return isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);
    }
    
    private boolean isSameTree(TreeNode a, TreeNode b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return a.val == b.val && isSameTree(a.left, b.left) && isSameTree(a.right, b.right);
    }
}
```

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
- Create the clone of the input node. Add it to visited with original node as key.
- Create a queue and add the original node to it.
- While the queue is not empty: Dequeue a node (current). Iterate over all neighbors of current.
- For each neighbor: If neighbor is not in visited: Create a clone of neighbor and Add to visited.
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
                    visited.put(neighbor, new Node(neighbor.val)); //Create neighbor node
                    queue.add(neighbor);
                }
                visited.get(current).neighbors.add(visited.get(neighbor));
            }
        }
        
        return clonedNode;
    }
}
```
```
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
15. [Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/description/) - O(nm) & O(nm)
- Question: Determine which cells in a given grid can allow rainwater to flow to both the Pacific and Atlantic oceans. The Pacific Ocean is located to the top and left of the grid, while the Atlantic Ocean is located to the bottom and right. Water can flow from a cell to adjacent cells with equal or lower height.
- Solution: 1. BFS 2. DFS
- Hmm. So the approach here is probably to find all cells that can reach the Pacific and all cells that can reach the Atlantic, then find the intersection of those cells. But how do I efficiently find which cells can reach each ocean?
- Let me think. For the Pacific, the starting points are the top and left edges. Similarly, for the Atlantic, the starting points are the bottom and right edges. But instead of starting from each cell and seeing if it can reach an ocean, which would be O(n^2*m^2) time if done naively, which is not feasible for large grids, maybe we can reverse the flow. Because water flows from higher to lower, so maybe we can start from the ocean edges and see which cells can be reached by flowing upwards (i.e., cells that are higher or equal). That way, we can perform a BFS or DFS from all the edge cells and mark the reachable cells for each ocean. Then, the cells that are reachable from both oceans are the answer.
- For Pacific: start from all cells in the first row and first column. Do a BFS/DFS to find all cells that can reach the Pacific by moving to higher or equal height cells.
- For Atlantic: start from all cells in the last row and last column. Do the same to find cells that can reach the Atlantic.
- The intersection of these two sets is the answer. So the algorithm steps are:
- Create two matrices, pacific and atlantic, initialized to false.
- For each cell in the first row and first column (Pacific edge), perform BFS/DFS to mark all cells that can be reached from there by moving to higher or equal cells.
- For each cell in the last row and last column (Atlantic edge), do the same.
- Iterate through all cells and collect those that are marked in both matrices.
- Now, how to implement BFS/DFS. Let's think of BFS. For each cell in the initial positions, add them to a queue. Then, for each cell in the queue, check all four directions. If the neighboring cell is within bounds, and hasn't been visited yet, and its height is >= current cell's height, then mark it as reachable and add to the queue.
- Initialize a queue with all edge cells (for Pacific, first row and first column; for Atlantic, last row and last column).
- Mark those cells as reachable.
- For each cell in the queue, check all four directions. For each neighbor:
- If neighbor is in bounds.
- If neighbor hasn't been marked yet.
- If neighbor's height >= current cell's height.
- If so, mark the neighbor as reachable and add to the queue.
- This way, all cells that can reach the ocean (via flowing downhill) are marked. Because the BFS is in reverse, starting from the ocean and moving uphill.
```
class Solution {
    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        List<List<Integer>> result = new ArrayList<>();
        if (heights == null || heights.length == 0 || heights[0].length == 0) {
            return result;
        }
        
        int m = heights.length;
        int n = heights[0].length;
        
        boolean[][] pacific = new boolean[m][n];
        boolean[][] atlantic = new boolean[m][n];
        
        Queue<int[]> pQueue = new LinkedList<>();
        Queue<int[]> aQueue = new LinkedList<>();
        
        // Initialize Pacific queue with top and left edges
        for (int j = 0; j < n; j++) {
            pacific[0][j] = true;
            pQueue.offer(new int[]{0, j});
        }
        for (int i = 0; i < m; i++) {
            if (!pacific[i][0]) {
                pacific[i][0] = true;
                pQueue.offer(new int[]{i, 0});
            }
        }
        
        // Initialize Atlantic queue with bottom and right edges
        for (int j = 0; j < n; j++) {
            atlantic[m-1][j] = true;
            aQueue.offer(new int[]{m-1, j});
        }
        for (int i = 0; i < m; i++) {
            if (!atlantic[i][n-1]) {
                atlantic[i][n-1] = true;
                aQueue.offer(new int[]{i, n-1});
            }
        }
        
        // Perform BFS for both oceans
        bfs(heights, pQueue, pacific);
        bfs(heights, aQueue, atlantic);
        
        // Collect cells that can reach both oceans
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (pacific[i][j] && atlantic[i][j]) {
                    result.add(Arrays.asList(i, j));
                }
            }
        }
        
        return result;
    }
    
    private void bfs(int[][] heights, Queue<int[]> queue, boolean[][] visited) {
        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        int m = heights.length;
        int n = heights[0].length;
        
        while (!queue.isEmpty()) {
            int[] cell = queue.poll();
            int i = cell[0];
            int j = cell[1];
            for (int[] dir : dirs) {
                int ni = i + dir[0];
                int nj = j + dir[1];
                // Check if the new cell is within bounds, not visited, and has sufficient height
                if (ni >= 0 && ni < m && nj >= 0 && nj < n && !visited[ni][nj] && heights[ni][nj] >= heights[i][j]) {
                    visited[ni][nj] = true;
                    queue.offer(new int[]{ni, nj});
                }
            }
        }
    }
}
```
