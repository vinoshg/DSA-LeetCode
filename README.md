## Data Structures & Algorithms

1. [Binary Search - O(nlog n) & O(1)](#binary-search)
2. [Cyclic Sort - O(n) & O(1)](#cyclic-sort)
3. [Bitwise](#bitwise) - Missing Number || Reverse Bits || Number of 1 Bits || Counting Bits || Sum of Two Integers 
4. [Prefix Sum - O(n) & O(1)](#prefix-sum)
5. [Kadane's Algorithm - O(n) & O(1)](#kadane's-algorithm)
6. [Top K elements - O(nlogk) and O(k)](#top-k-elements)
7. [Monotonic Stack - O(n) & O(n)](#monotonic-stack) 
8. [Sliding Window - O(n) & O(1)](#sliding-window)
9. [Two Pointers - O(n) and O(1)](#two-pointers)
10. [Overlapping Intervals - O(nlogn) & O(n)](#overlapping-intervals)
11. [Backtracking](#backtracking)
12. [Dynamic Programming](#dynamic-programming) - Longest Common Subsequence(LCS) || Target Sum || Coin Change || Climbing Stairs || Longest Increasing Subsequence(LIS) || Word Break || Maximum Product Subarray || Unique Paths || House Robber || House Robber II || Decode Ways || Edit Distance   
13. [Ad-hoc](#ad-hoc)
14. [Greedy Algorithm](#greedy-algorithm)
15. [Linked List](#linked-list)
16. [Stack](#stack)
17. [Queue](#queue) 
18. [Heap - Priority Queue](#heap)
19. [Trie - O(L) & O(N*L)](#trie)
20. [Tree](#tree) - Serialize and Deserialize Binary Tree | Subtree of Another Tree | Validate Binary Search Tree(BST) | Invert Binary Tree | Same Tree | Binary Tree Level Order Traversal | Kth Smallest Element in a BST | Maximum Depth of Binary Tree | Construct Binary Tree from Preorder and Inorder Traversal | Lowest Common Ancestor of a Binary Search Tree (BST) | Binary Tree Maximum Path Sum 
21. [Graph - O(V+E) & O(V)](#graph) - Number of Islands | Clone Graph | Pacific Atlantic Water Flow | Word Search | Course Schedule | Graph Valid Tree | Alien Dictionary | Number of connected components in an undirected graph | 

## Binary Search
1. Binary Search - O(nlog n) & O(1)
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

## Cyclic Sort 
1. Cyclic Sort - O(n) and O(1)
- Question: Range (0,n) or (1,n) - Find the target element in an unsorted array
- Solution: 1. while loop i<array.length index=array[i]-1, if array[i]!=array[index] then swap else i++
2. Sort an Array - Sorting Algorithm - O(n*n) and O(1)
- Solution: 1. Bubble Sort - Nested loops i=0 to array.length and j=0 to array.length-i, if array[j]>array[j+1] swap and end ith loop flag break; 2. Insertion Sort - Nested loops i=0 to array.length and j=i+1;j>0;j--, if array[j]<array[j-1] swap else break; 3. Selection Sort - For loop i=0 to array.length last=array.length-i-1 getMaxIndex= range(0, last) swap(last,maxIndex)
3. [Missing Number / Find all numbers disappeared in an array](https://leetcode.com/problems/missing-number/description/) - O(n) & O(1)
- Solution: 1. Apply Cyclic Sort and then For loop if array[i]!=i+1 got and(i+1)
4. [Find the Duplicate Number / Find all Duplicates in an array](https://leetcode.com/problems/find-the-duplicate-number/description/) - O(n) & O(1)
- Question: Find the duplicate number in an array of integers where each integer is in the range [1, n] and there is exactly one duplicate number. 
- Solution: 1. Apply Cyclic Sort and then For loop if array[i]!=i+1 got ans(array[i]) 2. Binary Search -O(nlog n) & O(1)
```
class Solution {
    public int findDuplicate(int[] arr) {
        int i = 0;
        
        while (i < arr.length) {
            if(arr[i] != i + 1) {
                int correct = arr[i] - 1;
                if(arr[i] != arr[correct]) {
                    swap(arr, i, correct);
                } else {
                    return arr[i];
                }
            } else {
                i++;
            }
        }
        return -1;
    }

    public void swap(int[] arr, int first, int second) {
        int temp = arr[first];
        arr[first] = arr[second];
        arr[second] = temp;
    }
}
```
- Solution: 2. Binary Search -O(nlog n) & O(1)
- The approach we use here is the binary search method. This method leverages the properties of numbers in a range and the pigeonhole principle. The key idea is to count how many numbers in the array are less than or equal to a midpoint value. Based on this count, we adjust our search range to narrow down the duplicate number.
- Initialization: Start with low as 1 and high as n (where n is the length of the array minus 1).
- Binary Search: While low is less than high, calculate the midpoint mid of the current range.
- Count Elements: Count how many numbers in the array are less than or equal to mid.
- Adjust Search Range: If the count exceeds mid, it means the duplicate number is in the lower half; otherwise, it is in the upper half.
- Converge: Continue narrowing the search range until low equals high, which will be the duplicate number.
```
class Solution {
    public int findDuplicate(int[] nums) {
        int low = 1;
        int high = nums.length - 1;
        while (low < high) {
            int mid = (low + high) / 2;
            int count = 0;
            for (int num : nums) {
                if (num <= mid) {
                    count++;
                }
            }
            if (count > mid) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        return low;
    }
}
```
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
4. [Counting Bits](https://leetcode.com/problems/counting-bits/description/) -
- Question: Count the number of 1 bits in the binary representation of every number from 0 to a given integer n.
- Solution: 1. Dynamic programming (For each number i, the number of 1 bits can be derived from previously computed results.)
- First, the straightforward way is to iterate through each number from 0 to n, and for each number, count the number of 1s in their binary representation. But wait, that might be inefficient, especially if n is large. Because for each number, counting bits could take O(number of bits) time. Like for numbers up to n, that's O(n log n) time. Is there a better way?
- I remember that there's a dynamic programming approach for this problem. Let me think. The idea is to use previously computed results to build the current one. For example, if I know the count for a number divided by two, maybe I can use that. Because when you divide a number by two (right shift), you're removing the last bit. So the count of 1s for the current number is the count for the shifted number plus the least significant bit. Like, for even numbers, the last bit is 0, so the count is same as i/2. For odd numbers, it's count of i/2 plus 1. Wait, but how do I check if the current number is even or odd? Well, the parity can be determined by checking if i is even or odd. So for any i, bits[i] = bits[i >> 1] + (i & 1). Because i & 1 gives 0 if even, 1 if odd. That could work.
- So the steps would be:
- Initialize an array of size n+1.
- Set bits[0] = 0.
- For each i from 1 to n, compute bits[i] as bits[i >> 1] + (i & 1).
- Return the array.
- Wait, but how about another approach? Like, using the fact that the number of set bits in i is equal to the number of set bits in i with the last set bit removed, plus 1. For example, for i=5 (101), the last set bit is at position 0. So i & (i-1) gives 4 (100), and bits[5] = bits[4] +1. So another recurrence is bits[i] = bits[i & (i-1)] +1. Because i & (i-1) removes the last set bit.
```
class Solution {
    public int[] countBits(int n) {
        int[] result = new int[n + 1];
        result[0] = 0;
        for (int i = 1; i <= n; i++) {
            result[i] = result[i >> 1] + (i & 1); // result[i/2] + 0 or 1 (even or odd i)
        }
        return result;
    }
}
```
5. [Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/description/) - O(1) & O(1) since the loop runs a constant number of times (at most 32 iterations for 32-bit integers
- Question: Find the sum of two integers without using the + or - operators.
- Solution: 1. Bitwise operations
- Wait, I remember that in some programming challenges, people use bitwise operations for this. Like, using XOR and AND maybe. Let me recall. Oh right, because when you add two binary numbers, the XOR gives the sum without considering the carry, and the AND gives the carry bits. Then you shift the carry left by one and add it again. But since we can't use +, we have to do this iteratively until there's no carry left.
- So the steps would be something like:
- Calculate the sum without carry (a XOR b)
- Calculate the carry (a AND b) shifted left by 1.
- Now, add these two results together, but since we can't use +, we repeat the process using these two new numbers until the carry is zero.
- But wait, how do you handle this in a loop? Like, we need to keep doing this until the carry is zero. So maybe in each iteration, we compute the new sum and new carry, and set a and b to be the sum and carry. Then when carry becomes zero, the sum is the answer.
```
class Solution {
    public int getSum(int a, int b) {
        while (b != 0) {
            int sum = a ^ b;
            int carry = (a & b) << 1;
            a = sum;
            b = carry;
        }
        return a;
    }
}
```

## Prefix Sum 
1. Prefix Sum Array - O(n)
- Solution: 1. For loop i=1 to array.length array[i] = array[i-1] + array[i] 
2. Range Sum Query
- Solution: 1. Build Prefix Sum Array and return prefixSum[right] - prefixSum[left-1]. Handle if left==0 then return prefixSum[right]
3. Number of Subarrays whose sum is k
- Solution: 1. PrefixSum with HashMap for if map.containsKey(prefixSum-k) after map.put(prefixSum, map.getOrDefault(prefixSum,0) + 1)          2. Nested loops

## Kadane's Algorithm 
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
## Top K elements
- Question: K largest/smallest, K most/least frequent, Top k elements  - O(nlogk) and O(k)
1. Top K largest elements
- Solution: 1. Min heap(Priority Queue) add at max k elements and remaining poll 2. Sort Array and take elements - O(nlogn)
2. [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/) - O(nlogk) & O(n+k)
- Question: Find the top k frequent elements in an integer array.
- Solution: 1. Min heap having HashMap 2. Sort
- So first, I need to think about how to count the frequency of each element. Oh right, I can use a hash map for that. Like, iterate through the array, and for each number, increment its count in the map. That part seems straightforward.
- Once I have the frequencies, I need to find the top k elements with the highest counts. The challenge here is how to efficiently get the k most frequent elements. Sorting all the elements by their frequency would take O(n log n) time, but maybe there's a better way, especially since the problem could have constraints where k is much smaller than the number of unique elements.
- Wait, what about using a priority queue (heap)? Like, a min-heap that keeps track of the k most frequent elements. That way, when we process all elements, the heap will contain the top k. The advantage is that inserting into a heap of size k takes O(log k) time, which is better than sorting everything if k is small.
- So the steps would be:
- Count the frequency of each number using a hash map.
- Create a min-heap based on the frequency. The idea is to keep the heap size at k. For each entry in the hash map, we add it to the heap. If the heap size exceeds k, we remove the smallest element (the one with the least frequency in the heap). That way, the heap always contains the k largest elements.
- Once all elements are processed, extract the elements from the heap. Since it's a min-heap, the elements are not in order, but the order doesn't matter as the problem states any order is acceptable.
- But wait, how do I implement the heap in Java? Oh right, the PriorityQueue. But the PriorityQueue in Java is a min-heap by default. So when adding elements, the smallest (according to their natural ordering) is at the front. So to implement a min-heap based on frequency, I need to compare the frequencies of the elements.
- So for the priority queue, the comparator should order elements by their frequency in ascending order. That way, the element with the smallest frequency in the heap is at the top, and when we add a new element that has a higher frequency than the smallest, we remove the top and add the new one, keeping the heap size at k.
- But how to store the elements in the heap? Since we need to track both the number and its frequency, maybe we can have a helper class or use a Map.Entry. But in Java, it's easier to create a list of entries from the frequency map and then process them. Alternatively, the priority queue can hold the entries and compare based on their values (frequency).
- So the code steps:
- Create the frequency map:
Map<Integer, Integer> frequencyMap = new HashMap<>();
- for each num in nums:
frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
- Then, create a priority queue that compares entries by their value (frequency). The priority queue will have a size of k. For each entry in the frequency map, add to the queue. If the size exceeds k, remove the head (smallest frequency entry).
- But wait, the priority queue should be a min-heap of size k. So when adding entries, if the queue's size is less than k, we add. Otherwise, if the current entry's frequency is larger than the smallest in the heap (which is the head), then we remove the head and add the current entry.
- So the comparator for the priority queue should order entries by their frequency in ascending order. So the priority queue will be initialized with a comparator that compares the Entry's values.
- But in Java, when using a priority queue of Map.Entry, how to do that? Because the entries are from the hash map. So perhaps we can create a priority queue that uses a comparator comparing the entries by their value.
- So the priority queue can be initialized like this:
PriorityQueue<Map.Entry<Integer, Integer>> heap =
new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());
- This comparator will order entries with lower frequencies first. So the head of the queue is the entry with the smallest frequency in the heap.
- Then, for each entry in the frequency map's entry set, we add it to the heap. If the heap's size exceeds k, we poll the head (the smallest frequency entry). So after processing all entries, the heap will contain the k entries with the highest frequencies.
- Once we have the heap, we can extract the elements. The order doesn't matter, so we just need to collect the keys from the entries in the heap.
- So the result array will be of size k. Iterate through the heap, extract each entry's key and add to the result array.
- Putting it all together:
- Create frequency map.
- Initialize the priority queue with a comparator that sorts entries by frequency in ascending order.
- Iterate over each entry in the frequency map:
a. Add entry to the heap.
b. If the heap size exceeds k, remove the head.
- Once all entries are processed, the heap contains the top k entries. Extract the keys into an array.
```
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        // Create a frequency map to count occurrences of each number
        Map<Integer, Integer> frequencyMap = new HashMap<>();
        for (int num : nums) {
            frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
        }
        
        // Create a min-heap based on the frequency of the elements
        PriorityQueue<Map.Entry<Integer, Integer>> heap = new PriorityQueue<>(
            (a, b) -> a.getValue() - b.getValue()
        );
        
        // Add entries to the heap, maintaining the size at k
        for (Map.Entry<Integer, Integer> entry : frequencyMap.entrySet()) {
            heap.offer(entry);
            if (heap.size() > k) {
                heap.poll();
            }
        }
        
        // Extract the top k frequent elements from the heap
        int[] result = new int[k];
        int index = 0;
        while (!heap.isEmpty()) {
            result[index++] = heap.poll().getKey();
        }
        
        return result;
    }
}
```
3. K closest points to the origin
- Solution: 1. Max heap - PriorityQueue<int[]> maxHeap = new PriorityQueue<>((a,b) -> getDistance(b) - getDistance(a)) and getDistance is point[0]^2+point[1]^2

## Monotonic Stack 
- Question: 1. Next greater/smaller, previous greater/smaller - O(n)
1. Next greater element for each number in an array
- Solution: 1. Monotonic decreasing stack i.e. greater element index at top -For loop and  while !stack.isEmpty() && array[i]>array[stack.peek()] then index= stack.pop(); result[index]=array[i] afer while stack.push(i); 2. Nested loops - O(n^2)
- For Smaller element : change while !stack.isEmpty() && array[i]<array[stack.peek()]
2. Given a list of daily temperatures, determine how many days you've to wait for the next warmer day
- Solution: 1. Like Next greater element - result[index]=i - index

## Sliding Window 
- Fixed Sliding window (Find Subarray/Substring of a fixed length) 2. Dynamic Sliding window (Longest/Shortest Subarray or Substring that satisfies the condition) - (e.g., max sum, longest substring with unique characters) - I remember that sliding window techniques are useful for substring problems. - O(n) & O(1)
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

## Two Pointers 
- Applicable to Linear Data Structure Arrays, String, LinkedList - Converging pointers (Two pointers start at 0th and array.length-1 and converge together) | Parallel pointers (Right pointer is used to get new info and the left is used to track. Both start from 0/1) | Trigger based pointers (left pointer move only after right pointer reaches particular position) | Expand Around Center - O(n) and O(1) 
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
9. [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/description/) - O(n) & O(1)
- Question: Calculate the amount of rainwater that can be trapped between the bars of an elevation map represented by an array of integers.
- Solution: 1. Two Pointers 2. Dyanamic programmic left and right array 3. Brute force approach would be for each element, find the maximum height to the left and right. Then calculate the water for that position. But that's O(n^2) time
- The input is an array of integers representing the heights, and the output is the total trapped water. Hmm, let's think about how to approach this.
- So, the basic idea is that water can be trapped between two bars (or elevations) if there are higher bars on both sides. The amount of water each position can hold depends on the minimum of the maximum heights on the left and right, minus the height of the current position. Right, that's the key formula: water += min(left_max, right_max) - height[i].
- But how do I compute this efficiently? The brute force approach would be for each element, find the maximum height to the left and right. Then calculate the water for that position. But that's O(n^2) time, which might not be efficient for large arrays. The problem probably expects a better solution.
- I remember there are a few approaches to this problem. Let me think: there's the dynamic programming approach where you precompute the left_max and right_max arrays. Then iterate through the array once, using these precomputed values. That's O(n) time and O(n) space. But maybe there's a way to do it with O(1) space?
- Wait, there's also the two-pointer method. Let me recall. The idea is to have two pointers, left and right, starting at the beginning and end of the array. Also, keep track of the left_max and right_max as you move the pointers. Depending on which side has a lower maximum, you process that side. Because the amount of water trapped is determined by the lower of the two maximums. Let's see:
- If left_max is less than right_max, then the water trapped at the left pointer's current position is determined by left_max (since the right_max is higher, and the minimum is left_max). So you can calculate the water there, and move the left pointer forward. Otherwise, you do the same for the right pointer. This way, you can do it in O(n) time and O(1) space.
- Yes, that sounds promising. Let me outline the steps:
- Initialize left pointer to 0, right pointer to n-1.
- Initialize left_max and right_max to 0.
- Initialize result to 0.
- While left <= right:
- If height[left] < height[right]:
- If height[left] > left_max:
- left_max = height[left]
- Else:
- result += left_max - height[left]
- left += 1
- Else:
- If height[right] > right_max:
- right_max = height[right]
- Else:
- result += right_max - height[right]
- right -=1
- Wait, why does this work? Because when height[left] is less than height[right], it means the left_max is the limiting factor. Because right_max could be even higher. So for the current left position, the maximum left side is left_max, and the right side is at least as high as the current right pointer's height. So the trapped water is left_max - current height. Similarly for the other case.
- But how do left_max and right_max get updated? Let's see. For each step, we compare the current height at left and right. If the left is smaller, then we check if the current left height is higher than left_max. If so, update left_max. Otherwise, add the difference (left_max - height[left]) to the result. Then move the left pointer. The same logic applies when the right is smaller.
- Wait, but in this approach, we are moving the pointers inward, and keeping track of the maximums encountered so far. So left_max is the maximum height from the left up to the current left pointer, and similarly for right_max. But since we process the side which has the smaller current height, the logic is that the other side's maximum is at least as big as the current one. So the trapped water at the current position is determined by the current left_max or right_max, whichever is smaller.
- Let me test this logic with an example. Take the sample input [0,1,0,2,1,0,1,3,2,1,2,1].
- According to the solution, let's walk through:
- Initially, left=0 (height 0), right=11 (height 1). left_max=0, right_max=0.
- Compare height[left] (0) < height[right] (1): yes. So check if height[left] (0) > left_max (0)? No. So add left_max - height[left] (0-0=0) to result (remains 0). Then left moves to 1.
- Now left=1 (height 1), right=11 (1). left_max is still 0. Wait, no, because after moving left to 1, the code first checks if height[left] (1) > left_max (0). If so, update left_max. Because in the code, when height[left] < height[right], then:
- If height[left] > left_max → update left_max. Else add the difference.
- So in the first step, height[left] is 0, which is not greater than left_max (0), so add 0. Then left increments to 1.
- Now, height[left] is 1. Since height[left] (1) > left_max (0), so left_max becomes 1. Then left increments to 2.
- Now left is at 2 (height 0). Compare with right (height 1). So again, height[left] (0) < height[right] (1). So check if height[left] (0) > left_max (1). No. So add left_max (1) - 0 → 1 to result. Result is now 1. left increments to 3.
- At left=3 (height 2). Now compare with right (height 1). 2 > 1, so we process the right side. Check if height[right] (1) > right_max (0). Yes. So right_max becomes 1. Then right decrements to 10.
- Now compare left=3 (height 2) with right=10 (height 2). Since 2 equals 2, but the condition is if left height < right height. So else clause. Now check if height[right] (2) > right_max (1). Yes, update right_max to 2. Decrement right to 9.
- Left=3 (height 2), right=9 (height 1). Since 2 >1, process right side. height[right] is 1, which is less than right_max (2). So add 2-1=1 to result. Result becomes 1+1=2. Then right decrements to 8.
- Left=3 (height 2), right=8 (height 2). height[left] (2) is equal to height[right] (2). So else clause. Check if height[right] (2) > right_max (2). No. So add 2-2=0. Result remains 2. Right decrements to 7.
- Now left=3, right=7. Height[right] is 3. Compare left height (2) < 3. So process left. Check if height[left] (2) > left_max (1). Yes. Update left_max to 2. left increments to 4.
- Left is 4 (height 1). Compare with right=7 (height 3). 1 <3. Process left. Check height[left] (1) > left_max (2). No. Add 2-1=1 to result. Result now 3. left increments to 5.
- Left=5 (height 0). Compare to 3. 0<3. Process left. height is 0. Add 2-0=2. Result becomes 5. left increments to 6.
- Left=6 (height 1). 1<3. Process left. Add 2-1=1. Result 6. left increments to 7. Now left is 7 (height 3). Compare to right=7 (3). So height[left] equals height[right]. Process the else clause. Check if height[right] (3) > right_max (2). Yes, update right_max to 3. right decrements to 6. Now left is 7, right is 6. Loop ends.
- So total trapped water is 6. But the correct answer for this example is 6, right? Wait, the sample input given in the problem description is [0,1,0,2,1,0,1,3,2,1,2,1], and the output is 6. So this approach works here.
```
class Solution {
    public int trap(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }
        
        int left = 0;
        int right = height.length - 1;
        int leftMax = 0;
        int rightMax = 0;
        int result = 0;
        
        while (left <= right) {
            if (height[left] < height[right]) {
                if (height[left] > leftMax) {
                    leftMax = height[left];
                } else {
                    result += leftMax - height[left];
                }
                left++;
            } else {
                if (height[right] > rightMax) {
                    rightMax = height[right];
                } else {
                    result += rightMax - height[right];
                }
                right--;
            }
        }
        
        return result;
    }
}
```

## Overlapping Intervals 
- Question: Overlapping intervals, scheduling conflicts - Approach: Sort intervals and merge overlapping ones - O(nlogn) & O(n)
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
4. [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/description/) - O(nlog n) & O(1)
- Question: Determine if a person can attend all meetings, we need to check if any of the meeting intervals overlap.
- Solution: 1. Sorting the intervals and checking adjacent pairs for overlaps.
- Meeting Rooms, which asks whether a person can attend all meetings. The input is an array of meeting time intervals, and we need to determine if there's any overlap. If all meetings are non-overlapping, return true; otherwise, false.
- So, the approach here is to check if any two intervals overlap. The standard way to do this is to sort the intervals based on their start times. Then, compare each interval with the next one. If the end time of the current interval is greater than the start time of the next, there's an overlap, so return false. Otherwise, after checking all, return true.
- Wait, but how do I handle the sorting? In Java, I can sort the array using a custom comparator. The intervals are given as an array of int arrays, like [[start1, end1], [start2, end2], ...]. So, I need to sort this array based on the start times.
- Let me outline the steps:
- Check if the input array is empty or has one interval. If so, return true.
- Sort the intervals based on the start time in ascending order.
- Iterate through the sorted intervals starting from the first one up to the second last. For each interval i, check if its end time is greater than the start time of interval i+1. If yes, return false.
- If all checks pass, return true.
- Wait, but what if the input is empty? According to the problem statement, maybe the person can attend zero meetings, which is considered possible, so return true. Similarly for a single meeting.
- Now, about the sorting. How to implement the comparator in Java? For an array of int arrays, the Arrays.sort method can take a comparator. Since Java 8, we can use a lambda expression. For example:
- Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
- This will sort the intervals based on their start times.
- Then, loop through the sorted intervals from index 0 to intervals.length - 2. For each index i, check if intervals[i][1] > intervals[i+1][0]. If so, return false. Because the next meeting starts before the current one ends.
- Edge cases to consider:
- Empty array: return true.
- Single interval: return true.
```
public class Solution {
    public boolean canAttendMeetings(int[][] intervals) {
        if (intervals.length <= 1) {
            return true;
        }
        
        // Sort the intervals based on start time
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        
        // Check each consecutive pair of intervals
        for (int i = 0; i < intervals.length - 1; i++) {
            if (intervals[i][1] > intervals[i + 1][0]) {
                return false;
            }
        }
        
        return true;
    }
}
```
5. [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/description/) - O(nlog n) & O(n)
- Question: Determining the minimum number of conference rooms required for given meeting intervals
- Solution: 1. Use a priority queue (min-heap) to efficiently track the end times of meetings and determine room availability. 2. Two-pointer approach - O(n) & O(n)
- Hmm, I remember that this problem can be tackled using a priority queue (or a min-heap) to keep track of the end times of the meetings. The idea is to sort the meetings by their start times and then check if a new meeting can be accommodated in an existing room or if a new room is needed.
- Wait, let me break it down. First, if there are no meetings, return 0. Then, sort the intervals based on their start times. Then, use a priority queue to keep track of the earliest ending time of the currently occupied rooms. For each meeting, if its start time is after the earliest end time in the queue, we can reuse that room (by polling the end time and adding the new end time). Otherwise, we need to add a new room. The size of the priority queue at the end will be the minimum number of rooms required.
- Alternatively, another approach is the chronological ordering method. Split the start and end times into two separate arrays, sort them, and then use two pointers to iterate through them. Increment the count when a start time is encountered and decrement when an end time is encountered. The maximum count during this process is the answer. But I think the heap method is more straightforward.
- Wait, let me think through an example. Suppose we have intervals like [[0,30],[5,10],[15,20]]. After sorting by start time, we have the same order. The heap starts empty. Add 30's end time. Next meeting starts at 5, which is before 30, so add a new room (heap now has 10 and 30). Then next meeting starts at 15, which is after 10, so we can replace 10 with 20. So the heap size is 2. So answer is 2. That makes sense.
- So the steps are:
- Sort the intervals by start time.
- Initialize a min-heap to keep track of end times.
- Iterate through each interval:
a. If the current interval's start time is >= the earliest end time in the heap, remove the earliest end time.
b. Add the current interval's end time to the heap.
- The size of the heap is the minimum rooms needed.
- But wait, for step 3a: even if the start time is >= earliest end time, we can remove that end time and add the current end time. This effectively reuses the room. If not, we have to add a new room. The heap will always contain all the end times of the currently occupied rooms.
- Yes, that makes sense. So the heap's size at the end is the answer.
- Let me test this logic with the example. The first interval [0,30] is added to the heap. Heap has 30. Next interval [5,10]. Since 5 < 30, we can't reuse, so add to heap. Heap size 2. Next interval [15,20]. The earliest end time is 10. 15 >=10, so we remove 10 and add 20. Heap size remains 2. So answer is 2. Correct.
- Another example: [[7,10],[2,4]]. Sorted by start time: [[2,4], [7,10]]. First interval added to heap (4). Second starts at 7 >=4, so remove 4, add 10. Heap size is 1. So answer 1. Correct.
- Now, how to implement this in Java. The priority queue in Java is a min-heap by default. So for the earliest end time, we can just use a priority queue.
- So code steps:
- Check if intervals is empty, return 0.
- Sort the intervals array based on start time.
- Initialize a priority queue (min-heap) for end times.
- Iterate over each interval in the sorted list:
- if the queue is not empty and current interval's start >= queue.peek(), then poll the queue.
- add the current interval's end to the queue.
- The size of the queue is the minimum rooms.
- Yes. That should work.
- Now, code:
- But first, how to sort the intervals. Since in Java, we can use Arrays.sort with a comparator. For intervals, which is an int[][], we can sort using a comparator that compares the start times.
- So: Arrays.sort(intervals, (a, b) -> a[0] - b[0]); Then, the priority queue is a PriorityQueue<Integer>.
-  The time complexity is O(n log n) for sorting and each heap operation is O(log n), so total O(n log n). Space O(n) for the heap.
- Alternatively, the two-pointer approach:
- Sort start times and end times. Then, use two pointers. When a start time is less than the current end time, we need a room. Otherwise, we can reuse a room. The maximum number of overlapping meetings at any point is the answer.
- But how to track that. For example: Sort starts: [0,5,15] Sort ends: [10,20,30]
- Initialize startPointer=0, endPointer=0. rooms=0.
- Loop over each start:
- start=0, end=10. 0 <10: need room. rooms=1. startPointer++.
- start=5 <10: need another room. rooms=2. startPointer++.
- start=15 >=10: can reuse a room. endPointer++. rooms remains 2.
- Then startPointer moves beyond array. So answer is 2.
- But how to track the maximum. Wait, here's the approach:
- Sort all the start times and end times.
- Initialize two pointers, startPtr and endPtr, both at 0.
- Initialize rooms=0 and maxRooms=0.
- While startPtr < n:
- If the current start time < current end time:
- rooms++ startPtr++ update maxRooms if needed
- Else: rooms-- endPtr++
- This way, for each start time, if it's before the current end time, you need a room. Once a meeting ends (i.e., when the start time is >= end time), you can reuse that room.
```
public class Solution {
    public int minMeetingRooms(int[][] intervals) {
        if (intervals == null || intervals.length == 0) {
            return 0;
        }
        
        // Sort the intervals based on their start times
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        
        // Min-heap to keep track of the end times of the meetings in rooms
        PriorityQueue<Integer> heap = new PriorityQueue<>();
        
        for (int[] interval : intervals) {
            // If the current meeting starts after the earliest ending meeting, reuse that room
            if (!heap.isEmpty() && interval[0] >= heap.peek()) {
                heap.poll();
            }
            // Add the current meeting's end time to the heap
            heap.offer(interval[1]);
        }
        
        // The size of the heap is the minimum number of rooms required
        return heap.size();
    }
}
```
```
//Two-pointer approach
public int minMeetingRooms(int[][] intervals) {

    if (intervals.length == 0) return 0;

    int[] starts = new int[intervals.length];

    int[] ends = new int[intervals.length];

    for (int i = 0; i < intervals.length; i++) {

        starts[i] = intervals[i][0];

        ends[i] = intervals[i][1];

    }

    Arrays.sort(starts);

    Arrays.sort(ends);

    int startPtr = 0, endPtr = 0;

    int rooms = 0, maxRooms = 0;

    while (startPtr < intervals.length) {

        if (starts[startPtr] < ends[endPtr]) {

            rooms++;

            maxRooms = Math.max(maxRooms, rooms);

            startPtr++;

        } else {

            rooms--;

            endPtr++;

        }

    }

    return maxRooms;

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

## Dynamic Programming
- Iteration(Two variable & dp[] array) and Memoization(Recursion)
- Maximise/Minimise/Fewest of certain value or number of ways
- Question: Optimize recursive problems with overlapping subproblems - 0/1 Knapsack | Unbounded Knapsack | Longest Common Subsequence (LCS) | Fibonacci sequence pattern
- 0/1 Knapsack(Subset selection with constraints) | Unbounded Knapsack(Unlimited item usage) | LCS(String alignment problems)
1. [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/description/) - O(mn) & O(mn)
- Question: Finding the length of the longest common subsequence (LCS) between two given strings. A subsequence is a sequence that can be derived from another sequence by deleting some elements without changing the order of the remaining elements.
- Solution: 1. DP
- Hmm, right. So the classic approach here is dynamic programming. I remember that. So, the idea is to create a 2D array where dp[i][j] represents the length of the longest common subsequence between the first i characters of text1 and the first j characters of text2.
- Let me think about the base cases. If either i or j is 0, then the subsequence length is 0 because one of the strings is empty. So dp[0][j] and dp[i][0] are all zeros.
- Now, for the recursive relation. If the current characters (text1[i-1] and text2[j-1]) are the same, then dp[i][j] = dp[i-1][j-1] + 1. Because we can take the LCS up to i-1 and j-1 and add this matching character.
- If they are not the same, then we take the maximum of dp[i-1][j] (excluding the current character of text1) and dp[i][j-1] (excluding the current character of text2). That makes sense because the LCS could be in either of those possibilities.
- So the steps are:
- Initialize a DP table with (m+1) x (n+1) dimensions, where m and n are the lengths of text1 and text2.
- Iterate through each character in text1 and text2.
- For each pair (i,j), check if the characters match.
- Update the DP table accordingly.
- Wait, but the problem is about optimization. What about space? The standard DP uses O(mn) space. But maybe we can optimize it to O(n) or O(m) since we only need the previous row. But for now, maybe just implement the standard solution, since it's straightforward.
```
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        return dp[m][n];
    }
}
```
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
10. [House Robber II](https://leetcode.com/problems/house-robber-ii/description/) - O(n) & O(1)
- Question: Determine the maximum amount of money that can be robbed from houses arranged in a circle without triggering the security system. The key challenge is that the first and last houses are adjacent, meaning they cannot both be robbed on the same night.
- Solution: 1. DP
- Wait, how did the original problem work? Oh right, for each house, you decide to rob it or not. If you rob it, you add its value to the previous non-adjacent one. So the dynamic programming approach was to keep track of two variables: the max if you rob the previous house and the max if you don't. Or maybe it's better to track the maximum up to each house considering whether you took the previous or not.
- But in this circular case, the first and last are connected. So the problem reduces to two scenarios: either you rob the first house (and can't rob the last) or you don't rob the first (and can rob the last). Then, the solution is the maximum of these two scenarios.
- So the plan is: split the problem into two separate cases. One where we exclude the first house and calculate the max for the rest, and another where we exclude the last house. Then take the maximum of these two results. But wait, how do I handle those cases?
- For example, if the array is nums = [2,3,2], then the max is 3. Because if you take the first and last, they are adjacent, so you can't. So you take the middle one.
- So the approach would be to compute the regular house robber problem on two subarrays: one from 0 to n-2 (excluding the last) and another from 1 to n-1 (excluding the first). Then take the maximum of those two results.
- But wait, what if the array has only one element? Then the answer is that element. Also, if there are two elements, the answer is the max of the two.
- So first, handle edge cases. If nums is empty, return 0. If length 1, return nums[0]. If length 2, return max(nums[0], nums[1]).
- Otherwise, for the general case:
- max_amount = max(rob_linear(nums, 0, n-2), rob_linear(nums, 1, n-1))
- Where rob_linear is the helper function that computes the maximum for a straight line of houses from start to end indices.
- So how to implement rob_linear? Let's think. The helper function would take the nums array, a start index, and an end index. Then compute the maximum you can rob from houses start to end, with adjacent houses being a problem.
- The original DP approach for the linear case uses variables like prevMax and currMax. Let's see:
- In the linear case, you can either take the current house plus the max from two steps back, or not take it and keep the max from the previous step.
- So for the helper function, we can loop from start to end, and compute the max.
```
class Solution {
    public int rob(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        return Math.max(robLinear(nums, 0, nums.length - 2), robLinear(nums, 1, nums.length - 1));
    }
    
    private int robLinear(int[] nums, int start, int end) {
        int prev = 0, curr = 0;                    //prev and curr, to keep track of the maximum money robbed up to the previous and current house
        for (int i = start; i <= end; i++) {
            int temp = curr;
            curr = Math.max(curr, prev + nums[i]); // max upto previous house not considering current house, max of two step back+current house
            prev = temp;
        }
        return curr;
    }
}
```
11. [Decode Ways](https://leetcode.com/problems/decode-ways/description/) - O(n) & O(n)
- Question: Determine the number of ways to decode a given string of digits into valid letters, where each letter corresponds to a number from 1 to 26.
- Solution: 1. DP
- First, I remember that this is a dynamic programming problem. Because each position's result depends on previous choices. Like, if the current digit is valid by itself, or combined with the previous digit.
- Wait, what are the possible cases? For each step, the current digit can be a single number (if it's not zero), or combined with the previous digit (if between 10 and 26). So the total ways for the i-th position is the sum of the ways from the previous position (if current is valid) plus the ways from two positions back (if the combination is valid).
- But there are edge cases. For example, if the string starts with '0', then there's no way to decode it. So the answer is zero. Also, if two consecutive digits form a number larger than 26, then they can't be combined. Like '27' can only be split into 2 and 7, not 27.
- So, the plan is to use a DP array where dp[i] represents the number of ways to decode the first i characters. Let's see.
- Let's initialize dp[0] as 1. Because an empty string has one way to decode (doing nothing). Then, for each position from 1 to n:
- Check if the current digit is not zero. If yes, add dp[i-1] to dp[i].
- Check if the previous two digits form a number between 10 and 26. If yes, add dp[i-2] to dp[i].
- But wait, how to handle the initial steps. Like, for i=1, we need to check the first character. For i=2, check if the first two form a valid number.
- Also, need to handle cases where the string is of length 1. For example, '0' returns 0, '1' returns 1, etc.
```
class Solution {
    public int numDecodings(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1; //Handle string length 1
        for (int i = 2; i <= n; i++) {
            int first = s.charAt(i - 2) - '0';
            int second = s.charAt(i - 1) - '0';
            int twoDigit = first * 10 + second;
            // Check single digit
            if (second >= 1 && second <= 9) {
                dp[i] += dp[i - 1];
            }
            // Check two digits
            if (twoDigit >= 10 && twoDigit <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[n];
    }
}
```
12. [Edit Distance](https://leetcode.com/problems/edit-distance/description/) - O(mn) & O(mn)
- Question: Compute the minimum number of operations required to convert one string into another using insert, delete, or replace operations.
- Solution: 1. DP - classic example of the edit distance (Levenshtein distance) problem
- I have to find the minimum number of operations required to convert word1 into word2. The operations allowed are insert, delete, or replace a character. Hmm, this reminds me of the edit distance problem. Right, because that's exactly what this is. So I need to compute the Levenshtein distance between the two words.
- Let me think about how to approach this. The standard approach for edit distance is dynamic programming. So, the idea is to build a DP table where dp[i][j] represents the minimum number of operations needed to convert the first i characters of word1 to the first j characters of word2.
- The base cases would be when one of the strings is empty. For example, if word1 is empty, we need to insert all characters of word2, so the distance is the length of word2. Similarly, if word2 is empty, we need to delete all characters of word1, so the distance is the length of word1.
- For the general case, if the characters at positions i-1 and j-1 (since we start from 0) are the same, then we don't need any operation, and dp[i][j] = dp[i-1][j-1]. Otherwise, we have three choices: replace the character (1 + dp[i-1][j-1]), delete from word1 (1 + dp[i-1][j]), or insert into word1 (which is similar to deleting from word2, so 1 + dp[i][j-1]). The minimum of these three options is the value for dp[i][j].
- So the DP table needs to be (m+1) x (n+1), where m and n are the lengths of word1 and word2, respectively.
- Let me try to work through an example. Suppose word1 is "horse", word2 is "ros". The DP table would start with the base cases:
- For i=0 (word1 is empty), each j would be j. For j=0 (word2 is empty), each i would be i.
- Then filling the table:
- For i from 1 to m (length of word1), and j from 1 to n (length of word2):
- If word1[i-1] == word2[j-1], then dp[i][j] = dp[i-1][j-1]
- Else, take the min of the three possible operations and add 1.
- Wait, the replace operation is taking the previous value (i-1, j-1) and adding 1. Delete from word1 would be (i-1, j) +1. Insert into word1 would be (i, j-1) +1. So the min of those three.
- So the steps are:
- Initialize a DP table of (m+1) rows and (n+1) columns.
- Fill the first row and first column with the base cases.
- Then iterate through each cell from (1,1) to (m,n), filling in based on the above logic.
- But wait, since the DP table could be space-optimized. Because each row only depends on the previous row and the current row. So maybe we can use a 1D array. But maybe for the first time, let's code the 2D version for clarity.
```
class Solution {
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        
        // Initialize base cases for converting to/from empty string
        // Initialize first row and column with 0,1,2,3.. m or n
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j;
        }
        
        // Fill the DP table
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + Math.min(dp[i - 1][j - 1], 
                                           Math.min(dp[i - 1][j], dp[i][j - 1]));
                }
            }
        }
        
        return dp[m][n];
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
5. [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/description/) - O(n) & O(n)
- Question: Determine if an array contains any duplicate elements. If any value appears at least twice, we return true; otherwise, we return false.
- Solution: 1. HashSet 2. Sort the array first. Because if there are duplicates, after sorting, they would be next to each other. So then I can just iterate through the sorted array and check if any two consecutive elements are the same - O(n log n) 3. Brute force O(n^2) approach where you check every pair.
- Idea is to use a hash set. For each element in the array, we check if it's already in the set. If it is, return true. Otherwise, add it to the set. If we finish all elements without finding duplicates, return false. The time complexity here is O(n) on average, since hash set operations are O(1). But the space complexity is O(n) because of the set.
- The steps would be:
- Initialize a HashSet.
- Loop through each number in nums:
a. If the number is already in the set, return true.
b. Add the number to the set.
- After processing all elements, return false.
```
class Solution {
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> seen = new HashSet<>();
        for (int num : nums) {
            if (!seen.add(num)) { // if(seen.conatins(num); seen.add(num) return false if num exists in a set
                return true;
            }
            // seen.add(num);
        }
        return false;
    }
}
```
6. [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/) - O(n) & O(1)
- Question: Find an array that each element at index i is the product of all the numbers in the original array except nums[i]. And the catch is that I can't use division and have to do it in O(n) time.
- Solution: 1. Left Pass: Compute the product of all elements to the left of each index and store these products in the output array. Right Pass: Compute the product of all elements to the right of each index while updating the output array by multiplying the left product with the right product.
- So, the straightforward way would be to calculate the product of all elements and then divide by nums[i] for each position. But the problem says no division, right? So that approach is out. Plus, if there's a zero in the array, dividing would cause issues anyway. So I need another method.
- Wait, the problem mentions that the solution must run in O(n) time. So nested loops are probably not allowed. Let me think. Maybe using prefix and suffix products. Like, for each element, the product of all elements before it (prefix) multiplied by the product of all elements after it (suffix). That makes sense.
- For example, take nums = [1,2,3,4]. For index 2 (element 3), the prefix would be 1*2=2, the suffix would be 4, so the product is 2*4=8. Which is correct because the product except self for 3 is 1*2*4=8.
- So, maybe I can create two arrays: left and right. The left array would store the product of all elements to the left of i, and the right array the product to the right. Then, the answer is left[i] * right[i] for each i.
- But the problem says that the space complexity should be O(1) except for the output array. Wait, the problem might allow using the output array as part of the space. Let me check the note: "The output array does not count as extra space for space complexity analysis." So that's allowed. So maybe I can first compute the left products in the output array, then compute the right products on the fly and multiply them with the left products stored in the output array.
- Let me outline the steps:
- Initialize the answer array with all 1s. The same size as nums.
- Compute the left products for each element. For the first element, there's nothing to the left, so left product is 1. Then for each i from 1 to n-1, answer[i] = answer[i-1] * nums[i-1].
- Then compute the right products. Start from the end. We can keep a running product variable. Initialize right_product to 1. Then, for each i from n-1 down to 0, multiply the answer[i] by right_product, then update right_product by multiplying nums[i].
```
public class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] answer = new int[n];
        Arrays.fill(answer, 1);
        
        // Left pass: compute product of elements to the left of each index
        for (int i = 1; i < n; i++) {
            answer[i] = answer[i - 1] * nums[i - 1];
        }
        
        int rightProduct = 1;
        // Right pass: compute product of elements to the right and multiply with left product
        for (int i = n - 1; i >= 0; i--) {
            answer[i] *= rightProduct;
            rightProduct *= nums[i];
        }
        
        return answer;
    }
}
```
7. [Valid Anagram](https://leetcode.com/problems/valid-anagram/description/) - O(n) & O(1)
- Question: Determine if two strings are anagrams, we need to check if they contain the same characters with the same frequencies.
- Solution: 1.
- Hmm, anagrams mean that the characters are the same in both strings but rearranged, right? So first, I should think about the approach.
- Let's see. If the two strings are of different lengths, then they can't be anagrams. So the first check should be to compare the lengths. If they are different, return false immediately.
- If the lengths are the same, then I need to count the frequency of each character in both strings. One way to do this is by using a hash map. Like, for each character in the first string, increment its count, and for the second string, decrement the count. Then check if all counts are zero.
- Wait, but maybe using an array is more efficient, especially since there are only 26 lowercase letters possible? Oh, wait, the problem description says if the input strings contain unicode characters. Oh right, the original problem might have Unicode. Wait, but the problem here says s and t consist of lowercase English letters. Wait, looking back at the code, the function is isAnagram, which is a classic problem. Assuming that the problem is the standard one where s and t are anagrams if they have the same character counts.
- So, assuming that the input is lowercase letters. So maybe using an array of size 26 is sufficient.
- So here's the plan:
- Check if the lengths of s and t are different. If yes, return false.
- Create an array of 26 integers to represent the count of each character.
- Iterate through each character in s, increment the count for that character.
- Iterate through each character in t, decrement the count for that character.
- After processing both strings, check if all elements in the array are zero. If any is not zero, return false. Otherwise, return true.
```
class Solution {
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        int[] counts = new int[26];
        for (int i = 0; i < s.length(); i++) {
            counts[s.charAt(i) - 'a']++;
            counts[t.charAt(i) - 'a']--;
        }
        for (int c : counts) {
            if (c != 0) {
                return false;
            }
        }
        return true;
    }
}
```
8. [Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/description/) - O(nl) & O(nl) n is the number of strings,l is average length of the string
- Solution: 1.
- So, maybe the standard approach is to encode the length of each string before the string itself. Like, for each string, we write its length, then a delimiter, then the string. That way, when decoding, we can first read the length, then know exactly how many characters to take for the next string.
- Wait, but what if the string itself contains numbers or the delimiter? For example, if the delimiter is a comma, and the string has a comma, that would mess up the decoding. So the key is to use a delimiter that's not part of the string's content, but how?
- Alternatively, using a non-printable character as a delimiter might work, but I'm not sure if that's allowed. Or perhaps we can encode the length in a way that's unambiguous. Like, using a fixed-length prefix for the length. For example, always using 4 characters to represent the length, padded with zeros. Then, the decoder would read the first 4 characters to get the length, then take that many characters as the string. But this approach could be problematic if the strings are very long, but 4 digits (up to 9999) might be sufficient for most cases. Or maybe use 8 digits for larger lengths. But the problem might not specify constraints on string length.
- Wait, but the problem says "encode and decode strings" in Java. So, perhaps the standard way is to prefix each string with its length followed by a delimiter. For example, "5/hello3/abc" would encode two strings: "hello" and "abc". The '/' here is the delimiter between the length and the string. Wait, but if the string contains '/', that's part of the string, so the decoder needs to know to read the exact number of characters specified after the delimiter.
- Wait, in that example, when encoding "hello" which is 5 characters, you write "5/hello", then for "abc" it's "3/abc". Then, the entire encoded string would be "5/hello3/abc". Then, during decoding, we process the encoded string by first finding the '/', take the number before it as the length, then take the next 'length' characters as the string. Then, continue processing the remaining string.
- But how to handle cases where the length itself contains non-digit characters? For example, if the length is prefixed but the number is not properly formed. But since we're encoding, we can ensure that the length is correctly written as a number followed by the delimiter. So during encoding, we convert the integer length to a string, add the delimiter, then add the actual string.
- So the steps are:
- Encode:
- For each string in the input list, write the length of the string, followed by a delimiter (like '/'), followed by the string itself.
- Concatenate all these together into one encoded string.
- Decode:
- Parse the encoded string by first reading until the delimiter to get the length, then read the next 'length' characters as the string, and repeat until the entire encoded string is processed.
- But how to choose the delimiter? Using '/' could be a problem if the original string contains '/', but in this encoding scheme, the delimiter is only used once per string, right after the length. So during decoding, the first '/' encountered after processing the length is part of the format, not the string. So even if the string contains '/', since we read exactly 'length' characters after the delimiter, those '/'s would be part of the string.
- Wait, yes. Let's see. For example, encoding a string like "ab/cd". The length is 5. So encoded as "5/ab/cd". Then during decoding, when we parse the length as 5, we take the next 5 characters after the '/', which are "ab/cd". So even though there's a '/' in the string, it's correctly read because we are taking exactly 5 characters. So the delimiter after the length is not part of the string, and the string can contain any characters, including the delimiter, because the decoder knows exactly how many characters to read.
- Yes, that makes sense. So the delimiter is just a separator between the length and the string. The string can contain any characters, including the delimiter, as long as the length is correctly specified.
- So, the encoding process would be:
- For each string s in the input list:
- Convert the length of s to a string.
- Append the delimiter (e.g., '/') to the length.
- Append the original string s.
- Concatenate all these parts.
- The decoding process would be:
- Initialize a pointer at the start of the encoded string.
- While the pointer hasn't reached the end:
- Find the next occurrence of the delimiter '/' starting from the pointer.
- The substring from the pointer to the delimiter is the length.
- Convert that substring to an integer.
- Move the pointer past the delimiter (i.e., pointer = delimiter index + 1).
- Take the next 'length' characters starting from the pointer. Those are the original string.
- Add this string to the result list.
- Move the pointer forward by 'length' positions.
```
public class Codec {

    // Encodes a list of strings to a single string.
    public String encode(List<String> strs) {
        StringBuilder encoded = new StringBuilder();
        for (String s : strs) {
            encoded.append(s.length()).append('/').append(s);
        }
        return encoded.toString();
    }

    // Decodes a single string to a list of strings.
    public List<String> decode(String s) {
        List<String> decoded = new ArrayList<>();
        int i = 0;
        while (i < s.length()) {
            int slash = s.indexOf('/', i);
            int length = Integer.parseInt(s.substring(i, slash));
            String str = s.substring(slash + 1, slash + 1 + length);
            decoded.add(str);
            i = slash + 1 + length;
        }
        return decoded;
    }
}
```
9. [Roman to Integer](https://leetcode.com/problems/roman-to-integer/description/) - O(n) & O(1)
- Question: Converting a Roman numeral string to an integer, we need to handle both additive and subtractive combinations of Roman numeral symbols. 
- Solution: 1.
- Let's think about how Roman numerals work. The basic symbols are I, V, X, L, C, D, M, which correspond to 1, 5, 10, 50, 100, 500, 1000. But there are cases where subtraction is used, like IV for 4 or IX for 9. Similarly, XL is 40, CD is 400, etc.
- So the idea is that normally, each character adds its value, but if a smaller number comes before a larger one, it means subtract. For example, in IV, I (1) is before V (5), so it's 5-1=4.
- So the plan is to iterate through each character in the string, compare the current value with the next one. If the current is less than the next, subtract it; otherwise, add it.
- Wait, but how to handle that? Let's see. For each symbol, check the next symbol. If current < next, then subtract current from the total, else add current. Wait, no. Wait, for example, in IV: we have I (1) followed by V (5). So the total is 5 -1 =4. But how to compute that. Maybe we can start from the end. Or maybe process each character, and add its value, but if the previous character is less than current, subtract twice the previous value. Hmm.
- Another approach: Initialize total to 0. For each character, add its value. If the previous character is less than current, subtract twice the previous value. Because, for example, IV: when we process I first, add 1. Then process V, add 5. But since I (1) is less than V (5), we subtract 2*1. So total becomes 1+5 -2=4. That works.
- So the steps would be:
- Create a mapping of Roman characters to integers. Like a hashmap or a switch case.
- Initialize total to 0.
- Iterate over each character in the string from left to right.
- For each character, get its value. Add to total.
- Check if the previous character's value is less than the current. If so, subtract twice the previous value. Because when we added the previous value, it was part of the total, but since it's part of a subtractive pair, we need to subtract it twice: once to reverse the previous addition, and once to account for the subtraction. Like in IV: first add I (1), then add V (5). Now total is 6. But since I is before V, we subtract 2*1 =2, resulting in 4.
```
class Solution {
    public int romanToInt(String s) {
        int total = 0;
        int prev = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            int current = getValue(c);
            total += current;
            if (prev < current) {
                total -= 2 * prev;
            }
            prev = current;
        }
        return total;
    }
    
    private int getValue(char c) {
        switch(c) {
            case 'I': return 1;
            case 'V': return 5;
            case 'X': return 10;
            case 'L': return 50;
            case 'C': return 100;
            case 'D': return 500;
            case 'M': return 1000;
            default: return 0;
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
3. [Best Time to Buy and Sell Stock (Single trade)](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/) - O(n) & O(1)
- Question: Determine the maximum profit that can be achieved by buying a stock on one day and selling it on a different day in the future. If no profit is possible, we return 0.
- Solution: 1. Track the minimum price encountered so far as we iterate through the array of stock prices. For each day, we calculate the potential profit if we were to sell the stock on that day (current price minus the minimum price seen so far). By keeping track of the maximum profit encountered during this iteration
- Hmm, let's think. The brute force approach would be to check every possible pair of buy and sell days. Like, for each day i as buy day, check all days j > i and calculate the profit (prices[j] - prices[i]). Then keep track of the maximum profit found. But that's O(n^2) time, which is probably not efficient enough for large arrays. Since the constraints could have up to like 1e5 elements, O(n^2) would be way too slow. So I need a better approach.
- Wait, what's the optimal way here? Maybe track the minimum price seen so far as we iterate through the array. Then, for each day, calculate the profit if we sell on that day (current price minus the minimum so far) and keep track of the maximum profit.
- Oh right! Because the best day to sell after a certain point is when you have the highest possible price after the lowest buying price. So if we keep track of the minimum price up to the current day, then each day's potential profit is prices[i] - min_price_so_far. Then we just take the maximum of all those.
- So the steps would be:
- Initialize min_price to be the first element. Or maybe start with a very high value.
- Initialize max_profit to 0.
- Iterate through each price in the array:
a. If the current price is lower than min_price, update min_price.
b. Else, calculate the profit (current price - min_price) and see if it's larger than max_profit. If yes, update max_profit.
- After going through all elements, return max_profit.
```
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length <= 1) {
            return 0;
        }
        
        int minPrice = prices[0];
        int maxProfit = 0;
        
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            } else {
                int currentProfit = prices[i] - minPrice;
                if (currentProfit > maxProfit) {
                    maxProfit = currentProfit;
                }
            }
        }
        
        return maxProfit;
    }
}
```

4. [Best Time to Buy and Sell Stock II (Multiple trades)](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/) - O(n) & O(1)
- Question: Determine the maximum profit that can be achieved by buying and selling stocks multiple times, given that each subsequent buy must happen after the previous sell. This problem is an extension of the classic stock trading problem where multiple transactions are allowed.
- Solution: 1. Maximize our profit by capturing all positive price differences between consecutive days. This is because we can buy on a day and sell on the next day whenever the price increases, thereby accumulating all possible profits from these small trades.
- For the problem where you can buy and sell multiple times, but not hold multiple stocks (i.e., you have to sell before buying again), the solution is to accumulate all the profits from consecutive increasing days. Like, for every day where price is higher than previous, add the difference to the total profit.
- So maybe this problem is the latter case. Let me think. For example, if the prices are [7,1,5,3,6,4], then buying at 1 and selling at 5 (profit 4), buying at 3 and selling at 6 (profit 3) gives total 7.
- In that case, the approach is to sum up all the positive differences between consecutive days. So, for each i from 1 to n-1, if prices[i] > prices[i-1], add (prices[i] - prices[i-1]) to the total.
- So the code would be simple: initialize profit to 0, loop through the array from day 1, check if current price is higher than previous, add the difference to profit.
- Yes, that's the solution for the problem where you can do multiple transactions.
```
class Solution {
    public int maxProfit(int[] prices) {
        int profit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                profit += prices[i] - prices[i - 1];
            }
        }
        return profit;
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
- Question: Need to reverse a singly linked list. The goal is to reverse the direction of the pointers such that the last node becomes the first node, and each node points to its previous node instead of the next one.
- Solution: 1. Three-pointers (previous, current, and next) 2. Copy all the elements to an array and reverse it. Update LL according to the array - O(n) & O(n)
- Hmm, I remember that there's an iterative way to reverse a linked list. Let me recall. Oh right, you can use three pointers: previous, current, and next. You iterate through the list, changing the current node's next pointer to point to the previous node, then move all pointers forward.
- Wait, let me break it down step by step. Starting with the head, which is the first node. The idea is to reverse each link as you go through the list.
- Let's see. Initially, previous is null, current is head. Then, while current is not null, save the next node (next = current.next), then set current.next to previous. Then, move previous to current, current to next. Repeat until current is null. The new head is previous.
- Yes, that makes sense. So the steps would be:
- Initialize previous as null, current as head.
- Loop while current is not null:
a. Save next node (next = current.next)
b. Reverse the link (current.next = previous)
c. Move previous and current forward (previous = current, current = next)
- Return previous as the new head.
```
public class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode current = head;
        while (current != null) {
            ListNode nextTemp = current.next;
            current.next = prev;
            prev = current;
            current = nextTemp;
        }
        return prev;
    }
}
```
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
9. [LRU Cache](https://leetcode.com/problems/lru-cache/description/) - O(1) & O(capacity)
- Question: Implement an LRU (Least Recently Used) cache that supports get and put operations in O(1) average time complexity. The LRU cache evicts the least recently used item when it reaches its capacity.
- Solution: 1. HashMap and a doubly linked list.
- Hmm, what's LRU again? Oh right, Least Recently Used. So when the cache reaches capacity, the least recently accessed item gets evicted. Right. So the operations required are get and put.
- For the data structures, I remember that a combination of a hash map and a doubly linked list is commonly used. The hash map allows O(1) access to the nodes, and the doubly linked list allows maintaining the order of usage. Because when a node is accessed (get or put), it needs to be moved to the front (most recently used). And when inserting a new node, if the capacity is exceeded, we remove the last node (least recently used).
- So the plan is: create a Node class that has key, value, prev, next pointers. The doubly linked list will have a head and tail dummy nodes to make edge cases easier. The hash map will map keys to their corresponding nodes.
- Wait, for the linked list, the head could be the most recently used, and the tail is the least. So every time a node is accessed, it's moved to the head. When evicting, we remove the node before the tail.
- So let's outline the steps:
- When initializing the LRUCache, we need to set the capacity. Also, initialize the hash map (like a HashMap in Java), the head and tail nodes. The head and tail are connected initially.
- For the get method:
- Check if the key exists in the hash map.
- If not, return -1.
- If it does, move the node to the front (after head), then return the value.
- For the put method:
- If the key exists, update the value and move the node to the front.
- If it doesn't exist:
- Check if the capacity is reached. If yes, remove the node before the tail (least recently used), and also remove it from the hash map.
- Create a new node, add it to the hash map, and insert it after the head.
- Wait, but inserting after the head would make it the most recent. Yes.
- So the helper functions needed would be:
- addNode(Node node): adds a node right after the head.
- removeNode(Node node): removes the node from the linked list.
- moveToHead(Node node): removes the node and adds it after the head.
- popTail(): removes the node before the tail and returns it (for eviction).
- Wait, maybe the steps are:
- For adding a new node, add it right after the head. Then add to the map. If the capacity is exceeded, we remove the node before tail (tail.prev) and remove it from the map.
- Now, in Java, the data structures. The hash map is straightforward using a HashMap<Integer, Node>. The linked list needs to be managed manually.
- So the Node class should have key and value, because when we remove the tail node (for eviction), we need to get the key to remove it from the map.
- Wait, when we evict the least recently used node (the node before tail), we need to get its key so we can remove it from the hash map. So each Node must store the key and value.
- So the Node structure would be:
class Node {
  int key;
  int value;
  Node prev;
  Node next;
  public Node(int k, int v) {
    key = k;
    value = v;
  }
}
- Then, the LRUCache class will have:
- capacity: int
- size: int (current number of elements)
- map: HashMap<Integer, Node>
- head, tail: Node
- In the constructor:
- Initialize head and tail as dummy nodes. Connect head.next to tail, and tail.prev to head.
- For the get method:
- Check if the key is in the map. If not, return -1. Otherwise, get the node, move it to the front (using moveToHead), then return the value.
- For the put method:
- If the key exists in the map, update the node's value and move it to the front.
- If the key doesn't exist:
- Create a new node.
- Add the node to the map.
- Add the node to the front (addNode).
- Increment size.
- If size exceeds capacity, remove the tail.prev node (popTail), remove it from the map, decrement size.
- Wait, but when you add a new node, you have to check if capacity is reached. So the steps would be:
- Check if key exists. If not:
- create new node
- add to map
- add to head
- increment size
- if size > capacity:
- evict the last node (tail.prev)
- remove from map
- decrement size
- But when evicting, the size would be capacity +1, so after removal, size becomes capacity.
- Now, implementing the helper methods.
- AddNode: add the node right after the head.
- node.prev = head;
node.next = head.next;
head.next.prev = node;
head.next = node;
Similarly, removeNode is:
node.prev.next = node.next;
node.next.prev = node.prev;
MoveToHead is just removeNode followed by addNode.
- PopTail: get the tail.prev node (the least recently used), then remove it from the list and return it.
- So, when evicting, we call popTail, get the node, remove it from the map using its key, then decrease size.
- So putting this all together.
```
public class LRUCache {

    class Node {
        int key;
        int value;
        Node prev;
        Node next;
        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    private HashMap<Integer, Node> map;
    private int capacity;
    private int size;
    private Node head, tail;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.map = new HashMap<>();
        this.size = 0;
        this.head = new Node(0, 0);
        this.tail = new Node(0, 0);
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        Node node = map.get(key);
        if (node == null) {
            return -1;
        }
        moveToHead(node);
        return node.value;
    }

    public void put(int key, int value) {
        Node node = map.get(key);
        if (node != null) {
            node.value = value;
            moveToHead(node);
        } else {
            Node newNode = new Node(key, value);
            map.put(key, newNode);
            addNode(newNode);
            size++;
            if (size > capacity) {
                Node tailNode = popTail();
                map.remove(tailNode.key);
                size--;
            }
        }
    }

    private void addNode(Node node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }

    private void removeNode(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void moveToHead(Node node) {
        removeNode(node);
        addNode(node);
    }

    private Node popTail() {
        Node res = tail.prev;
        removeNode(res);
        return res;
    }
}
```

## Stack
- Stacks are often useful in problems where we need to track sequences and handle consecutive elements
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
2. [Remove All Adjacent Duplicates in String II](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/description/) - O(n) & O(n)
- Question: Reducing a string by removing K consecutive identical characters
- Solution: 1. Use a stack-based deque approach to efficiently track consecutive characters and their counts. 2. Brute-force approach would be to scan the string for k-length runs, remove them, and repeat until no more can be removed.
-  I have to reduce a string by removing k consecutive identical characters. Let's see. So the task is, given a string like "aabbccd" and a k value, say 2, I have to remove any group of exactly k identical consecutive characters. But wait, maybe it's any group of at least k? Wait, the problem says "K consecutive identical characters". So if there are more than k, like 3, and k is 2, do I remove 2 and leave 1? Or do I need to remove exactly k?
- Wait, looking back at the problem statement: "Reduce the string by removing K consecutive identical characters". So perhaps whenever there are k consecutive identical characters, they are removed. But after removal, the remaining parts are concatenated, and this process is repeated until no more K-length consecutive duplicates can be removed. For example, if the string is "aabbccd" and k=2, then removing "aa", "bb", "cc" would leave "d".
- Alternatively, maybe we need to do it in a single pass. Hmm. But how? Because after removing some parts, new groups of k duplicates might form. For example, consider "aaabbb" and k=3. Remove the first three 'a's to get "bbb", then remove those three 'b's. But if the input is "abbbaa" and k=3, first the "bbb" is removed, then the "aaa" is formed and removed.
- So the approach would need to process the string and handle these cases where after a removal, new k-length groups are formed.
- The challenge is to do this efficiently. For example, a brute-force approach would be to scan the string for k-length runs, remove them, and repeat until no more can be removed. However, this could be inefficient for large strings because each iteration requires scanning the entire string again.
- So, how can we do this more efficiently? Let's think about using a stack-based approach. Stacks are often useful in problems where we need to track sequences and handle consecutive elements.
- The idea is to use a stack that keeps track of the characters and their counts. For each character in the input string, we check the top of the stack. If the top character is the same as the current one, we increment the count. If the count reaches k, we pop the character from the stack. Otherwise, we update the count. If the top character is different, we push the current character with a count of 1.
- This approach processes the string in a single pass and uses a stack to keep track of consecutive counts. Let's see how this works with an example.
- But what about a case where after popping, the previous elements combine? For example, "abbaaa" with k=2. Processing 'a' (count 1), 'b' (count 1), then another 'b' (count 2, so pop 'b's). Now the stack has 'a' with count 1. Then 'a's start: count becomes 2 (after first two), then third 'a' makes count 3. Since k=2, but count 3, so we can't pop. Wait, but k is 2. So after two 'a's, the count is 2, which would be popped. Then the third 'a' would be pushed again with count 1. So the final string would be 'a'.
- Wait, let's walk through "abbaaa" with k=2:
- Initial stack: empty.
Process 'a': stack becomes [ ('a',1) ]
Process 'b': stack becomes [ ('a',1), ('b',1) ]
Process 'b': stack top is 'b', so increment count to 2. Now count ==k=2, so pop. Stack becomes [ ('a',1) ]
Process 'a': stack top is 'a', count becomes 2. Now count ==k, so pop. Stack is empty.
Process 'a': stack becomes [ ('a',1) ]
Process 'a': stack top is 'a', count becomes 2. Pop. Stack is empty.
- So the plan is to implement a stack where each element is a pair of (character, current consecutive count). For each character in the input string, check the top of the stack. If same character, increment count. If the count reaches k, pop. Otherwise, update the count. If different, push the new character with count 1.
- Once the entire string is processed, the stack contains the remaining characters with their counts, which can be used to build the resulting string.
- In Java, a stack can be implemented using a Deque or a custom class to hold the character and count. Each stack element can be an object of a helper class, or we can use an array of objects. Alternatively, since Java doesn't have tuples, we can use a class with two fields: character and count.
- But perhaps the most straightforward way is to use a Stack of objects, each containing the character and the count.
- So, first, define a helper class:
- Then, create a stack of Pair objects.
- Initialize the stack as empty.
- Loop through each character in the input string:
- For each character:
- If the stack is not empty and the top element's character is equal to current character:
- Increment the count.
- If the count == k, then pop the top element.
- Else, update the top element's count.
- Else:
- Push a new Pair with current character and count 1.
```
class Solution {
    public String reduceString(String s, int k) {
        Deque<Pair> deque = new ArrayDeque<>();
        for (char c : s.toCharArray()) {
            if (!deque.isEmpty() && deque.peekLast().c == c) {
                Pair top = deque.peekLast();
                top.count++;
                if (top.count == k) {
                    deque.removeLast();
                }
            } else {
                deque.addLast(new Pair(c, 1));
            }
        }

        StringBuilder sb = new StringBuilder();
        for (Pair p : deque) {
            for (int i = 0; i < p.count; i++) {
                sb.append(p.c);
            }
        }
        return sb.toString();
    }

    private static class Pair {
        char c;
        int count;
        Pair(char c, int count) {
            this.c = c;
            this.count = count;
        }
    }
}
```
3. [Remove All Adjacent Duplicates In String](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/description/) - O(n) & O(n)
- Question: Need to repeatedly remove  two adjacent and equal duplicate characters from a string until no such duplicates remain.
- Solution: 1. Using Stack 2. Brut force - O(n^2)
- The example given is like, if the input is "abbaca", the output should be "ca". Because first, the two 'b's are adjacent and removed, resulting in "aaca", then the two 'a's are adjacent, leaving "ca".
- Hmm, how do I approach this? Let's think. The straightforward way is to simulate the process. But simulating each step by checking adjacent pairs and removing them might be inefficient, especially for large strings. Because each removal could cause new adjacent duplicates. So like, in the worst case, the string could be all the same characters, and each step removes two, which would take O(n^2) time. That's probably not good enough for big inputs.
- Wait, maybe there's a smarter way. Oh right! Using a stack. Because when we process each character, we can check if it's the same as the top of the stack. If it is, then we pop it (since they form a pair and get removed). Otherwise, push the current character onto the stack. This way, we process the string in one pass, which is O(n) time. Let's see how that works.
- Let me think of an example. Take "abbaca". Let's process each character:
- Start with stack empty. First character 'a' is pushed. Stack: [a]
- Next 'b' is not the same as 'a', push. Stack: [a, b]
- Next 'b' is same as top. Pop. Stack becomes [a]
- Next 'a' is same as top. Pop. Stack is empty.
- Next 'c' is pushed. Stack: [c]
- Next 'a' is pushed. Stack: [c, a]
- So the final stack is ['c','a'], which is "ca". That works.
- So the plan is:
- Initialize a stack (could use a Deque in Java, or even a StringBuilder as a stack for efficiency).
- Iterate through each character in the string.
- For each character, check if the stack is not empty and the top element is equal to the current character.
- If yes, pop the top element.
- If no, push the current character.
- At the end, convert the stack to a string and return.
- Now, implementing this in Java. Let's see.
- Using a StringBuilder as a stack:
- Initialize a StringBuilder. For each character in s:
- Check if the length of the StringBuilder is > 0 and the last character is equal to current char.
- If yes, delete the last character.
- Else, append the current character.
```
class Solution {
    public String removeDuplicates(String s) {
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (sb.length() > 0 && sb.charAt(sb.length() - 1) == c) {
                sb.deleteCharAt(sb.length() - 1);
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }
}
```
## Queue

## Heap 
- Priority Queue
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

## Trie 
- O(L) and O(N*L)
1. [Implement Trie - Prefix Tree](https://leetcode.com/problems/implement-trie-prefix-tree/description/) - Insert : O(n) & O(n), Search and Prefix check : O(n) & O(1)
- Question: insert, search, startsWithPrefix
- Solution: Trie has Node[] children; boolean eow;
```
class Trie {

    class Node {
        Node[] children;
        boolean eow;

        public Node() {
            children = new Node[26];
            eow = false;
        }
    }

    Node root; 

    public Trie() {
        root = new Node();
    }
    
    public void insert(String word) {
        Node current = root;
        for(int i=0; i<word.length(); i++) {
            int index = (int) (word.charAt(i) - 'a');
            if(current.children[index] == null) {
                Node node = new Node();
                current.children[index] = node;
            }
            current = current.children[index];
        }
        current.eow = true;
    }
    
    public boolean search(String word) {
        Node current = root;
        for(int i=0; i<word.length(); i++) {
            int index = (int) (word.charAt(i) - 'a');
            if(current.children[index] == null) {
                return false;
            }
            if((i == word.length() - 1) && current.children[index].eow == false) {
                return false;
            }
            current = current.children[index];
        }
        return true;
    }
    
    public boolean startsWith(String prefix) {
        Node current = root;
        for(int i=0; i<prefix.length(); i++) {
            int index = (int) (prefix.charAt(i) - 'a');
            if(current.children[index] == null) {
                return false;
            }
            // if((i == word.length() - 1) && current.children[index].eow == false) {
            //     return false;
            // }
            current = current.children[index];
        }
        return true;
    }
}
```
2. Word Break 
- Question: Break the word into multiple parts and check whether all parts are present in the dictionary
- Solution: 1. Reverse DP with dp[i] = dp[i + w.length()] 2. Trie insert, search and recursion helper for dividing string 
3. Count Unique Substrings / Count Trie Nodes
- Solution: 1. HashSet with Nested loops to get a substring 2. Trie insert and count nodes
4. Longest word with all prefixes / Longest Word in Dictionary
- Solution: 1. HashSet contains its prefix of (length - 1) 2. Trie inserts and builds a string with it's all prefixes present in the dictionary along with recursion and backtracking
5. [Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/description/) - addWord : O(L) & O(L), searchWord : O(26^L) & O(L) L is the length of the word.
- Question: Design a data structure that allows adding words and searching for words efficiently, even when the search includes wildcard characters ('.') that can match any single character. 
- Solution: 1.  Trie (prefix tree)
- First, the addWord function. That's straightforward. Like a normal trie insertion. Each node has children for each character. So when adding a word, we go through each character and create nodes as needed.
- The tricky part is the search function. Especially when there's a '.', because that can match any character. For example, if the word is ".ad", then the first character can be any letter, and then 'a' and 'd' must follow. So during the search, whenever we encounter a '.', we need to check all possible children of the current node for the remaining part of the word.
- So maybe the trie structure is the way to go here. Let's think about the trie node structure. Each node will have a map or array of children, and a flag indicating if it's the end of a word.
- Wait, since the characters are lowercase letters, we can use an array of 26 elements (for each letter a-z). So each node has an array of 26 children, and a boolean isEnd.
- So for addWord, we iterate through each character in the word, and for each step, check if the child exists. If not, create a new node. Then move to that child. At the end, set isEnd to true.
- For search, we need a recursive approach, especially when there's a wildcard. Because when we hit a '.', we need to check all possible children. So the search function could be implemented with a helper that takes the current node and the current index in the word.
- For example, the helper function could look like this: boolean searchInNode(String word, int index, TrieNode node). For each character starting at index, if the character is '.', then we loop through all possible children (non-null ones) and recursively check if the rest of the word can be matched. If any of them return true, then return true. If the character is a letter, check if the child exists, and proceed recursively.
- So the main search function would call this helper starting from the root node and index 0.
- Let me outline this.
- The TrieNode structure:
class TrieNode {

TrieNode[] children = new TrieNode[26];

boolean isEnd = false;

}

- The WordDictionary class will have a root TrieNode.
- In the constructor, initialize the root.
- AddWord: start from root, iterate through each character in the word. For each char, calculate the index (c - 'a'). If the child at that index is null, create a new node. Move to the child. At the end, mark isEnd.
- Search: For the given word, start at the root. For each character in the word:
- If the character is '.', we need to check all existing children. For each child that is not null, recursively check if the remaining substring (from next index) matches. If any of them returns true, return true. Otherwise, false.
- If it's a normal character, check if the child exists. If not, return false. Otherwise, proceed to the next character.
- So the base case is when the index reaches the end of the word. We check if the current node's isEnd is true.
- Now, implementing this in Java.
- Potential edge cases: adding empty string? But probably the problem says words are non-empty? The problem statement says "addWord adds word to the data structure", but the constraints may not specify. So need to handle words of length zero? Probably not, but perhaps in the problem constraints.
- Another edge case: searching with a word that's longer than any added word, but has wildcards. But the trie structure would naturally handle that by checking the depth.
```
class WordDictionary {

    private class TrieNode {
        TrieNode[] children = new TrieNode[26];
        boolean isEnd = false;
    }

    private TrieNode root;

    public WordDictionary() {
        root = new TrieNode();
    }
    
    public void addWord(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) {
                node.children[index] = new TrieNode();
            }
            node = node.children[index];
        }
        node.isEnd = true;
    }
    
    public boolean search(String word) {
        return searchHelper(word, 0, root); //For recursion call
    }

    private boolean searchHelper(String word, int index, TrieNode node) {
        if (index == word.length()) {
            return node.isEnd;
        }
        char c = word.charAt(index);
        if (c == '.') {
            for (TrieNode child : node.children) {
                if (child != null && searchHelper(word, index + 1, child)) {
                    return true; //return searchHelper(word, index + 1, child);
                }
            }
            return false;
        } else {
            int i = c - 'a';
            if (node.children[i] == null) {
                return false;
            }
            return searchHelper(word, index + 1, node.children[i]);
        }
    }
}
```
6. [Word Search II](https://leetcode.com/problems/word-search-ii/description/) - O(W⋅L+M×N×4^L) & O(W⋅L+L) W is the number of words in words, L is the average length of each word.
- Question: Find all words from a given list that can be formed by sequentially adjacent cells on a 2D board. Adjacent cells are those horizontally or vertically neighboring, and the same cell can't be used more than once in a word. 
- Solution: 1. Trie (prefix tree) data structure to store the words and perform depth-first search (DFS) with backtracking to explore all possible paths on the board.
- First, I remember that for searching words in a grid, a common approach is to use backtracking with a trie. The trie helps in efficiently checking the prefixes, so we can stop early if a prefix isn't present. That way, we don't waste time exploring paths that can't form any of the words.
- Let me think. The steps might be:
- Build a trie from the list of words. Each node in the trie represents a character, and paths from the root to leaves represent the words.
- Then, for each cell in the board, perform a depth-first search (DFS) to explore all possible paths, checking if the sequence of characters forms a prefix present in the trie.
- When a word is found (i.e., we reach a node in the trie marked as the end of a word), add it to the result list and perhaps mark that node so we don't add duplicates if the same word is found again through another path.
- To avoid revisiting the same cell during the current DFS path, we can mark the cell as visited (maybe by temporarily changing its character to a special symbol and then restoring it after backtracking).
- Wait, but building a trie for all the words makes sense. Then, during the DFS, we can check if the current path is a valid prefix. If not, we can backtrack immediately. That's more efficient than checking each word individually, especially if there are many words.
- But how to handle duplicates? For example, if the same word can be found in multiple ways. The problem states that the output list should have all the words on the board, but each instance should be present once. So when a word is found, we need to add it to the result once, even if found multiple times. So perhaps once a word is found, we can remove it from the trie to prevent duplicates. Or mark it as found so we don't add it again.
- Wait, another idea: during the trie traversal, once a word is found (i.e., a node's isEnd is true), we can add that word to the result set and then set isEnd to false to avoid adding duplicates. But we need to collect all unique words, so maybe using a set to store the results and then converting it to a list at the end would handle duplicates automatically. That might be a good approach.
- So the plan is:
- Build a trie from the given words.
- Use a set to collect the found words to avoid duplicates.
- For each cell in the board, start a DFS, tracking the current path in the trie.
- If during DFS we reach a node where isEnd is true, add the corresponding word to the set.
- To optimize, once a word is found, we can prune the trie node's isEnd to prevent adding the same word multiple times (but maybe it's better to just let the set handle duplicates, but pruning could save some time in further searches).
- Wait, but building the trie for all words and then checking each possible path on the board. So the steps in code would be:
- Create the Trie structure with nodes. Each node has children and a boolean indicating if it's the end of a word. Also, maybe each node can store the actual word, but perhaps it's better to build the word as we traverse the trie. Alternatively, when inserting a word into the trie, each node along the path can track the characters, and when we reach the end node, we have the full word. Alternatively, during the DFS traversal, we can keep track of the current string being built.
- Wait, maybe during the insertion of words into the trie, each end node can store the word itself. That way, when we reach an end node during DFS, we can immediately get the word without having to track the characters from the root. That could be efficient.
- So, for example, when inserting "apple", each node in the path a -> p -> p -> l -> e would have their respective children. The last node (e) would have isEnd = true and word = "apple".
- That way, during the DFS, whenever we reach a node that isEnd, we can add node.word to the result set.
- Now, to implement this:
- The TrieNode class would have a map or an array of children (since there are 26 letters) and a String word to store the word at the end node.
So, steps for the solution:
- Insert all the words into the trie. For each word, traverse the trie, adding nodes as needed. At the end node, set isEnd (or word != null) to indicate the end of a word and store the word there.
- Iterate over each cell in the board. For each cell, check if the character exists as a child of the trie root. If yes, start a DFS from that cell.
- In the DFS, for the current cell (i,j) and current trie node:
a. Check if the current cell's character matches the current trie node's child.
b. If the current trie node has a word (isEnd), add it to the result set.
c. Mark the current cell as visited (maybe by changing board[i][j] to a special character like '#').
d. Explore all four directions (up, down, left, right), checking if the next cell is within bounds and not visited.
e. For each direction, proceed to the next trie node (child of current node) if the next character exists.
f. After backtracking, restore the cell's original character.
- Once all cells are processed, convert the result set to a list and return.
- But wait, the root of the trie is the starting point. For each cell, we need to check if the root has a child corresponding to the cell's character. So, for example, if the cell is 'a', check if root.children['a' - 'a'] exists. If yes, proceed with the DFS.
- Another thing to note: the same cell cannot be used more than once in the same word. So during the DFS, we need to track visited cells. The standard approach for this is to mark the cell as visited temporarily, then unmark it when backtracking.
```
class Solution {
    class TrieNode {
        TrieNode[] children = new TrieNode[26];
        String word;
    }

    public List<String> findWords(char[][] board, String[] words) {
        TrieNode root = new TrieNode();
        for (String word : words) {
            insert(root, word);
        }
        
        Set<String> result = new HashSet<>();
        int m = board.length;
        int n = board[0].length;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                char c = board[i][j];
                int idx = c - 'a';
                if (root.children[idx] != null) {
                    dfs(board, i, j, root.children[idx], result);
                }
            }
        }
        
        return new ArrayList<>(result);
    }
    
    private void insert(TrieNode root, String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            int idx = c - 'a';
            if (node.children[idx] == null) {
                node.children[idx] = new TrieNode();
            }
            node = node.children[idx];
        }
        node.word = word;
    }
    
    private void dfs(char[][] board, int i, int j, TrieNode node, Set<String> result) {
        if (node.word != null) {
            result.add(node.word);
            node.word = null; // Prevent duplicate entries
        }
        
        char originalChar = board[i][j];
        board[i][j] = '#'; // Mark as visited
        
        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int[] dir : directions) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || x >= board.length || y < 0 || y >= board[0].length) {
                continue;
            }
            char nextChar = board[x][y];
            if (nextChar == '#') continue;
            int idx = nextChar - 'a';
            if (node.children[idx] != null) {
                dfs(board, x, y, node.children[idx], result);
            }
        }
        
        board[i][j] = originalChar; // Restore the original character
    }
}
```

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
3. [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/description/?) - O(n) & O(h) n is the number of nodes in the tree,h is the height of the tree
- Question: Determine if a binary tree is a valid Binary Search Tree (BST), we need to ensure that for every node, all nodes in its left subtree have values less than the node's value, and all nodes in its right subtree have values greater than the node's value. This must hold true for all nodes recursively.
- Solution: 1. Using a helper function that tracks the valid range (minimum and maximum values) for each node. Starting from the root node, the valid range is between negative infinity and positive infinity.
- Wait, right. So for example, if the root is 5, and the left child is 3, but somewhere in the left subtree there's a 6, that's invalid. So I need to make sure that all nodes in the left are less than 5, and all nodes in the right are greater than 5. And this applies recursively for every node.
- So how to approach this? Maybe using a recursive approach where each node must lie within a certain range. Like, for each node, there's a minimum and maximum value it can have. For the root, the min could be negative infinity and max positive infinity. Then, when we go to the left child, the max becomes the root's value, and the min remains the same. Similarly, for the right child, the min is the root's value and the max remains the same.
- Wait, that makes sense. For example, the root is 5. The left child must be <5, and all descendants must also be <5. Then, for that left child (say 3), its right subtree must be >3 but still <5. So each time, we update the allowed range for the child nodes.
- So the idea is to carry along the current node's allowed min and max. Let's see:
- Base case: if the node is null, return true (since an empty tree is a valid BST).
- Otherwise, check if the current node's value is within the (min, max) range. Then, recursively check the left and right subtrees. For the left subtree, the max becomes the current node's value. For the right subtree, the min becomes the current node's value.
- But wait, the original problem states that a valid BST has left nodes with values strictly less than the parent and right nodes strictly greater. So the range for left is (-infinity, node.val), and for right is (node.val, +infinity). So when checking the current node, it must be greater than the min and less than the max. So for the left subtree, the max is updated to current node's value. For the right, the min is updated to current node's value.
- So in code, the helper function would take a node, a lower bound and an upper bound. Initially, for the root, the lower is -infinity and upper is +infinity.
- Wait, but in Java, how to represent those? Since integers can be up to certain values, but the problem uses the TreeNode which has int values. Wait, no, the problem's TreeNode is defined with int val? Or is it Integer? Wait, the code given uses TreeNode which is likely the standard one where val is int. So, but if the BST can have values up to Integer.MAX_VALUE, then how to handle the initial upper and lower? For example, if a node has the value Integer.MAX_VALUE, then its right child must be greater than that, which is impossible. But according to BST rules, the right subtree must have values greater than the node. So if the node's value is Integer.MAX_VALUE, then it can't have a right child. Similarly for left child with Integer.MIN_VALUE.
- So for the initial call, the lower can be null (meaning no lower bound) and upper can be null (meaning no upper bound). Wait, but how to handle that in Java. Because if the node's value is, say, 5, then the left must be less than 5. So we can represent lower as a Long to handle the cases where the value is Integer.MAX_VALUE. Because if we use Integer, then when the upper is Integer.MAX_VALUE, and the node's value is that, the right child would have to be greater, but that's impossible. Wait, but using Long.MIN_VALUE and Long.MAX_VALUE as the initial bounds. Because if we use Long, even if the node's value is Integer.MAX_VALUE, the upper bound for the right subtree is Long.MAX_VALUE, which is larger.
- Alternatively, maybe using Long for the initial min and max. So for the root, the lower is Long.MIN_VALUE and upper is Long.MAX_VALUE. Then, for each node, check if its value is greater than lower and less than upper. Then, for the left subtree, upper becomes the node's value (as a long), and lower remains the same. Similarly for the right subtree, lower becomes the node's value.
- This way, even if the node's value is Integer.MAX_VALUE, the right subtree's lower would be Integer.MAX_VALUE, and upper remains Long.MAX_VALUE. So any right child must be greater than Integer.MAX_VALUE, but since the nodes are integers, that's impossible. So the right subtree of such a node must be empty. Which is correct.
- So in Java, the helper function could take the node, a long lower, and a long upper. Then, check if the current node's value is > lower and < upper. If not, return false. Then recursively check left and right.
```
class Solution {
    public boolean isValidBST(TreeNode root) {
        return isValid(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }
    
    private boolean isValid(TreeNode node, long lower, long upper) {
        if (node == null) return true;
        if (node.val <= lower || node.val >= upper) return false;
        return isValid(node.left, lower, node.val) && isValid(node.right, node.val, upper);
    }
}
```
4. [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/description/) - O(n) & O(h) In the worst case (a skewed tree), this would be O(n), but for a balanced tree, it would be O(log n).
- Question: Invert a binary tree such that each left subtree becomes the right subtree and vice versa. 
- Solution: 1. Post-order traversal
- Wait, how do I approach this? Maybe a recursive approach would work. Let me see. If the root is null, then return null. Otherwise, swap the left and right children, and then invert the left and right subtrees. Wait, but do I swap first and then invert the subtrees, or invert first then swap? Let me think. Suppose I have a node with left and right children. If I swap them first, then the left becomes the right, and then I invert the new left (which was the original right). So that's correct. Because after swapping, the left and right are swapped, but their own subtrees still need to be inverted.
- So the steps would be:
- If the current node is null, return.
- Swap the left and right children.
- Recursively invert the left subtree (which is the original right).
- Recursively invert the right subtree (which is the original left).
- Wait, wait. Wait after swapping, the left and right are swapped. So when I call invert on left, it's actually the original right. So maybe the order of steps 3 and 4 is correct. Let's see: For example, take a root node with left child A and right child B. Swap them, so left is B, right is A. Then invert the left (B) and invert the right (A). Which makes sense.
- Alternatively, maybe the steps are:
- invert left subtree
- invert right subtree
- swap left and right
- But does that work? Let's see. Suppose the original left and right are A and B. If I invert A (turning into A'), invert B (turning into B'), then swap left and right. So the root's left is B' and right is A'. That's correct. But that would invert the subtrees first, then swap
```
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }
}
```
5. [Same Tree](https://leetcode.com/problems/same-tree/description/) - O(n) & O(h)
- Question: Determine if two binary trees are identical, we need to check both their structure and node values.
- Solution: 1. Pre-order traversal
- First, what's the definition of being the same tree? Oh right, they need to have the same structure and the same node values. So each corresponding node must have the same value, and their left and right children must also be the same.
- So maybe a recursive approach would work here. Because for each node, we need to check if the current nodes are equal, and then check their left and right subtrees.
- Let's think about the base cases. If both p and q are null, then they are the same, so return true. If one of them is null and the other isn't, then they are different, return false. Otherwise, check if their values are equal. If the values are not equal, return false.
- Then, if the current nodes are equal, we need to check the left and right subtrees. So recursively call isSameTree on the left children of p and q, and on the right children. If both return true, then the trees are the same.
- Wait, right. So the steps are:
- Check if both nodes are null → true.
- Check if one is null and the other isn't → false.
- Check if the values are different → false.
- Otherwise, check left and right subtrees recursively.
- So putting that into code. The function will first handle the base cases. Then compare the values. Then return the logical AND of the left subtree check and the right subtree check.
```
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        if (p.val != q.val) {
            return false;
        }
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
}
```
6. [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/description/) - O(n) & O(n)
- Question: Need to perform a level order traversal of a binary tree, where each level of the tree is grouped into a sublist. The result should be a list of lists, with each sublist containing the values of nodes at the corresponding level of the tree.
- Solution: 1. Breadth-First Search (BFS)
- Hmm, the standard approach for level order traversal is using a BFS algorithm with a queue. Right. Because BFS naturally processes nodes level by level. So the plan is to use a queue to keep track of the nodes at each level.
- Wait, but how do I track when one level ends and the next begins? Oh right, in the BFS, for each level, we can process all nodes that are currently in the queue (which represents the current level) and then add their children to the queue for the next level. So for each iteration, we dequeue all elements of the current level, add their values to the current sublist, and enqueue their children. Then add that sublist to the result list.
- So here's how the steps would go:
- Initialize a queue. If the root is not null, add it to the queue.
- While the queue is not empty:
- Get the number of nodes in the queue (this is the current level size).
- Create a new list to hold the current level's values.
- For each node in the current level (loop for size times):
- Dequeue the node.
- Add its value to the current list.
- Enqueue the left and right children if they exist.
- Add the current list to the result.
- That makes sense. Now, how to implement this in Java.
- Wait, in Java, the Queue is an interface, and we can use a LinkedList as the implementation. So we can do something like Queue<TreeNode> queue = new LinkedList<>();
- But for handling each level, we need to know the size first. So for each iteration of the while loop, first get the size of the queue (which is the number of nodes in the current level). Then loop that many times, processing each node.
```
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>();
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                currentLevel.add(node.val);
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
            }
            result.add(currentLevel);
        }
        
        return result;
    }
}
```
7. [Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/) - O(h+k) & O(h) = O(log n) (balanced) to O(n) (worst case, skewed tree).
- Question: Find the kth smallest element in a Binary Search Tree (BST).
- Solution: 1. In-order traversal with stack 2. In-order traversal with recursion
- Hmm, how do I approach this? Let's think. Oh right, in a BST, the in-order traversal gives the elements in sorted order. So if I do an in-order traversal and collect the elements in a list, the k-1 index would be the answer. But wait, maybe I can optimize it so that I don't have to traverse the entire tree. Like, once I reach the kth element, I can stop early.
- Right, so the idea is to perform an in-order traversal but keep track of the count. Once the count reaches k, return that element. That way, we don't have to process all nodes if k is small.
- So how to implement this? Let's see. I can use a recursive approach for in-order traversal. But recursion might hit stack limits for very large trees. Alternatively, an iterative approach using a stack might be better. But since the problem is about a BST, and the size is probably manageable, recursion should work.
- Wait, but with recursion, even if we find the kth element early, we have to unwind all the recursive calls. So maybe better to use an iterative approach so we can stop as soon as we find the kth element.
- Let me outline the steps for iterative in-order traversal:
- Initialize a stack. Start with the root node. Then, go left as much as possible, pushing nodes onto the stack. Then pop from the stack, process the node (increment count, check if it's k). Then move to the right child of the popped node and repeat.
- But how to track the count. Let's see. Let's have a counter initialized to 0. Every time we process a node (after popping from the stack), increment the counter. When the counter equals k, return that node's value.
- Yes. So the steps would be:
- Initialize stack and current pointer to root.
- Initialize count to 0.
- Loop until stack is not empty or current is not null:
- While current is not null, push it to stack and move to left child.
- Pop node from stack. Increment count.
- If count ==k, return node.val.
- Move current to node's right.
```
class Solution {
    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode current = root;
        int count = 0;
        
        while (current != null || !stack.isEmpty()) {
            while (current != null) {
                stack.push(current);
                current = current.left; //left,root,right
            }
            current = stack.pop(); //stack has at least one element at this step
            count++;
            if (count == k) {
                return current.val;
            }
            current = current.right;
        }
        return -1; // This line is theoretically unreachable as per problem constraints
    }
}
```
8. [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/description/) - O(n) & O(h) or O(log n)
- Question: Determine the maximum depth of a binary tree. The maximum depth is defined as the number of nodes along the longest path from the root node down to the farthest leaf node.
- Solution: 1. Post-order traversal DFS
- Hmm, let's think. The maximum depth is the number of nodes along the longest path from the root down to the farthest leaf node. So, how do I approach this?
- Well, maybe using recursion. Because each subtree's depth is the maximum of its left and right subtree's depth plus one. Like, for the root node, its depth is 1 plus the max of left and right children's depth. That makes sense. So the base case would be when the root is null, then the depth is 0. Otherwise, we recursively check left and right.
- Wait, let me test that logic. For example, if the tree is just a single node, then the depth is 1. If the root has a left child with no other nodes, then the depth is 2. Because 1 (root) + 1 (left) = 2. Yeah, that works.
- So the steps are: if the current node is null, return 0. Otherwise, return 1 + the maximum of maxDepth(left) and maxDepth(right). That seems right.
```
class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        } else {
            int leftDepth = maxDepth(root.left);
            int rightDepth = maxDepth(root.right);
            return Math.max(leftDepth, rightDepth) + 1;
        }
    }
}
```
9. [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/) - O(n) & O(n)
- Question: Construct a binary tree from given preorder and inorder traversal arrays. The key insight is to use the properties of these traversals: the first element in the preorder traversal is the root of the tree, and elements to the left of this root in the inorder traversal form the left subtree, while elements to the right form the right subtree. 
- Solution: 1. Hashmap for Inorder Indices and Pre-order traversal
- Wait, but how do we find the root_pos efficiently? Because for each node, we need to search the inorder array between in_start and in_end to find the root's position. If we do a linear search each time, that would take O(n) time per node, leading to O(n^2) time overall. But that's not efficient. So we can optimize this by creating a hashmap that maps the values to their indices in the inorder array. That way, each lookup is O(1).
- Yes. So first, create a hashmap/dictionary where the key is the node value, and the value is its index in the inorder array. Then, for any root value, we can find its position in O(1) time.
- So the steps are:
- Create a hashmap to store the value to index mapping for inorder array.
- Initialize a global variable (or a wrapper) to track the current preIndex starting at 0.
- Define a helper function that takes in_start and in_end as parameters.
- In the helper function:
a. If in_start > in_end, return null.
b. Get the current root value from preorder[preIndex], create a node.
c. Increment preIndex.
d. Find the root's position in the inorder array using the hashmap.
e. Recursively build the left subtree by calling helper(in_start, root_pos -1).
f. Recursively build the right subtree by calling helper(root_pos +1, in_end).
g. Return the root.
```
class Solution {
    private Map<Integer, Integer> inorderMap;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        inorderMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            inorderMap.put(inorder[i], i);
        }
        int[] preIndex = new int[1]; // Using an array to track the preorder index
        return buildTreeHelper(preorder, 0, inorder.length - 1, preIndex);
    }

    private TreeNode buildTreeHelper(int[] preorder, int inStart, int inEnd, int[] preIndex) {
        if (inStart > inEnd) return null;

        int rootVal = preorder[preIndex[0]];
        TreeNode root = new TreeNode(rootVal);
        preIndex[0]++;

        int rootPos = inorderMap.get(rootVal);

        root.left = buildTreeHelper(preorder, inStart, rootPos - 1, preIndex);
        root.right = buildTreeHelper(preorder, rootPos + 1, inEnd, preIndex);

        return root;
    }
}
```
```
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if(preorder.length == 0) {
            return null;
        }

        int r = preorder[0];
        int index = 0;

        for(int i = 0; i < inorder.length; i++) {
            if(r == inorder[i]) {
                index = i;
            }
        }

        TreeNode node = new TreeNode(r);

        node.left = buildTree(Arrays.copyOfRange(preorder, 1, index + 1), Arrays.copyOfRange(inorder, 0, index));
        node.right = buildTree(Arrays.copyOfRange(preorder, index + 1, preorder.length), Arrays.copyOfRange(inorder, index + 1, inorder.length));

        return node;
    }
}
```
10. [Lowest Common Ancestor of a Binary Search Tree (BST)](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/) - O(h) & O(1) h is the height of the tree. In the worst case, this is O(n) for a skewed tree, but O(log n) for a balanced BST.
- Question: Find the lowest common ancestor (LCA) of two given nodes in a binary search tree (BST). The LCA of two nodes p and q is the deepest node that has both p and q as descendants (where we allow a node to be a descendant of itself).
- Solution: 1. BST
- The key insight to solve this problem efficiently is to leverage the properties of a BST. In a BST:
- All nodes in the left subtree of a node have values less than the node's value.
- All nodes in the right subtree of a node have values greater than the node's value.
- Using these properties, we can determine the LCA by comparing the values of the nodes p and q with the current node's value as we traverse the tree starting from the root:
- If both p and q have values greater than the current node's value, their LCA must be in the right subtree.
- If both p and q have values less than the current node's value, their LCA must be in the left subtree.
- If one of the nodes is on one side and the other is on the other side (or one of the nodes is the current node), then the current node is their LCA.
```
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        TreeNode current = root;
        int pVal = p.val;
        int qVal = q.val;
        
        while (current != null) {
            int currVal = current.val;
            
            if (pVal > currVal && qVal > currVal) {
                current = current.right;
            } else if (pVal < currVal && qVal < currVal) {
                current = current.left;
            } else {
                return current;
            }
        }
        
        return null; // This line is theoretically unreachable given problem constraints
    }
}
```
11. [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/description/) - O(n) & O(h)
- Question: Find the maximum path sum in a binary tree where the path can start and end at any node in the tree. The path must contain at least one node and can traverse through the tree in any direction as long as adjacent nodes are connected by edges.
- Solution: 1. Post-order traversal approach
- So, the problem says a path is any sequence of nodes where each adjacent pair is connected by an edge. And the path must contain at least one node. So the maximum path sum could be a single node, or a combination of left and right paths through a root.
- Wait, but how do I calculate this? I remember that for binary tree problems, recursion is often useful. Maybe a post-order traversal approach?
- Let me think. For each node, the maximum path sum that includes that node could be the node's value plus the max of the left and right subtree contributions. But wait, if the left or right subtree contributions are negative, adding them would decrease the sum. So maybe for each node, the maximum possible contribution it can give to its parent is node.val plus the maximum of left or right (but not both) only if those are positive.
- But the actual maximum path could be the node's value plus both left and right. Because that forms a path that goes through the node, using both children. So maybe, for each node, the maximum path sum considering that node as the "root" of the path (i.e., the highest node in the path) is node.val + left + right. But when passing up, the node can only contribute node.val + max(left, right), since you can't take both left and right in a path that continues upwards.
- So the idea is to compute two things for each node:
- The maximum path sum where the node is the highest point (the "root" of the path). This would be the sum of the node's value plus the maximum path from left and right (but if those are negative, we don't take them). Wait, no. Wait, when considering a path through the node, the path can go left then up to node then right. So that path's sum is node.val + leftMax + rightMax, where leftMax is the max path sum starting from left child (if it's positive) and same for right. So for each node, we calculate this value and compare it to the global maximum.
- The maximum path sum that can be contributed to the parent node. This would be node.val plus the maximum of leftMax or rightMax. But again, if leftMax and rightMax are negative, then the best contribution is just the node's value.
- So the algorithm would be:
- Use a helper function that returns the maximum contribution of the subtree (the max path sum starting at the subtree's root, moving downwards). But while doing that, compute the maximum path sum that includes the current node as the root (using both left and right contributions) and update a global variable.
- Wait, but Java doesn't have pass-by-reference for primitives. So maybe I need to have a mutable object to hold the maximum, or use a class variable. Alternatively, have the helper function return two values. But Java can't return multiple values. Hmm. So perhaps the helper function will return the maximum contribution (the max path sum that can be extended upwards), but during its computation, it updates a global or class variable that keeps track of the maximum path sum found so far.
- Yes. So the helper function will:
- For a given node:
- If node is null, return 0 (since we can't contribute anything).
- Recursively get the max path sum contributions from left and right. But if those are negative, we don't take them. So leftMax = max(0, helper(left)), same for rightMax.
- The maximum path sum considering this node as the root would be node.val + leftMax + rightMax. We compute this and compare it with the global max. Update the global max if necessary.
- The value returned by the helper function is node.val + max(leftMax, rightMax). Because the path can't split here; it has to choose either left or right to contribute to the parent path.
```
class Solution {
    private int maxSum;
    
    public int maxPathSum(TreeNode root) {
        maxSum = Integer.MIN_VALUE;
        helper(root); //Need helper because to return ans
        return maxSum;
    }
    
    private int helper(TreeNode node) {
        if (node == null) return 0;
        
        int leftMax = Math.max(0, helper(node.left));
        int rightMax = Math.max(0, helper(node.right));
        
        int currentSum = node.val + leftMax + rightMax;
        maxSum = Math.max(maxSum, currentSum);
        
        return node.val + Math.max(leftMax, rightMax);
    }
}
```

## Graph 
1. Connected and Disconnected Graph - BFS (Using Queue) and DFS (Using Recursion) along with the visited boolean array - Time:O(V+E) Space:O(V)
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
16. [Word Search](https://leetcode.com/problems/word-search/description/) - O(N * 3^L) & O(1) N is the number of cells and L is the length of the word.
- Question: Determine if a given word can be constructed from letters of a 2D grid where each letter must be adjacent (horizontally or vertically) and each cell can be used only once in the construction of the word.
- Solution: 1. Backtracking approach with depth-first search (DFS).
- Let me think. So, the approach here is probably using backtracking. Because for each character in the word, we need to explore all possible directions on the board, and backtrack if the current path doesn't lead to a solution.
- First, I need to iterate over each cell in the board. For each cell, if it matches the first character of the word, then start a DFS search from there. The DFS will check all four possible directions (up, down, left, right) to see if the next character is present. But we have to mark the cell as visited to avoid reusing it in the current path. Oh right, and then unmark it after backtracking so other paths can use it.
- Wait, how do I mark a cell as visited? Maybe by changing the character to a special symbol (like '#' or something) and then reverting it back after the recursive calls. Because modifying the board directly is allowed here, as long as we restore it afterward.
- So the steps would be:
- Iterate through each cell in the board.
- For each cell that matches the first character of the word, start DFS.
- In the DFS function:
a. Check if the current index of the word is beyond the length (base case: return true if index equals word length).
b. Check if the current cell is out of bounds or doesn't match the current character. If so, return false.
c. Mark the current cell as visited by changing its value.
d. Explore all four directions recursively.
e. Restore the cell's original value after backtracking.
f. Return true if any of the directions lead to a successful path.
- Edge cases to consider:
- The board is empty.
- The word is longer than the total number of cells.
- The word has characters not present in the board at all.
- Multiple cells have the first character, need to check all of them.
```
public class Solution {
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || word == null || word.length() == 0) {
            return false;
        }
        
        int m = board.length;
        int n = board[0].length;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == word.charAt(0)) {
                    if (dfs(board, i, j, word, 0)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    private boolean dfs(char[][] board, int i, int j, String word, int index) {
        if (index == word.length()) {
            return true;
        }
        
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != word.charAt(index)) {
            return false;
        }
        
        char temp = board[i][j];
        board[i][j] = '#';  // Mark as visited
        
        boolean found = dfs(board, i + 1, j, word, index + 1)
                     || dfs(board, i - 1, j, word, index + 1)
                     || dfs(board, i, j + 1, word, index + 1)
                     || dfs(board, i, j - 1, word, index + 1);
        
        board[i][j] = temp;  // Backtrack
        
        return found;
    }
}
```
17. [Course Schedule](https://leetcode.com/problems/course-schedule/description/) - O(V+E) & O(V+E) V is the number of courses and E is the number of prerequisites.
- Question: Determine if all given courses can be completed given their prerequisites. This is equivalent to checking if the directed graph formed by the courses and their prerequisites is a Directed Acyclic Graph (DAG). If there is a cycle in the graph, it is impossible to complete all courses.
- Solution: 1. Topological sort - Kahn's algorithm (BFS using in-degree and a queue) 2. DFS to detect cycles.
- Let me think. The input is the number of courses and a list of prerequisite pairs. Each pair [a, b] means you have to take course b before a. So this forms a directed edge from b to a. The question is whether this graph has a topological order, which is possible only if there are no cycles.
- So the approach here is to perform a topological sort. There are two common ways to do this: Kahn's algorithm (using in-degree and a queue) or using DFS to detect cycles.
- Hmm. Let me consider Kahn's algorithm first. Kahn's algorithm works by repeatedly removing nodes with no incoming edges. If all nodes are removed, the graph has no cycle. Otherwise, there's a cycle.
- So the steps would be:
- Build an adjacency list to represent the graph.
- Calculate the in-degree for each node.
- Initialize a queue with all nodes having in-degree zero.
- While the queue is not empty:
a. Dequeue a node.
b. Reduce the in-degree of its neighbors by one.
c. If any neighbor's in-degree becomes zero, enqueue it.
- If the number of processed nodes equals the total nodes, then no cycle exists.
- Alternatively, using DFS to detect cycles. For each node, we can track visited nodes and the recursion stack. If during traversal we encounter a node that's already in the recursion stack, there's a cycle.
- But for topological sorting, the DFS method involves visiting nodes and adding them to a stack once all their dependencies are processed. However, for cycle detection, we need to track the current path.
- Wait, but in this problem, we just need to check if a topological sort exists, which is equivalent to checking if the graph is a DAG (Directed Acyclic Graph). So either approach works.
- Let me think about implementing Kahn's algorithm first.
- First, build the adjacency list. For each prerequisite [a, b], since b is a prerequisite for a, the edge is from b to a. So the adjacency list entry for b will include a. Then, the in-degree for a is incremented.
- Once the adjacency list and in-degree array are built, we start with nodes (courses) that have zero in-degree. If there are no such nodes initially, then there's a cycle.
- We process each node, decrementing the in-degree of its neighbors. Each time a neighbor's in-degree becomes zero, it's added to the queue. The count of processed nodes is tracked. If this count equals the total number of courses, then it's possible to finish.
- So let's outline the steps for code:
- Create adjacency list: a list of lists. For each course, list the courses that depend on it.
- Create inDegree array: for each course a, inDegree[a] is the number of prerequisites.
- Initialize a queue with all courses where inDegree is zero.
- Initialize a counter for processed nodes.
- While queue is not empty:
a. Dequeue course u.
b. Increment counter.
c. For each neighbor v of u:
i. Decrement inDegree[v] by 1.
ii. If inDegree[v] becomes zero, enqueue v.
- If counter equals numCourses, return true. Else, false.
```
public class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<Integer>[] adj = new ArrayList[numCourses];
        for (int i = 0; i < numCourses; i++) {
            adj[i] = new ArrayList<>();
        }
        int[] inDegree = new int[numCourses];
        
        for (int[] prereq : prerequisites) {
            int course = prereq[0];
            int prerequisite = prereq[1];
            adj[prerequisite].add(course);
            inDegree[course]++;
        }
        
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) { //add node those don't have incoming edges and remove then in bfs
                queue.offer(i);
            }
        }
        
        int count = 0;
        while (!queue.isEmpty()) {
            int current = queue.poll();
            count++;
            for (int neighbor : adj[current]) { //remove neighbor don't have incoming edges by adding in the queue
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        
        return count == numCourses;
    }
}
```
18. [Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/description/) - O(n+e) & O(n+e) e is the number of edges
- Question: Determine if a given undirected graph is a valid tree, we need to check two conditions: the graph must be connected and it must have exactly n-1 edges (where n is the number of nodes). This approach ensures the graph is acyclic and fully connected, which are the defining characteristics of a tree.
- Solution: 1. BFS or DFS
- First, I remember that a tree is a connected graph with no cycles. So to validate if the graph is a tree, it must satisfy two conditions:
- It must be connected, meaning all nodes are reachable from any starting node.
- It must have exactly n-1 edges. Because a tree with n nodes has exactly n-1 edges. If there are more than n-1 edges, there's definitely a cycle. If fewer, it's not connected.
- Wait, but maybe even if the number of edges is exactly n-1, there could still be a cycle and disconnected components. Hmm, no. Wait, if the graph has n nodes and n-1 edges, but it's disconnected, then each component would have to be a tree. But the total number of edges would be (sum of (k_i -1) for each component with k_i nodes) which would be less than n-1 if there's more than one component. Wait, that's not possible. Let me think: suppose the graph is split into two components. Then the total edges would be (a-1) + (b-1) = a + b -2. But a + b is n, so edges are n -2. But the given edges are n-1, so that's impossible. So, if the graph has exactly n-1 edges and is acyclic, then it's connected. Wait, maybe not. Wait, if the graph has n-1 edges but is cyclic, then it's not a tree. So the two conditions are: exactly n-1 edges and connected. Or maybe the other way: connected and acyclic.
- Alternatively, a tree is a connected acyclic graph. So the conditions are: the graph is connected and has no cycles. But how to check that?
- So the steps could be:
- Check if the number of edges is exactly n-1. If not, return false (since a tree must have n-1 edges).
- Check if the graph is connected. If it is, then it's a tree.
- But wait, suppose there are n-1 edges but the graph is disconnected. Then according to my previous thought, that's impossible. Let me see: if the graph is disconnected into two components, then each component must have at least a tree (so edges are (k-1) + (m-1) where k + m = n). So total edges are k-1 + m-1 = n -2. So if the graph has n-1 edges and is disconnected, that would require edges to be n-2, which contradicts. So, if the number of edges is exactly n-1, then the graph must be connected and acyclic. Wait, but that's not necessarily true. For example, imagine a graph with 4 nodes and 3 edges. If the edges form a cycle (like 0-1, 1-2, 2-0), but that would have 3 edges and 3 nodes, but that's a cycle. Wait, but 4 nodes and 3 edges would require another component. So maybe the initial check is that if the number of edges is not n-1, return false. Then, check if the graph is connected. If both conditions are met, then it's a tree.
- So the approach would be:
- First, check if edges.length != n-1, return false.
- Then, check if the graph is connected. If yes, return true; else, false.
- Because if there are n-1 edges and the graph is connected, then it's a tree (since connected and no cycles). Wait, but how do we ensure there are no cycles? Because if the graph is connected and has n-1 edges, then it must be a tree. Because a tree is a connected acyclic graph. So a connected graph with n nodes and n-1 edges is a tree. Because any connected graph with n nodes and n-1 edges is a tree. So cycles can't exist in such a graph. Because adding an edge to a tree creates a cycle, and removing an edge disconnects it. So the two conditions (edges ==n-1 and connected) are sufficient.
- So, the steps are:
- If the number of edges is not equal to n-1, return false.
- Check if the graph is connected. If yes, return true; else, return false.
- So now, how to check if the graph is connected. Since the graph is undirected, we can perform a BFS or DFS starting from one node and see if all nodes are visited.
- So the plan is:
- Check if edges.length != n-1: return false.
- Build adjacency list.
- Do BFS/DFS starting from node 0 (assuming nodes are 0-based), and count visited nodes.
- If the count is n, then connected, else not.
- Edge cases:
- n=1: then edges must be empty. Because 0 nodes have 0 edges. Wait, n=1: nodes are [0]. So edges must be empty. So if n=1, edges must be empty. So in code, need to handle that.
```
class Solution {
    public boolean validTree(int n, int[][] edges) { //n=4, edges = [[0,1],[1,2],[2,3],[3,0]]. edges.length=4 each row having only two elements
        // Check if the number of edges is exactly n-1
        if (edges.length != n - 1) return false;
        
        // Handle the case when n is 1 (only possible with 0 edges)
        if (n == 0) return false;
        if (n == 1) return true;
        
        // Build adjacency list
        List<List<Integer>> adjList = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            adjList.add(new ArrayList<>());
        }
        for (int[] edge : edges) {
            adjList.get(edge[0]).add(edge[1]);
            adjList.get(edge[1]).add(edge[0]);
        }
        
        // Check connectivity using BFS
        boolean[] visited = new boolean[n];
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(0);
        visited[0] = true;
        int count = 1;
        
        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int neighbor : adjList.get(node)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    count++;
                    queue.offer(neighbor);
                }
            }
        }
        
        return count == n;
    }
}
```
19. [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/description/) - O(nm+c) & O(c) m is the average word length, c is the total number of unique characters.
- Question: Determine the order of characters in an alien language given a list of words sorted according to that language's rules. If the input is invalid due to conflicting orders or incorrect word structures, we should return an empty string.
- Solution: 1. Topological Sort - BFS
- Hmm, let me remember what the problem is about. From what I recall, the problem is that given a list of words sorted according to the alien language's dictionary order, I need to determine the order of the characters in that alien language. If the words are not in a valid order (like a conflicting order where a word is both before and after another word inconsistently), then I should return an empty string. Also, if there are multiple valid orders, return any one of them. Oh right, and if the input is invalid, like a word is followed by a shorter word that is a prefix of the longer one, that's invalid too. For example, if we have "app" followed by "apple", that's invalid because the shorter comes first but is a prefix. But if the words are ["apple", "app"], that's invalid because according to the alien dictionary, the first word is longer but the second should be considered smaller. Wait, actually, the rule is that if two words are such that the first is longer than the second and the second is a prefix of the first, then that's invalid. Because in a correct dictionary, shorter prefixes come first. So in such a case, the order is invalid, and we should return empty string.
- So the approach here is to build a graph of characters and their dependencies, then perform a topological sort to determine the order. Let's break it down.
- First, I need to compare each pair of consecutive words in the list. For each pair, find the first differing character between them. The character in the first word comes before the character in the second word. So, for example, if we have "wrt" and "wrf", the first differing character is 't' vs 'f', so 't' comes before 'f'. So we can create an edge from 't' to 'f' indicating that 't' must come before 'f' in the order.
- But before doing that, I need to check if the words are in a valid order. Like, if the first word is longer than the second and the second is a prefix of the first, that's invalid. For example, "abcd" followed by "abc" is invalid. Because "abc" is a prefix of "abcd", but since it's shorter, it should come first. So in such a case, the input is invalid, return empty string.
- So first, I need to process all consecutive word pairs and:
- Check for the invalid case where the first word is longer than the second and the second is a prefix of the first. If any such pair exists, return empty string.
- For valid pairs, find the first differing character between the two words. Add an edge from the character in the first word to the character in the second word. Also, make sure to collect all unique characters present in the words, in case some characters don't have any edges (they can be placed anywhere in the order, but need to be included in the result).
- Once the graph is built, perform a topological sort. The topological order will give the correct alien dictionary order. But since there might be multiple valid orders, any of them is acceptable. However, if during the topological sort, a cycle is detected (meaning there's a contradiction in the order), then return empty string.
- So steps in code:
- Build the graph and in-degree map.
- Process each consecutive word pair to add edges and check for validity.
- Collect all unique characters.
- Perform topological sort using Kahn's algorithm (BFS-based approach).
- Check if the sorted order contains all characters. If yes, return the order; else, return empty string.
- Let me think about how to implement this.
- First, collect all unique characters. For each word in the words array, iterate through each character and add to a set. Then, the set can be used to get all unique characters.
- But wait, some characters may not have any edges (they are not compared with others), so they can be placed anywhere in the order. However, during the topological sort, they need to be included. So, during the topological sort, after processing all dependencies, any remaining nodes with zero in-degree can be added in any order.
- Now, processing consecutive word pairs.
- For words[i] and words[i+1], compare each character until the first differing one. Let's say at position j, words[i][j] != words[i+1][j]. Then we add an edge from words[i][j] to words[i+1][j], and break the loop. Also, need to check if j is beyond the length of the shorter word. Wait, no. For example, if words[i] is "abc", words[i+1] is "abde". The first two characters are the same, third character is 'c' vs 'd'. So 'c' comes before 'd'. So edge from c to d. But if words[i] is "abc" and words[i+1] is "ab", then since the first two characters are the same, but words[i+1] is shorter, then it's invalid. Because "ab" should come before "abc" in a valid dictionary. So this case needs to be checked.
- So for each pair of consecutive words, we need to compare each character up to the minimum of the two lengths. If all characters are equal up to the minimum length, and the first word is longer than the second, then it's invalid.
```
public class AlienDictionary {
    public String alienOrder(String[] words) {
        Map<Character, Set<Character>> graph = new HashMap<>();
        Map<Character, Integer> inDegree = new HashMap<>();
        Set<Character> uniqueChars = new HashSet<>();

        // Collect all unique characters
        for (String word : words) {
            for (char c : word.toCharArray()) {
                uniqueChars.add(c);
            }
        }

        // Initialize graph and in-degree for each character
        for (char c : uniqueChars) {
            graph.put(c, new HashSet<>());
            inDegree.put(c, 0);
        }

        // Build the graph and check for invalid cases
        for (int i = 0; i < words.length - 1; i++) {
            String word1 = words[i];
            String word2 = words[i + 1];
            int len1 = word1.length();
            int len2 = word2.length();
            int minLen = Math.min(len1, len2);
            boolean foundDiff = false;
            for (int j = 0; j < minLen; j++) {
                char c1 = word1.charAt(j);
                char c2 = word2.charAt(j);
                if (c1 != c2) {
                    if (!graph.get(c1).contains(c2)) {
                        graph.get(c1).add(c2);
                        inDegree.put(c2, inDegree.get(c2) + 1);
                    }
                    foundDiff = true;
                    break;
                }
            }
            // Check if the second word is a prefix of the first and shorter
            if (!foundDiff && len1 > len2) {
                return "";
            }
        }

        // Kahn's algorithm for topological sort
        Queue<Character> queue = new LinkedList<>();
        for (char c : inDegree.keySet()) {
            if (inDegree.get(c) == 0) {
                queue.offer(c);
            }
        }

        StringBuilder result = new StringBuilder();
        while (!queue.isEmpty()) {
            char c = queue.poll();
            result.append(c);
            for (char neighbor : graph.get(c)) {
                inDegree.put(neighbor, inDegree.get(neighbor) - 1);
                if (inDegree.get(neighbor) == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        // Check if all characters were included (no cycles)
        if (result.length() != uniqueChars.size()) {
            return "";
        }

        return result.toString();
    }
}
```
20. [Number of connected components in an undirected graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/) - O(E * α(n)) & O(n) where α is the inverse Ackermann function, which grows very slowly
- Question: Finding the number of connected components in an undirected graph
- Solution: 1. Union-Find (Disjoint Set Union) algorithm (Given list of edges) 2. DFS/BFS (Given  adjacency list) - O(V + E) & O(V + E)
-  Let me start by understanding what a connected component is. In an undirected graph, a connected component is a subgraph where every pair of vertices is connected via a path, and no edges exist connecting to vertices outside this subgraph. So the task is to count how many such components exist in the given graph.
- First, I should think about how the graph is represented. The problem probably gives the graph as a set of nodes and edges. Let's say the input is n nodes numbered from 0 to n-1, and a list of edges. So each edge is a pair of nodes. Since the graph is undirected, each edge connects two nodes both ways.
- Now, the approach to find connected components. The standard methods are either Depth-First Search (DFS) or Breadth-First Search (BFS), or using Union-Find (Disjoint Set Union, DSU). Each method has its pros and cons.
- Let me consider each approach:
- **DFS/BFS**: For each node that hasn't been visited yet, perform a DFS or BFS to visit all reachable nodes, marking them as visited. Each time I start a new search, that's a new component. The count of such starts gives the number of components.
- **Union-Find**: Create a parent array where each node starts as its own parent. For each edge, union the two nodes. After processing all edges, the number of distinct parents (roots) represents the number of components. But wait, in union-find, after all unions, the number of roots (nodes where parent[i] == i) gives the count of components. So that's another way.
- Which method is better? It depends on the constraints. If the graph is given as an adjacency list, DFS/BFS is straightforward. If it's given as a list of edges, Union-Find might be efficient, especially with path compression and union by rank. But let's see the problem statement again.
- Assuming the input is the number of nodes and the list of edges, let's proceed.
- So, let's outline the steps for each approach.
- **Approach 1: DFS or BFS**
- Create an adjacency list from the edges.
- Keep a visited array to track visited nodes.
- Iterate through each node, and if not visited, perform DFS/BFS, mark all reachable nodes as visited, and increment the component count.
- **Approach 2: Union-Find**
- Initialize parent array where parent[i] = i.
- For each edge, union the two nodes.
- After processing all edges, count the number of nodes where parent[i] == i. That's the number of components.
- But wait, what if there are nodes that are not connected to any others? Like, if the graph has 5 nodes but only edges between 0-1 and 2-3, then nodes 4 is a component by itself. The union-find approach will correctly count it because 4's parent is itself.
- So, for Union-Find, after all unions, the number of roots (parent[i] == i) is the answer.
- So both approaches should work. Let's think about time and space.
- For DFS/BFS:
- Time: O(V + E) since each node and edge is processed once.
- Space: O(V + E) for adjacency list and visited array.
- For Union-Find with path compression and union by rank:
- Time: O(E * α(V)), where α is the inverse Ackermann function (very slow growing).
- Space: O(V) for parent and rank arrays.
- If the number of nodes is large, Union-Find is more efficient. But for the problem here, either approach should be manageable unless there's a very high constraint.
- Now, let's think about implementing each approach.
- First, let's consider the input. The problem may be given as:
- int n: the number of nodes.
- int[][] edges: array of edges.
- So, for example, n=5, edges = [[0,1], [1,2], [3,4]] would result in 2 components: 0-1-2 and 3-4, plus node 4 if it's 5 nodes? The problem states that nodes are numbered from 0 to n-1. So if n=5, nodes are 0,1,2,3,4. So in the example above, the components are 0-1-2, 3-4. So two components. But then, what about node 4? No, in the example, 3 and 4 are connected. So the total components are two. So the code must account for all nodes. So if a node is not present in any edge, it's a component by itself.
- For example, if n=5 and edges are empty, then each node is its own component, so count is 5.
- So in the problem, the number of nodes is given, and edges. So when building the adjacency list, nodes with no edges are isolated, and each contributes to the component count.
- So, for the DFS/BFS approach:
- Create adjacency list for all nodes. For each node, even if it has no edges, it's present as a key with an empty list.
- But when building the adjacency list, how to handle nodes with no edges. For example, if we loop through all nodes from 0 to n-1 and create an entry for each.
- Now, the Union-Find approach.
- Implement the DSU with path compression and union by rank.
- The steps are:
- Initialize parent array where parent[i] = i.
- Initialize rank array (for union by rank) with all zeros.
- For each edge, perform a union operation on the two nodes.
- After all unions, count the number of elements where parent[i] == i (the roots). Each root represents a component.
- But wait, the initial count is n (each node is its own parent). Each union reduces the count by 1 if the two nodes were in different sets.
- So alternatively, during the union steps, we can track the count.
- But perhaps it's easier to process all edges and then count the number of roots.
- But here's the problem: some nodes may not be part of any edge. For example, if a node is isolated (no edges), then it's its own component. So the union operations do not touch it, so parent[i] remains i. So when we count the number of roots, it's correct.
```
class Solution {
    public int countComponents(int n, int[][] edges) {
        int[] parent = new int[n];
        int[] rank = new int[n];
        
        // Initialize each node as its own parent with rank 0
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            rank[i] = 0;
        }
        
        // Union each pair of nodes in the edges
        for (int[] edge : edges) {
            union(parent, rank, edge[0], edge[1]);
        }
        
        // Count the number of unique roots (components)
        int components = 0;
        for (int i = 0; i < n; i++) {
            if (find(parent, i) == i) {
                components++;
            }
        }
        return components;
    }
    
    private int find(int[] parent, int x) {
        if (parent[x] != x) {
            parent[x] = find(parent, parent[x]); // Path compression - Each compression moving towards the common parent, it'll update both parent[x] and x
        }
        return parent[x];
    }
    
    private void union(int[] parent, int[] rank, int x, int y) {
        int rootX = find(parent, x);
        int rootY = find(parent, y);
        
        if (rootX == rootY) {
            return; // Already in the same set
        }
        
        // Union by rank: attach smaller rank tree under root of higher rank tree
        if (rank[rootX] < rank[rootY]) { //Updating parent based on rank. parent[lower_rank] = higher_rank root
            parent[rootX] = rootY;
        } else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else {
            parent[rootY] = rootX; //Both rank is same then parent[2nd_rank]=1st_rank and rank[1st_rank]++;
            rank[rootX]++;
        }
    }
}
```

