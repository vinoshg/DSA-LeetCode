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

## Kadane's Algorithm - O(n)
1. Maximum Subarray - Array has -ve elemnets as well
- Solution: 1. for loop currentSum=Math.max(currentSum, currentSum+num) and compare with maxSum 2. Nested loop

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

## Sliding Window - O(n)
- Question: 1. Fixed Sliding window (Find Subarray/Substring of a fixed length) 2. Dynamic Sliding window (Longest/Shortest Subarray or Substring that satisfies the condition) - (e.g., max sum, longest substring with unique characters)
1. Find the Maximum Subarray of length k / Max avg sum of Subarray (avg=max/array.length) - Fixed Sliding window
- Solution: 1. windowSum += array[i] - array[i-k] 2. Nested loops
2. Find length of the longest substring without repeating character - Dynamic Sliding window
- Solution: 1. Two pointers with HashMap - left=0,maxLength=0 for right=0 to str.length() if(map.containsKey(ch) && map.get(ch) >= left) left=map.get(ch)+1 after map.put(ch, right) maxLength=Math.max(maxLength, right-left+1) 2. Two pointers with HashSet with left=0 and for right=0 to array.length while(set.contains(str.charAt(right)) set.remove(str.charAt(left)) left++ after while loop set.add(str.charAt(right)) sum=Math.max(sum, right-left+1) / Freq array with sliding window (Less number of char and only lower case, array size 128)

## Two Pointers - O(n) and O(1) 
- Question: Applicable to Linear Data Structure Arrays, String, LinkedList - Converging pointers | Parallel pointers | Trigger based pointers (Usually Apply for Sorted Array/LL) | Expand Around Center
1. Converging pointers - Two pointers start at 0th and array.length-1 and converging together
- Question: 1. Palindrome (Applicable for both Subarray/Substrings)
2. Parallel pointers
- Question: 1. The right pointer is used to get new info and the left is used to track. Both start from 0/1
3. Trigger based pointers
- Question: 1. left pointer move only after right pointer reaches particular position
4. Move all zeros to the end of the array
- Solution: 1. Trigger-based pointers (left, right) - for right=0 to array.length if(array[right]!=0) then swap, left++ 2. Take result array fill with 0 and traverse the array, add non-zero element to it
5. Container with most water - Array of vertical lines, find two lines that can trap most water
- Solution: 1. left=0,right=array.length-1 while(left<=right) area=width*minHeight; if(array[left]<array[right]) left++ else right--         2. Nested loops with find all possible area - i=0 to array.length and j=i+1 to array.length with area
6. Remove Duplicates from Sorted Array - Trigger based pointers
- Question: Remove duplicates from Sorted array and return unique elemenets
- Solution: 1. Two pointers - k=1 for i=1 to array.length if(array[i]!=array[i-1]) swap array[k]=array[i] k++ after loop return k. If both elements are same continue 2. HashSet and New Array
7. Two Sum II - Input Array Is Sorted (Keep in mind array declaration with two elements - return new int[]{left+1,right+1})
- Question: Given an array return the two indices (start from 1,2,3), sum of two numbers = target
- Solution: 1. Two pointers(Converging pointers) left=0,right=array.length-1 while(left<right) sum=array[left]+array[right] if(sum==target) return new int[]{left+1,right+1} else if(sum<target) left++ else right--; after while retrun new int[]{-1,-1} 2. Nested loops i=0 to array.length; j=i+1 to array.length
8. [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/description/) - O(n^2) & O(1)
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
1. Merge Intervals
- Question: intervals[][] (2D Array)
- Solution: 1. Sort an array based on start of the interval: Arrays.sort(array, (a,b) -> Integer.comapre(a[0],b[0]) / Arrays.sort(array, (a,b) -> a[0] - b[0]); Take a list and add 0th element. for i=1 to array.length if(current[0] <= last[1]) last[1] = Math.max(last[1], current[1]) else list.add(array[i]); after loop return list.toArray(new int[list.size()][])
2. Insert Interval - O(n) if array's sorted & O(n)
- Solution: 1. Sort an array with three cases a) For intervals that come before the new interval and don't overlap, I can add them directly
  b) Merging Overlapping Intervals and Inserting the Merged Interval c) Add the remaining intervals that don't overlap after the merged part
  Initialize newStart=newInterval[0],newEnd=newInterval[1],i=0,n=intervals.length
  
  a) while(i<n && intervals[i][1] < newStart) result.add(intervals[i]) i++; All intervals with end < newInterval's start
  
  b) while(i<n && intervals[i][0] <= newEnd) newStart=Min(two start),newEnd=Max(two end); After while result.add(new int[]{newStart,newEnd})
     interval start <= new/merged interval end along with comply(auto) by interval end >= new/merged interval start
  
  c) while(i<n) / while(i<n && intervals[i][0] > newEnd) result.add(intervals[i]) i++; add all intervals that come after the merged 
     interval (start > merged interval's end)

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
        if ((totalSum + target) % 2 != 0 || (totalSum + target) < 0) {
            return 0;
        }
        
        int requiredSum = (totalSum + target) / 2;
        int[] dp = new int[requiredSum + 1];
        dp[0] = 1; // Base case: one way to get sum 0 (using no elements)
        
        for (int num : nums) {
            for (int j = requiredSum; j >= num; j--) {
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
4. Climbing Stairs - O(n) & O(1)
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
13. Number of Islands - O(nm) and O(nm)
- Solution: 1. Nested loops with DFS on all four directions making it grid[i][j] = 0 when I found the island 2. BFS
