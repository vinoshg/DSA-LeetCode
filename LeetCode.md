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
- Solution: 1. for loop currentSum=Math.max(currentSum, surrentSum+num) and compare with maxSum 2. Nested loop

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
- Question: 1. Fixed Sliding window (Find Subarray/Substring of a fixed length) 2. Dynamic Sliding window (Longest/Shortest Subarray or Substring that satisfies the condition)
1. Find the Maximum Subarray of length k / Max avg sum of Subarray (avg=max/array.length) - Fixed Sliding window
- Solution: 1. windowSum += array[i] - array[i-k] 2. Nested loops
2. Find length of the longest substring without repeating character - Dynamic Sliding window
- Solution: 1. HashSet with left=0 and for right=0 to array.length / Freq array with sliding window (Less number of char and only lower case, array size 128)

## Two Pointers - O(n) and O(1)
- Question: Applicable to Linear Data Structure Arrays, String, LinkedList
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

## Greedy Algorithm
1. Longest Consecutive Sequence
- Solution: 1. HashSet fill with array element. For num if (!set.contains(num - 1)) currentNum=num while (set.contains(currentNum + 1)

## Dynamic Programming
1. Longest Common Subsequence - O(mn)
- Solution: 1. dp[m+1][n+1] Nested loops i,j=1 if (text1.charAt(i - 1) == text2.charAt(j - 1)) dp[i][j] = dp[i - 1][j - 1] + 1; else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]); after return dp[m][n];

## Linked List
- Fast and Slow pointers - O(n) & O(1)
1. Linked List Cycle
- Solution: 1. Fast and Slow pointers while(fast != null && fast.next != null) if(fast == slow) return true after while return false;     
     2. HashSet - while(current != null) if(set.contains(current)) return true; set.add(current) current=current.next; - O(n) & O(n)
2. Middle of the Linked List
- Solution: 1. Fast and Slow pointers, return slow; - Single parse 2. Find the length of the LL count=0 and Apply For loop again i=0 to count/2, return current - Two times for loop - O(n) & O(1)
3. Happy Number
- Question: Given num 14 = 1^2+4^2 = 17 ... If it ends with '1' it means Happy, else the loop again 17 is not a happy number
- Solution: 1. Fast and Slow pointers - slow=n,fast=getNext(n) while(fast!=1 && slow != fast) slow,fast increment after while return fast==1
  2. HashSet - while(n>0) findSquare and check in the if set.contains(square) return true otherwise set.add(squre)  after while return false
    
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
