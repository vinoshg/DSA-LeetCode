## Binary Search 
1. Binary Search - O(logn) and O(1)
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

## Trie
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
