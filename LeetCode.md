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
