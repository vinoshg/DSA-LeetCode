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
## Graph
1. Connected and Disconnected Graph - BFS (Using Queue) and DFS (Using Recursion) along with the visited boolean array
2. All Paths From Source to Target
- Solution: 1. DFS with Backtracking (As per the question nodes are to be visited multiple times, so the visited array is not required and backtracking will handle this list.remove(list.size() - 1))
3. Cycle Detection - Directed Graph
- Solution: 1. DFS recursion with the boolean stack and visited array
4. Cycle Detection - Undirected Graph
- Solution: 1. DFS recursion with parent element and visited array
5. Topological Sort - DAG
- Question: Directed Acyclic Graph(DAG) with no cycle (Not for non DAGs). It's a linear order of vertices u->v
- Solution: 1. 
