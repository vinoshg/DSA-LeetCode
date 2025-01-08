## Trie
1. Implement Trie (Prefix Tree) - insert, search, startsWithPrefix 
2. Word Break 
- Question: Break the word into multiple parts and check whether all parts are present in the dictionary
- Solution: Reverse dp[s.length]; dp[s.length()]=true; for(i=s.length()-1 to 0) and for(w : wordDict) if(i+w.length() <= s.length() && s.startWith(w, i)) then dp[i] = dp[i+w.length()) and if(dp[i]) then break; Outer the loop return dp[0] as ans
3. Word Break II 
4. Count Unique Substrings / Count Trie Nodes 
5. Longest word with all prefixes / Longest Word in Dictionary 
