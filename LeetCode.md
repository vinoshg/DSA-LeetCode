## Trie
1. Implement Trie (Prefix Tree) - insert, search, startsWithPrefix 
2. Word Break 
- Question: Break the word into multiple parts and check whether all parts are present in the dictionary
- Solution: Reverse DP
```
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        boolean[] dp = new boolean[s.length() + 1];
        dp[s.length()] = true;

        for (int i = s.length() - 1; i >= 0; i--) {
            for (String w : wordDict) {
                if (i + w.length() <= s.length() && s.startsWith(w, i)) {
                    dp[i] = dp[i + w.length()];
                }
                if (dp[i]) {
                    break;
                }
            }
        }
        return dp[0];
    }
}
```
3. Word Break II 
4. Count Unique Substrings / Count Trie Nodes 
5. Longest word with all prefixes / Longest Word in Dictionary 
