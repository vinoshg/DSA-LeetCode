## HashMap
```
Map<Integer, Integer> frequencyMap = new HashMap<>();
frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
// Add entries to the heap, maintaining the size at k
for (Map.Entry<Integer, Integer> entry : frequencyMap.entrySet()) {
  heap.offer(entry);
  if (heap.size() > k) {
      heap.poll();
   }
}
Map<String, List<String>> map = new HashMap<>();
if (!map.containsKey(key)) {
  map.put(key, new ArrayList<>());
}
map.get(key).add(s);
```
## Heap / PriorityQueue
```
// Create a min-heap based on the frequency(HashMap Value) of the elements
PriorityQueue<Map.Entry<Integer, Integer>> heap = new PriorityQueue<>(
  (a, b) -> a.getValue() - b.getValue()
);
heap.poll().getKey()
//Methods
heap.isEmpty()
heap.poll()
heap.peek()
heap.size()
heap.offer(interval[1])
//Linked List Node in the Heap, Sorting
PriorityQueue<ListNode> heap = new PriorityQueue<>((a,b) -> a.val - b.val);
```
## Queue
```
Queue<TreeNode> queue = new LinkedList<>();
queue.add(root);
queue.poll();
```
## Array
```
// Sort intervals based on their start times
Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
Arrays.sort(candidates); // Sort the candidates to enable pruning
int[] answer = new int[n];
Arrays.fill(answer, 1);
int[][] dp = new int[m + 1][n + 1];
char[] chars = s.toCharArray(); //Char Array
for (int[] interval : intervals) { }
```
## List
```
List<int[]> merged = new ArrayList<>();
merged.toArray(new int[merged.size()][]); //2D Array
List<Integer> currentLevel = new ArrayList<>();
currentLevel.add(node.val);
```
## Set
```
Set<Integer> seen = new HashSet<>();
if (!seen.add(num)) { // if(seen.conatins(num); seen.add(num) return false if num exists in a set
  return true;
}
```
## Deque and Stack
```
//Deque
Deque<Character> stack = new ArrayDeque<>();
stack.pop()
stack.isEmpty()
stack.push(c)
Deque<Pair> deque = new ArrayDeque<>();
deque.peekLast().c
deque.peekLast()
deque.isEmpty()
deque.removeLast();
deque.addLast(new Pair(c, 1));
//Stack
Stack<TreeNode> stack = new Stack<>();
stack.isEmpty()
stack.push(current);
stack.pop();
```
## String and StringBuilder
```
StringBuilder sb = new StringBuilder();
sb.append(p.c);
sb.toString();
sb.length()
sb.charAt(sb.length() - 1) == c
sb.deleteCharAt(sb.length() - 1);
for (char c : s.toCharArray()) { }
```
