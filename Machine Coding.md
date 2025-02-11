## Design a Rate Limiter for API Requests
- Solution: We can use a fixed window algorithm that tracks the number of requests per client within a specified time window.
- First, what is a rate limiter? It's a mechanism that restricts the number of API requests a client can make within a certain time window. If the client exceeds the limit, further requests are blocked or throttled until the window resets.
- So, the key components are tracking the number of requests per client and enforcing the limit. Now, which algorithm should I use? Common ones are Token Bucket, Leaky Bucket, Fixed Window, Sliding Window, or Sliding Log. Since the problem is for a machine coding interview, maybe the Fixed Window or Sliding Window would be easier to implement quickly.
- Fixed Window is simpler: for each client, track the count in a time window (e.g., 1 minute). If the count exceeds the limit, reject. But it can allow bursts at the window boundaries. Sliding Window is more accurate but a bit more complex. Maybe start with Fixed Window for simplicity.
- So, the plan is: for each client (identified by some key, like API key or IP), maintain a count and the current window's start time. When a request comes in, check if the current time is within the current window. If yes, increment count. If count exceeds limit, reject. If the time is beyond the window, reset the count and start a new window.
- But how to handle the storage? Need a data structure to map client keys to their current count and window start. A ConcurrentHashMap would be good for thread safety in Java, as multiple threads may be handling requests concurrently.
- Also, need to consider thread safety when updating the counts and windows. So, perhaps using atomic operations or synchronized blocks. For example, when checking and updating the window, we need to make sure that only one thread does that per client at a time. But using synchronized on the client's key might be tricky. Alternatively, use a thread-safe structure and computeIfAbsent with a lock per client, but that's more complex.
- Wait, maybe using the putIfAbsent method with a wrapper object that holds the count and window start. Each entry in the map would be an object that holds the current count and the window's start time. When a request comes in:
- Get the client's entry from the map.
- Check if the current time is within the window (current time < window start + window duration).
- If yes, check if count < limit. If yes, increment count and allow. If not, reject.
- If the current time is outside the window, reset the count to 1, update the window start to the current time, and allow.
- But this can have race conditions when multiple threads are updating the same client's entry. So to make this thread-safe, perhaps each entry (like a RateLimiterRecord) should be an atomic reference, or use synchronized blocks when updating.
- Alternatively, use a lock per client. But that's not efficient. Alternatively, use the compute method of ConcurrentHashMap, which allows atomic updates.
- Fixed Window Algorithm: This approach divides time into fixed intervals (windows). Each client can make a limited number of requests per window. If the window expires, the counter resets.
- ConcurrentHashMap: Stores client identifiers and their corresponding request records to handle concurrent access efficiently.
- Synchronization: Uses per-client synchronization to ensure atomic updates to request counts and window start times, preventing race conditions.

```
import java.util.concurrent.ConcurrentHashMap;

public class RateLimiter {
    private final int maxRequests;
    private final long windowSizeInMillis;
    private final ConcurrentHashMap<String, RateLimitRecord> rateLimitRecords;

    public RateLimiter(int maxRequests, long windowSizeInSeconds) {
        this.maxRequests = maxRequests;
        this.windowSizeInMillis = windowSizeInSeconds * 1000;
        this.rateLimitRecords = new ConcurrentHashMap<>();
    }

    public boolean allowRequest(String clientId) {
        RateLimitRecord record = rateLimitRecords.computeIfAbsent(clientId, k ->
                new RateLimitRecord(0, 0));

        synchronized (record) {
            long currentTime = System.currentTimeMillis();
            if (record.getWindowStart() == 0 || currentTime >= record.getWindowStart() + windowSizeInMillis) {
                // Reset the window for the current client
                record.setCount(1);
                record.setWindowStart(currentTime);
                return true;
            } else {
                if (record.getCount() < maxRequests) {
                    record.setCount(record.getCount() + 1);
                    return true;
                } else {
                    return false;
                }
            }
        }
    }

    private static class RateLimitRecord {
        private int count;
        private long windowStart;

        public RateLimitRecord(int count, long windowStart) {
            this.count = count;
            this.windowStart = windowStart;
        }

        public int getCount() {
            return count;
        }

        public void setCount(int count) {
            this.count = count;
        }

        public long getWindowStart() {
            return windowStart;
        }

        public void setWindowStart(long windowStart) {
            this.windowStart = windowStart;
        }
    }

    public static void main(String[] args) throws InterruptedException {
        // Example usage
        RateLimiter rateLimiter = new RateLimiter(5, 60); // 5 requests per minute

        String clientId = "client-1";

        // Test: 5 requests allowed, next denied
        for (int i = 0; i < 5; i++) {
            System.out.println("Request " + (i + 1) + ": " + rateLimiter.allowRequest(clientId)); // All true
        }
        System.out.println("Request 6: " + rateLimiter.allowRequest(clientId)); // False

        // Wait for 1 minute to reset
        Thread.sleep(60000);

        System.out.println("Request after reset: " + rateLimiter.allowRequest(clientId)); // True
    }
}
```
- Fixed window can allow bursts at the edges of windows, while sliding window is more smooth. Sliding window typically tracks timestamps of recent requests to determine if a new request is allowed. So for the sliding window algorithm, I need to store each request's timestamp or at least a list of timestamps within the current window for each client.
- Wait, but storing all timestamps might be memory-intensive. Oh right, there's an optimized version of the sliding window that uses a weighted average between previous window's count and current window's count. Alternatively, another approach is to use a circular buffer or a queue to track timestamps within the sliding window, and evict old entries as time progresses.
- But given that this is for a machine coding interview, the user might need a balance between accuracy and simplicity. The timestamp queue approach is straightforward but might have higher memory usage if there are many requests. However, for the sake of clarity and correctness, maybe the queue approach is better here.
- So the plan is: for each client, maintain a queue of timestamps. When a new request comes in, remove all timestamps older than the current time minus the window size. Then check if the queue's size is less than the maximum allowed requests. If yes, add the new timestamp and allow; else, deny.
- But handling concurrency here is important. Since multiple threads might be accessing the same client's queue, we need to synchronize properly. Using ConcurrentHashMap to store client queues, and perhaps synchronizing on the queue or a lock per client.
- Alternatively, using a data structure that allows concurrent modifications. But in Java, even with ConcurrentHashMap, the queue for each client would need thread-safe operations. So perhaps when accessing a client's queue, we need to synchronize on that queue to prevent race conditions.
- Let me outline the steps:
- Create a RateLimiter class with max requests and window size.
- Use a ConcurrentHashMap to map client IDs to their request queues (e.g., a deque of timestamps).
- When a request comes in, get the client's queue, synchronize on it.
- Remove all timestamps older than current time minus window size.
- If the size of the queue is less than max requests, add the current timestamp and allow the request.
- Else, deny the request.
- This approach ensures that for each client, we track all requests in the sliding window and check against the limit.
- Now, code structure:
- The RateLimiter class will have a ConcurrentHashMap<String, Deque<Long>>.
- The allowRequest method will compute the deque for each client if absent.
- Synchronize on the deque to handle concurrent access.
- Remove old entries, check size, add new timestamp if allowed.
- Testing with sample main method, similar to the previous example. Let me also think about edge cases, like when the window slides and some old entries are evicted.
- Potential issues: Synchronization on the deque. Since ConcurrentHashMap's computeIfAbsent is thread-safe, but when accessing the deque, we need to synchronize modifications. For example, two threads might try to add to the deque at the same time. So synchronizing on the deque instance would prevent that.
- Another thing: the deque could be a LinkedList, but since we are synchronizing, it's okay.
- Sliding Window Algorithm: Tracks the timestamps of requests for each client within a sliding time window. Requests older than the window are evicted before checking if a new request is allowed.
- ConcurrentHashMap: Stores client identifiers and their request timestamps in a thread-safe queue (Deque).
- Per-Client Synchronization: Ensures atomic updates to each client's request timestamp queue to prevent race conditions.
```
import java.util.Deque;
import java.util.LinkedList;
import java.util.concurrent.ConcurrentHashMap;

public class RateLimiter {
    private final int maxRequests;
    private final long windowSizeInMillis;
    private final ConcurrentHashMap<String, Deque<Long>> clientRequests;

    public RateLimiter(int maxRequests, long windowSizeInSeconds) {
        this.maxRequests = maxRequests;
        this.windowSizeInMillis = windowSizeInSeconds * 1000;
        this.clientRequests = new ConcurrentHashMap<>();
    }

    public boolean allowRequest(String clientId) {
        // Get or create the deque for the client
        Deque<Long> timestamps = clientRequests.computeIfAbsent(
            clientId, 
            k -> new LinkedList<>()
        );

        synchronized (timestamps) {
            long currentTime = System.currentTimeMillis();
            long windowStart = currentTime - windowSizeInMillis;

            // Evict outdated timestamps (older than the window)
            while (!timestamps.isEmpty() && timestamps.peekFirst() < windowStart) {
                timestamps.pollFirst();
            }

            if (timestamps.size() < maxRequests) {
                timestamps.addLast(currentTime);
                return true;
            } else {
                return false;
            }
        }
    }

    public static void main(String[] args) throws InterruptedException {
        // Example: Allow 5 requests per minute
        RateLimiter rateLimiter = new RateLimiter(5, 60);
        String clientId = "client-1";

        // Test 1: First 5 requests allowed
        for (int i = 0; i < 5; i++) {
            System.out.println("Request " + (i + 1) + ": " + rateLimiter.allowRequest(clientId));
        }

        // Test 2: 6th request denied (limit exceeded)
        System.out.println("Request 6: " + rateLimiter.allowRequest(clientId)); 

        // Test 3: Wait 60 seconds (window slides) and retry
        Thread.sleep(60_000);
        System.out.println("Request after reset: " + rateLimiter.allowRequest(clientId));
    }
}
```
