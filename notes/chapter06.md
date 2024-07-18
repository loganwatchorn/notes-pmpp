# Chapter 6: Performance Considerations

- [6.1: Memory coalescing](#61-memory-coalescing)
- [6.2: Hiding memory latency](#62-hiding-memory-latency)
- [6.3: Thread coarsening](#63-thread-coarsening)
- [6.4: A checklist of optimizations](#64-a-checklist-of-optimizations)
- [6.5: Knowing your computation's bottleneck](#65-knowing-your-computations-bottleneck)
- [6.6: Summary](#66-summary)

## 6.1 Memory coalescing
It takes tens of nanoseconds to evaluate whether a bit in DRAM is set or cleared - much longer than a clock cycle.

Each time a bit in DRAM is accessed, the nearby bits are also accessed. This is called a **burst**.

To increase access efficiency, have each thread in a warp access consecutive locations in global memory. That way, the memory accesses will be **coalesced** into bursts.

### Corner-turning
Consider matmul where A is stored in row-major, B in column-major, and C in row-major. The memory accesses for A's elements can be the same as what we've seen before. This will take advantage of bursting. For B, on the other hand, consecutive threads will not access consecutive memory addresses, and so bursting doesn't help us. To fix this, we should change the order of which threads in a warp access which memory addresses.

> Threads in a warp should access consecutive memory locations


## 6.2 Hiding memory latency

In a DRAM, there are **banks** which store the data. These are served by **channels**, which are attached to several banks by a bus. A bank can only serve a single burst at a time, so it's best for each channel to be querying several banks at once. That way, they can load the burst from one bank while processing the bursts for the other banks.

Let *R* be the the ratio of     