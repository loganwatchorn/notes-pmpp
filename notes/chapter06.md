# Chapter 6: Performance Considerations

- [6.1 Memory coalescing](#61-memory-coalescing)
- [6.2 Hiding memory latency](#62-hiding-memory-latency)
- [6.3 Thread coarsening](#63-thread-coarsening)
- [6.4 A checklist of optimizations](#64-a-checklist-of-optimizations)

## 6.1 Memory coalescing
It takes tens of nanoseconds to evaluate whether a bit in DRAM is set or cleared - much longer than a clock cycle.

Each time a bit in DRAM is accessed, the nearby bits are also accessed. This is called a **burst**.

To increase access efficiency, have each thread in a warp access consecutive locations in global memory. That way, the memory accesses will be **coalesced** into bursts.

### Corner-turning
Consider matmul where A is stored in row-major, B in column-major, and C in row-major. The memory accesses for A's elements can be the same as what we've seen before. This will take advantage of bursting. For B, on the other hand, consecutive threads will not access consecutive memory addresses, and so bursting doesn't help us. To fix this, we should change the order of which threads in a warp access which memory addresses.

> Threads in a warp should access consecutive memory locations


## 6.2 Hiding memory latency
A DRAM consists of **banks**, which store the bits, and **channels**, which connect a DRAM port to one or more banks.

Each bank can only serve a single burst at a time, so it's best for each channel to query multiple banks simultaneously. That way, multiple banks can come up with their bursts at the same time, and the channel can return the bursts as they are completed by the banks.

If *R* is the ratio of cell array access latency to data transfer time, we need at least *R+1* banks to use the channel's full bandwidth.

**Bank conflict** is when a channel tries to access the same bank multiple times at once. This can be avoided by having *R+1* or more banks per channel.

In GPUs, the DRAM is equipped with a cache so that similar queries performed near the same time will only require one set of bursts from the banks.


## 6.3 Thread coarsening
Costs of parallelization with maximal granularity:
- Redundant memory accesses by seperate blocks
- Reduntant work performed by the threads
- Synchronization overhead

When there's too much granularity, the GPU may have to execute the threads in series. This introduces overhead and is sub-optimal to simply having fewer threads doing more work. We call this **thread coarsening**.

The following is an example of thread-coarsened tiled matrix multiplication. Here, each thread block is responsible for computing `COARSE_FACTOR` tiles, and each thread computes one value for each of those tiles.

```c
#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__ void matmul(float* A, float* B, float* C, int N) {
    __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_s[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col_first = blockIdx.x * TILE_WIDTH * COARSE_FACTOR + threadIdx.x;

    // Initialize output elements
    float C_s[COARSE_FACTOR];
    for (int i = 0; i < COARSE_FACTOR; ++i) {
        C_s[i] = 0.0f;
    }

    // Compute output elements in phases
    for (int ph = 0; ph < N / TILE_WIDTH; ++ph) {
        A_s[threadIdx.y][threadIdx.x] = A[row*N + ph*TILE_WIDTH + threadIdx.y];
        
        for (int i=0; i<COARSE_FACTOR; ++i) {
            int col = col_first + i * TILE_WIDTH;

            // Collaborative loading of B into shared memory
            B_s[threadIdx.y][threadIdx.x] = B[(ph*TILE_WIDTH + threadIdx.y)*N + col];
            __syncthreads();

            for (int j=0; j<TILE_WIDTH; ++j) {
                C_s[j] += A_s[threadIdx.y][j] * B_s[j][threadIdx.x];
            }
            __syncthreads();
        }
    }

    // Load output elements to device global memory
    for (int i=0; i<COARSE_FACTOR; ++i) {
        int col = col_first + i * TILE_WIDTH;
        C[row*N + col] = C_s[i];
    }
}
```

Be careful when thread coarsening, because it won't increase performance unless it does away with redundant work, and assigning too much work to a single thread might mean you don't use all the hardware resources.


## 6.4 A checklist of optimizations
- Maximize occupancy
- Coalesce accesses to global memory
- Minimize control divergence
- Use tiling for reusable data
- Privatization (covered later)
- Thread coarsening


## 6.5 Knowing your computation's bottlenecks
The resource most responsible for limiting performance is what we call the bottleneck. If you make changes to optimize resources other than the bottleneck, it may in fact hurt performance. For example, introducing tiling will increase the use of shared memory and reduce calls to global memory. If the bandwidth is shared memory, than tiling will just reduce occupancy and cause worse performance. However, if global memory bandwidth is the bottleneck and SMs have shared memory to spare, than introducing tiling will help.

Make use of profiling tools to find which resource is your program's bottleneck. Some popular profilers for CUDA are:
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)