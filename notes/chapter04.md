# Chapter 4: Compute architecture and scheduling
1. [Architecture of a modern GPU](#41-architecture-of-a-modern-gpu)
2. [Block scheduling](#42-block-scheduling)
3. [Synchronization and transparent scalability](#43-synchronization-and-transparent-scalability)
4. [Warps and SIMD hardware](#44-warps-and-simd-hardware)
5. [Control divergence](#45-control-divergence)
6. [Warp scheduling and latency tolerance](#46-warp-scheduling-and-latency-tolerance)
7. [Resource partitioning and occupancy](#47-resource-partitioning-and-occupancy)
8. [Querying device properties](#48-querying-device-properties)


## 4.1 Architecture of a modern GPU
- Each GPU has an array of **SM**s (Highly-Threaded Streaming Multiprocessors)

- Each SM has several processing units called streaming processors or **CUDA cores**

    - Ampere A100 GPUs have 108 SMs with 64 cores each

- Each SM has its own on-chip memory unit

- **Memory** consists of the SMs' on-chip memory units

- **Global Memory** is the GPU's DRAM, which is off-chip
    - Old GPUs used [double data rate synchronous DRAM](https://en.wikipedia.org/wiki/Double_data_rate)

    - New GPUs use HBM ([high-bandwidth memory](https://en.wikipedia.org/wiki/High_Bandwidth_Memory)) or HBM2


## 4.2 Block scheduling
- All threads in a block are assigned to the same SM

- Multiple blocks can be assigned to the same SM

- Each SM can only execute so many blocks at once

- The runtime system keeps a list of blocks that are waiting

- After an SM executes a group of blocks, the runtime will assign a new group of blocks to it

- All threads in a block are scheduled simultaneously on the same SM

    - Because of this, threads in the same block can interact with **barrier synchronization**, which is communication via **shared memory** located on the SM

    - Threads in different blocks can interact with the [Cooperative Groups API](https://developer.nvidia.com/blog/cooperative-groups)


## 4.3 Synchronization and transparent scalability
- Threads in the same block can coordinate using the barrier synchronization function `__syncthreads()`

- When a thread calls `__syncthreads()`, it will halt until all other threads in its block reach that step

- If a kernel calls `__syncthreads()`, each thread in a block must reach the call

    - If it's in one of the blocks of an if statement, each thread in the same block must run the same path, either if-then or if-else

- The following is faulty kernel code, since it defines two distinct barriers
    ```c
    if (threadIdx.x % 2 == 0) {
        __syncthreads();
    } else {
        __syncthreads();
    }
    ```

- Incorrect barrier synchronization may cause:
    - Undefined execution behavior
    - Incorrect results
    - **Deadlock** (threads waiting forever for each other)

- Since threads in different can't perform barrier synchronization, GPUs can execute the blocks in any order


## 4.4 Warps and SIMD hardware
- Threads in a block can execute in any order

- Multi-phase algorithms should separate the phases with barrier synchronization

- Once a block has been assigned to an SM, it is divided into 32-thread units called **warps**

- If a block has a dimension that's not a multiple of 32, the last warp will be padded with threads that do nothing

- If a block has multiple dimensions of threads, the threads are linearized in row-major order and divided into warps that way.