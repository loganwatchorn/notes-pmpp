# Chapter 1: Introduction
Over the decades, CPU clock speeds have become faster and faster, meaning more operations per second (throughput). We're now at a point where the clock speed can't get much faster, because we're already dissipating as much heat as we can.

The workaround we've used is to have several processor cores in a CPU, computing separate independent tasks, or **threads**, at the same time.

But what do we do when we have thousands, millions, or quadrillions of threads to compute at the same time? We call this a **parallel program**, and it's what GPUs were invented for.

## 1.1 Heterogeneous parallel computing

### GPUs vs. CPUs
Two trajectories of microprocessor design:
- **multicore**: focuses on execution speed of several separate single-threaded programs each running on a separate core
    - ARM Ampere has 128 cores
- **many-thread**: focuses on execution throughput of a parallel application with orders of magnitude more threads
    - NVIDIA A100
        - 9.7 TFLOPS for 64b double-precision floating-point
        - 156 TFLOPS for 32b single-precision floating-point
        - 312 TFLOPS for 16b half-precision floating-point

Why is there such a large throughput difference between the two architectures?

CPUs speed up individual arithmetic operations by using more space and energy. They feature:
- "Arithmetic Logic Units" (ALU), large and power-hungry, but fast
- A large "last-level cache" storing frequently-called memory addresses, saving long-latency calls to DRAM.

GPUs have a smaller cache. The arithmetic units are slower, but use less space and energy. Because of this, a GPU with the same size as a CPU will be fit many more arithmetic units and require less energy per operation.

GPUs also need to read and update a significantly large number of DRAM addresses so that the application can access computed values (e.g. rgb values to send to screen pixels)

|             | CPU                           | GPU                           |
|-------------|-------------------------------|-------------------------------|
| ALU         | Large, power-hungry, fast, fewer | Smaller, less power, slower, far more numerous |
| Cache       | Huge                          | Small                         |
| DRAM access | Limited by legacy system requirements | More memory channels = ~10x CPU bandwith     |

Two ways to double the operation rate
- Doubling throughput: 2x number of AUs = 2x area + 2x power consumption
- Halving latency: 2x current = 2x area + 4x power

### CUDA
**Compute Unified Device Architecture (CUDA)** was invented by NVIDIA in 2007, and consists of the programming language as well as a modified hardware architecture for running it. Before CUDA, graphics chips required graphics libraries like OpenGL to be programmed, meaning you would have to frame your parallel computation so that matrices are a grid of pixels.

Some tasks will run far slower on GPUs than CPUs. Because of this, CUDA code runs on both the GPU and the CPU.

### FPGAs
Field-programmable gate arrays (FPGA) are another type of device which have been used for parallel computing, particularly for networking.


## 1.2 Why more speed or parallelism?
Expands capabilities in several domains.
- Biology: allows a new type of microscopes which see smaller than possible with optical lense based microscopes, by simulating what's going on based on measurements.
- Graphics: new consumer displays will use more parallelism to make high-res displays from low-res images
- Gaming: allows for real-time simulation rather than hardcoded scenes
- "Digital Twins": simulated copies of physical objects allowing for cheaper stress testing

A major application is AI. Most of the training of neural networks is matrix multiplication is vastly accelerated by parallel computing.


## 1.3 Speeding up real applications
We define **speedup** of a program running in system A vs system B as (time to run on B) / (time to run on A)

The speedup of a program on a parallel system vs on a serial system depends on how much of the program can be parallelized.

An important factor for how much speedup is possible is the latency of read/write operations to memory. With GPUs, we can use several techniques that use on-chip memory stores rather than DRAM, making memory accesses far faster.


## 1.4 Challenges in parallel computing
1. Can be difficult to design a parallel algorithm with the same time complexity as its sequential counterpart.
    - **Work efficiency** is a related concept we'll define later
2. **Memory bound programs**: programs limited by memory access latency. Chapters 5 (Memory architecture & data latency) and 6 (Performance Considerations) introduce methods to optimize this.
3. Data irregularity. If a dataset contains unexpectedly varying data types, some threads might take longer than others to finish. We will study methods to regularize data so that the threads take similar amounts of time to complete.
4. Some programs require **synchronization operations** to ensure no threads are left behind/too far forward. This takes extra work not needed in sequential systems. For some parallel programs, the threads are coded so that they will almost always be in sync, without needing to be synchronized from the outside. These are sometimes called **embarrassingly parallel**.


## 1.5 Related parallel programming interfaces
- **OpenMP**: for shared memory multiprocessor systems
    - Consists of a compiler and a runtime
    - Compiler takes directives and "pragmas" about a loop, and generates parallel code
    - Runtime executes code by managing parallel resources
    - Originally designed for CPU execution, now supports GPUs
- **MPI** (Message Passing Interface): for scalable cluster computing
    - Interface for running code across several nodes, each having their own separate memory
    - Interactions between nodes are handled by explicitly passing messages
    - Good tool for cluster-level computing, but not as good as CUDA or OpenMP for programs on a single-GPU
    - It's important (currently) for a parallel programmer working on large multi-GPU clusters to know how to write both MPI and CUDA. This will be covered in chapter 20 (Programming a Heterogeneous Computing Cluster)
- **NCCL** (NVIDIA Collective Compute Library): a CUDA API for multi-GPU programming
    - Increasing support for this
- **OpenCL** (Open Compute Language): Standardized parallel programming model
    - Developed by Apple, Intel, AMD/ATI, NVIDIA
    - Very similar to CUDA
