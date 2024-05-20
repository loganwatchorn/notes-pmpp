# Chapter 2: Heterogeneous data parallel computing

## 2.1 Data parallelism
= When the computations performed on different parts of the dataset can be done independently of each other.

An example of a program which can be parallelized is converting an RGB image to grayscale. To do so, compute the luminance of each pixel:
```
L = r*0.21 + g*0.72 + b*0.07
```
Each pixel is computed independently of each other, so this program can be computed in parallel.

RGB Image Representation
- Allowable (R,G,B) values differ between industry-specified **color spaces**
- For AdobeRGB: R and G must fit in a triangle on the R,G plane. B is then 1-R-G. See book for diagram.

**Task parallelism** is another concept, similar but different than data parallelism, in which *tasks* and not necessarily *data* can be split up into separate threads.


## 2.2 CUDA C program structure
CUDA C:
- Extends regular C with minimal new syntax and library functions
- Is moving to adopt C++ features
- Program structure reflects existence of a **host** (CPU) and one or more **devices** (GPUs).
- Programs can have a mixture of host code and device code.
- If a CUDA C program contains only host code, it's just a regular C program.

Device code
- Marked with keywords
- Contains **kernels**, which are functions whose code is run in parallel.

When a CUDA C program is launched:
- First, the host code is executed on the CPU.
- When a kernel is called, many threads are launched on a device.
- These threads form what's called a **grid**.
- When all threads in a grid have completed execution, the grid terminates, and execution on the host continues.

Host and device code may be specified to run at the same time or not.

Unlike CPU threads, GPU threads can be generated with very few clock cycles.

### Threads
- Consist of
    - The program's code
    - Current point of execution in the code
    - Current values of in-scope variables
- Execution is sequential


## 2.3 A vector addition kernel
Throughout the book, we will differentiate host and device data by using the suffixes `_h` and `_d` in the variable names.

Below is the traditional, sequential way of adding two vectors, using host code.
```c
void add(float *x_h, float *y_h, float *res_h, int n) {
    for (int i = 0; i < n; ++i) {
        res_h[i] = x_h[i] + y_h[i];
    }
}
```

We can modify this function to run in parallel as device code. The structure of such a program is as follows:
```c
void add(float *x, float *y, float *res, int in) {
    int size = n * sizeof(float);
    float *x_d, *y_d, *res_d;

    // Step 1: Allocate device memory for x, y, res
    //   and copy x and y to device memory

    // Step 2: Call kernel, launching a grid of threads
    //   to perform the addition

    // Step 3: Copy res from the device memory
    //   and free the device memory
}
```

The new version of the function is like an outsourcing agent that
1. Ships data to device
2. Activates calculations on device
3. Collects resulting data from device

The rest of this chapter will cover how to fill in the steps of the `add` function.

<br>

## 2.4 Device global memory and data transfer
On-device DRAM is called device **global** memory.

### Allocating and freeing
- `cudaMalloc(addr, size)` allocates `size` bytes on device global memory.
    - `addr`
        - address of a pointer variable which will be set to point to the allocated object.
        - should be cast to `void**`, since the function expects a generic pointer
- `cudaFree(ptr)` frees the object pointed to by `ptr`.

```c
// Example
float *x_d;
int size = n * sizeof(float);
cudaMalloc((void**)&x_d, size);
// ...
cudaFree(x_d);
```

Pointers to device global memory such as `x_d` should not be dereferenced in host code. This leads to runtime errors.

### Transferring data
- `cudaMemcpy(dest, src, size, type)`
    - `dest`: address of variable to copy data into
    - `src`: address of variable to copy data from
    - `size`: number of bytes to copy
    - `type`:
        - Transfers from host to device: use `cudaMemcpyHostToDevice` constant
        - Transfers from device to host: use `cudaMemcpyDeviceToHost` constant

### Error handling
CUDA API functions return flags that indicate whether an error has occured. This book will not check for errors in the examples. The following is an example of how to handle errors.
```c
float* x_d;
cudaError_t err = cudaMalloc((void**)&x_d, sizeof(float));
if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
}
```

<br>

## 2.5 Kernel functions and threading
A kernel function includes code executed by each thread separately.

### Thread blocks and built-in kernel variables
When host code calls a kernel, CUDA launches a grid of threads organized into a two-level heirarchy.
- A grid is an array of **thread blocks**
    - Each block is the same size
    - All threads within a block have access to the uint variable `blockIdx`
- Each block is a 1D, 2D, or 3D array of threads.
    - Currently, each block may contain up to 1024 threads
    - Host code specifies how many threads in each block
    - For hardware efficiency, each dimension should be a multiple of 32
        - This multiple should be chosen based on several factors discussed later
    - Within a thread, use the uints `blockDim.x`, `blockDim.y`, and `blockDim.z` to see the dimensions of the containing block
    - Within a thread, use the uints `threadIdx.x`, `threadIdx.y`, `threadIdx.z` to get the thread's block-wide unique index.

```c
// A thread's grid-wide unique index in a 1D block
int i = blockDim.x * blockIdx.x + threadIdx.x;
```

<br>

### Kernel, device, and host functions
The `__global__` keyword specifies a function as a **kernel function**. These are the top-level function executed by a single thread. Called by host. Launches a grid of threads.
```c
__global__
void vectorAdditionKernel(float *x, float *y, float *res, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        res[i] = x[i] + y[i];
    }
}
```
The `__device__` keyword specifies a function as a **device function**. These run in a thread can only be called from a kernel function or another device function. Does not launch any new threads.

The `__host__` keyword specifies a function as a **host function**. These run on the host and can only be called by the main thread or another host function. All non-kernel and non-device functions are host functions by default.

If you precede a function with both `__device__` and `__host__`, that function will be able to run on either the device or host.

<br>

## 2.6 Calling kernel functions
Use **execution configuration parameters** to set the thread block dimensions. This can be done calling the kernel function with the following syntax: 
```c
void hostFunc() {
    myKernelFunc<<<numBlocks, numThreads>>>(args...);
}
```

That's all we need to complete our code for vector addition.
```c
__global__
void vecAddKernel(float *x, float *y, float *res, int n) {
    // Applied to each element of the vector
    // Ran separately in each GPU thread
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        res[i] = x[i] + y[i];
    }
}

__host__
void vecAdd(float *x_h, float *y_h, float *res_h, int n) {
    int size = n * sizeof(float);
    float *x_d, *y_d, *res_d;

    // Step 1: Allocate space on device (GPU) for x, y, res
    cudaMalloc((void**)&x_d, size);
    cudaMalloc((void**)&y_d, size);
    cudaMalloc((void**)&res_d, size);

    // Step 2: Copy x and y from host to device
    cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, size, cudaMemcpyHostToDevice);

    // Step 3: Invoke kernel on a grid of 256-thread blocks
    int blockSize = 256;
    int blockCount = ceil(n / (float)blockSize);
    vecAddKernel<<<blockCount, blockSize>>>(x_d, y_d, res_d, n);

    // Step 4: Copy res from device to host
    cudaMemcpy(res_h, res_d, size, cudaMemcpyDeviceToHost);

    // Step 5: Free  memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(res_d);
}
```


<br>

## 2.7 Compilation
To compile a CUDA program, we can use **NVCC** (NVIDIA C compiler).

Host code is compiled using the host's standard C/C++ compilers.

Device code is compiled into **PTX** files, which contain virtual binary. PTX files are then compiled into object files for execution on NVIDIA GPUs.