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


## 2.4 Device global memory and data transfer