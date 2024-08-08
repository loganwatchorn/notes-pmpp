# Chapter 7: Convolution

- [7.1 Background](#71-background)
- [7.2 Parallel convolution: a basic algorithm](#72-parallel-convolution-a-basic-algorithm)
- [7.3 Constant memory and caching](#73-constant-memory-and-caching)
- [7.4 Tiled convolution with halo cells](#74-tiled-convolution-with-halo-cells)
- [7.5 Tiled convolution using caches for halo cells](#75-tiled-convolution-using-caches-for-halo-cells)
- [7.6 Summary](#76-summary)

## 7.1 Background
Convolution is an operation where each output element is computed by taking a weighted sum of nearby input elements. This can happen in one, two, or more dimensions.

The set of weights is called a convolution kernel. To avoid confusion with CUDA kernels, we will call the set of weights a **convolution filter**.

Boundary conditions occur for output elements near the edges of the output object. Here, we could assume that the input elements outside of the actual input objects are zero. These are called **ghost cells**. We won't always set them to zero.


## 7.2 Parallel convolution: a basic algorithm
```c
__global__ void convolution_kernel_2d(
    float* A_in,
    float* A_out,
    int A_width,
    int A_height,
    float* F,
    int F_radius
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A_height && col < A_width) {
        float A_out_val = 0.0f;

        int F_width = 2 * F_radius + 1;
        for (int rowF = 0; rowF < F_width; rowF++) {
            for (int colF = 0; colF < F_width; colF++) {
                int rowIn = row - F_radius + rowF;
                int colIn = col - F_radius + colF;

                if (0 <= rowIn && rowIn < A_height && 0 <= colIn && colIn < A_width) {
                    A_out_val += F[rowF * F_width + colF] * A_in[rowIn * A_width + colIn];
                } // else: ghost cell, A_in_val is zero
            }
        }

        A_out[row * A_width + col] = A_out_val;
    }
}
```

The ratio of FLOPs to global memory accesses is about 0.25 FLOP/B. This is because the multiply-accumulate line performs 2FLOPs and loads 8B. A program with a computational intensity of 0.25 FLOP/B will likely be bottlenecked by memory bandwidth far before reaching peak performance.

## 7.3 Constant memory and caching
The size of F is typically small, never changes, and is used by all threads in the same order. This makes it a good candidate for **global memory**.

The function `cudaMemcpyToSymbol(dest, src, size)` is responsible for copying the host object src to the device constant memory object dest.

The following host code will declare the constant-memory variable `F` and move the elements from the host variable `F_h` into `F`.
```c
#define FILTER_RADIUS 2
#define FILTER_WIDTH 5
__constant__ float F_const[FILTER_WIDTH][FILTER_WIDTH];

int main() {
    float F_host[FILTER_WIDTH][FILTER_WIDTH] = {/* ... */};
    size_t F_size = FILTER_WIDTH * FILTER_WIDTH * sizeof(float);
    cudaMemcpyToSymbol(F_const, F_host, F_size);
}
```

Now, all we have to do to our kernel is change the multiply-accumulate row to access `F` with `F_const[rowF][colF]` instead of `F[rowF*F_width + colF]`.

Note that CUDA constant-memory variables follow C scoping rules for constant variables. So if a function needed to access a constant-memory variable defined in another file, we would need to re-declare it in the current file.

### Caches
After accessing an address in global memory, the value will be copied into caches. Typically, caches are numbered according to their distance to the processor core.

L1 caches are directly attached to a processor core. They are fast but small.

L2 caches are connected to the L1 caches, and can store a bit more information but are slower.

L3 devices exist in some devices, and are between L2 caches and memory.

Caches are resource-heavy when they allow for writes to memory. For constant-memory, caches are much simpler to implement. These specialized caches are called **constant-caches**.

In the case of convolution, since F is small, it will generally entirely be stored in the cache, meaning accesses to F consume almost no DRAM bandwidth. This means the computational intensity of our program has doubled, from 0.25 FLOP/B to 0.5 FLOP/B.

## 7.4 Tiled convolution with halo cells
An **input tile** will be the set of input elements required to compute all elements of an **output tile**. In convolution, the input tile is divided into two parts. The center, and the halo. Each thread in the block corresponds with a single element in the tile's center, and the halo consists of the elements surrounding the center.

Like before, the threads in a block will collaboratively load the input tile. Each thread will load a single element from the tile's center. The halo is slightly trickier.

There are two strategies for setting block sizes.
- block size = output tile size. Each thread:
  - Loads 1 input tile center element
  - Loads 0 or more input tile halo elements
  - Computes 1 output tile element
- block size = input tile size. Each thread:
  - Loads 1 input tile element
  - Computes 0 or 1 output tile elements

The following is an example where the block size equals the input tile size, which is the simpler strategy.
```c
#define FILTER_RADIUS 7
#define FILTER_DIM 15

#define INPUT_TILE_DIM 32
#define OUTPUT_TILE_DIM (INPUT_TILE_DIM - 2 * FILTER_RADIUS)

__constant__ float F_c[FILTER_DIM][FILTER_DIM];
__global__ void convolution_kernel_2d(
    float* A_in,
    float* A_out,
    int A_width,
    int A_height
) {
    int row = blockIdx.y * OUTPUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int col = blockIdx.x * OUTPUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;

    // Collaboratively load input tile
    __shared__ A_in_s[INPUT_TILE_DIM][INPUT_TILE_DIM];
    if (row < 0 || row >= A_height || col < 0 || col >= A_width) {
        A_in_s[row][col] = 0.0f; // ghost cell
    } else {
        A_in_s[row][col] = A_in[row * A_width + col];
    }
    __syncthreads();

    // Calculating output elements
    int tileRow = threadIdx.x - FILTER_RADIUS;
    int tileCol = threadIdx.y - FILTER_RADIUS;
    if (0 <= row && row < A_height
        && 0 <= col && col < A_width
        && 0 <= tileRow && tileRow < OUTPUT_TILE_DIM
        && 0 <= tileCol && tileCol < OUTPUT_TILE_DIM
    ) {
        float A_out_val = 0.0f;
        for (int fRow=0; fRow<FILTER_DIM; ++fRow) {
            for (int fCol=0; fCol<FILTER_DIM; ++fCol) {
                A_out_val += F[fRow][fCol] * A_in[fRow + tileRow][fCol + tileCol];
            }
        }
        A_out[row * A_width + col] = A_out_val;
    }
}
```

The computational intensity of this algorithm is approximately `O(T^2 * F^2 / (T + F)) FLOP/B` where T is the width of the output tile and F is the width of the filter. Refer to page 167 for precise calculation.

## 7.5 Tiled convolution using caches for halo cells
The halo cells of an input tile are also the internal cells of neighboring input tiles, so it is likely they will exist within a cache. Because of this, we may avoid loading them into device shared memory.

## 7.6 Summary
Some applications of convolution are Convolutional Neural Networks (Chapter 16), and grid point force computations, used in Iterative MRI Reconstruction (Chapter 17).

A similar pattern, stencil (next chapter), can be used in partial differential equation solvers.