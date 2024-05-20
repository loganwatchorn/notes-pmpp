# Chapter 3: Multidimensional grids and data
In this chapter:
- How to process multi-dimensional arrays
- Converting a colored image to grayscale
- Matrix multiplication

## 3.1 Multidimensional grid organization
A kernel's execution configuration parameters are of type `dim3`, a struct containing ints x,y,z.
```c
dim3 dimGrid(ceil(n/256.0), 1, 1);
dim3 dimBlock(256, 1, 1);
myKernel<<<dimGrid, dimBlock>>>(...);
```
If we use only a 1D grid or 1D blocks, we can simply use ints instead of dim3s. The following is equivalent to the above.
```c
myKernel<<<ceil(n/256.0), 256>>>(...);
```
Within the kernel, we can access the first execution configuration parameter with `gridDim.(x|y|z)` and the second with `blockDim.(x|y|z)`.

- `gridDim` allowed values
    - `x: [1, 2^31)`
    - `y: [1, 2^16)`
    - `z: [1, 2^16)`
- `blockDim`
    - Allowed values: `1 <= x * y * z <= 1024`
    - For performance: `x`, `y`, `z` should all be multiples of 32


## 3.2 Mapping threads to multidimensional data
Photos are a 2D grid of pixels. To process them, use 2D grids of 2D blocks.

> Similar to matrix notation where A<sub>i,j</sub> indexes the element of A in the i<sup>th</sup> row and j<sup>th</sup> column, we index the blocks/threads with Z,Y,X, in that order.

For an image with 62X76 pixel, if we were to use 16X16 blocks, then we would need a 4X5 grid. (4 X 16 = 64 vertical threads, 5 X 16 = 80 horizontal threads).

A thread can get the coordinates of its corresponding pixel with the following.
```c
unsigned int i = blockDim.y * blockIdx.y + threadIdx.y;
unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;
```

To copy a dynamically-allocated 2D array from host to device, we need to **linearize**, or flatten, the 2D array into a 1D array. Two ways to do this:
1. **Row-major** layout: Each row's elements placed consecutively, then each row placed consecutively
    - Used by most languages including C
2. **Column-major** layour: Each columns elements placed consecutively, then each column placed consecutively
    - Used by FORTRAN

The following kernel converts a colored image to grayscale, using the equation `L = (r, g, b)x(0.21, 0.72, 0.07)` on each pixel.

```c
__global__
void convertRGBToGrayscale(
    unsigned char *grayPixels,
    unsigned char *rgbPixels,
    int width, 
    int height
) {
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < width && row < height) {
        int i = row * width + col;
        unsigned char r = rgbPixels[3 * i];
        unsigned char g = rgbPixels[3 * i + 1];
        unsigned char b = rgbPixels[3 * i + 2];
        grayPixels[i] = r*0.21f + g*0.72f + b*0.07f;
    }
}
```

For 3D grids, the three coordinates and linearized index can be computed as follows:
```c
unsigned int plane = blockDim.z * blockIdx.z + threadIdx.z;
unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
unsigned int i = width*height*plane + width*row + col;
```