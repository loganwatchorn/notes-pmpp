#include <stdio.h>
#include <jpeglib.h>
#include <cuda_runtime.h>


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


int main() {
    float F_host[FILTER_WIDTH][FILTER_WIDTH] = {/* ... */};
    size_t F_size = FILTER_WIDTH * FILTER_WIDTH * sizeof(float);
    cudaError_t err = cudaMemcpyToSymbol(F, F_host, F_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    convolution_kernel_2d<<</*...*/>>>(/*...*/);

    return 0;
}