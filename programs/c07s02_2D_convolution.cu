#include <stdio.h>
#include <jpeglib.h>
#include <cuda_runtime.h>

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