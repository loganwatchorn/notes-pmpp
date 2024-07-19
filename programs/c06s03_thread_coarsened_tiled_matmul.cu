#include <stdio.h>
#include <jpeglib.h>
#include <cuda_runtime.h>


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