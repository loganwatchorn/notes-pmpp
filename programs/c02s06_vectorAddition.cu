#include <stdio.h>
#include <cuda_runtime.h>

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

int main() {
    int N = 1000;
    float x_h[N], y_h[N], res_h[N];
    for (int i = 0; i < N; i++) {
        x_h[i] = (float)i;
        y_h[i] = (i / 2.0) * (i / 2.0);
    }

    vecAdd(x_h, y_h, res_h, N);

    for (int i = 0; i < 100; i++) {
        printf("res_h[%2d]: %6f, delta: %4f\n", i, res_h[i], res_h[i] - (x_h[i] + y_h[i]));
    }
    printf("...\n");

    return 0;
}