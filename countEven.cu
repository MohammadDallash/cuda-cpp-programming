#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>// CUDA kernel. Each thread takes care of one element of c


using namespace std;

__global__ void fun(long long *a, long long n)
{
    // Get our global thread ID
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n and id%2==0)
        atomicAdd(reinterpret_cast<unsigned long long*>(a), 1);
}

int main(int argc, char *argv[])
{
    // Size of vectors
    long long n = 1e12;

    // Host input number
    long long *h_a;
    h_a = (long long *)malloc(sizeof(long long));
    *h_a = 0;

    // Device input number
    long long *d_a;

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, sizeof(long long));

    // Copy host number to device
    cudaMemcpy(d_a, h_a, sizeof(long long), cudaMemcpyHostToDevice);

    long long blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (long long)ceil((float)n / blockSize);

    cout << "Block Size: " << blockSize << ", Grid Size: " << gridSize << '\n';

    // Execute the kernel
    fun<<<gridSize, blockSize>>>(d_a, n);

    // Copy array back to host
    cudaMemcpy(h_a, d_a, sizeof(long long), cudaMemcpyDeviceToHost);

    ::cout << "Result: " << *h_a << '\n';

    // Release device memory
    cudaFree(d_a);

    // Release host memory
    free(h_a);

    return 0;
}