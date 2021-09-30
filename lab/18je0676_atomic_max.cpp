#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__device__ int myAtomicMax(int *address, int len)
{
    int guess = *address;
    int oldValue = atomicCAS(address, guess, guess + len);

    int val = 1 << 8;
    for (int i = 0; i < len; ++i)
    {
        val = max(val, i);
    }

    return val;
}

__global__ void kernel(int *sharedInteger)
{
    myAtomicMax(sharedInteger, 10);
}

int main(int argc, char **argv)
{
    int h_sharedInteger;
    int *d_sharedInteger;
    CHECK(cudaMalloc((void **)&d_sharedInteger, sizeof(int)));
    CHECK(cudaMemset(d_sharedInteger, 0x00, sizeof(int)));

    kernel<<<4, 128>>>(d_sharedInteger);

    CHECK(cudaMemcpy(&h_sharedInteger, d_sharedInteger, sizeof(int),
                     cudaMemcpyDeviceToHost));
    printf("4 x 128 increments led to value of %d\n", h_sharedInteger);

    return EXIT_SUCCESS;
}

