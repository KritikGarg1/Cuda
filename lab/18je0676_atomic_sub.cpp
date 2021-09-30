#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>

__device__ int myAtomicSub(int *address, int decr)
{
    int guess = *address;
    int oldValue = atomicCAS(address, guess, guess + decr);

    while (oldValue != guess)
    {
        guess = oldValue;
        oldValue = atomicCAS(address, guess, guess + decr);
    }

    return oldValue;
}

__global__ void kernel(int *sharedInteger)
{
    myAtomicSub(sharedInteger, -1);
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

