// Rishabh Agarwal - 18JE0676
#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;
const int block_size = 512;

// kernel function

__global__  void parallelReductionKernel(float * din, float * dout, int inputElements) {
    
    // Load a segment of the din vector into shared memory
    const int block_size = 512;
    __shared__ float partialSum[2 * block_size];
    
    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    if ((start + t) < inputElements) {
        partialSum[t] = din[start + t];      
    }
    else {       
        partialSum[t] = 0.0;
    }

    if ((start + blockDim.x + t) < inputElements) {   
        partialSum[blockDim.x + t] = din[start + blockDim.x + t];
    }
    else {
        partialSum[blockDim.x + t] = 0.0;
    }

    // Traverse reduction tree
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
      __syncthreads();
        if (t < stride) {
            partialSum[t] += partialSum[t + stride];
        }
    }
    __syncthreads();

    // Write the computed sum of the block to the output vector at correct index
    if (t == 0 && (globalThreadId*2) < inputElements) {
        dout[blockIdx.x] = partialSum[t];
    }
}

// parallelReduction Function

void parallelReduction(float *in, float *out, int inputElements, int outputElements) {

    float *din, *dout;
    
    // device (my nvidia gpu) memory allocation
    cudaMalloc((void **)&din, inputElements * sizeof(float));
    cudaMalloc((void **)&dout, outputElements * sizeof(float));

    // transfer memory from host (laptop intel processor) to device (my nvidia gpu)
    cudaMemcpy(din, in, inputElements * sizeof(float), cudaMemcpyHostToDevice);

    // Number of Blocks requiredand number of threads in each block
    dim3 DimGrid( outputElements, 1, 1);
    dim3 DimBlock(block_size, 1, 1);

    // now calling cuda kernel for parallel reduction
    parallelReductionKernel<<<DimGrid, DimBlock>>>(din, dout, inputElements);
    
    // transfer memory from device (my nvidia gpu) to host (laptop intel processor)
    cudaMemcpy(out, dout, outputElements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(din);
    cudaFree(dout);
    
    return;
}

// main

int main() {
    
    int inputElements, outputElements;
    float *in, *out;
    
    cout << "Enter input elements: ";
    cin >> inputElements;
    outputElements = inputElements / (block_size<<1);
    if (inputElements % (block_size<<1)) {
        outputElements++;
    }
    
    // allocating memory
    in = (float *) malloc(sizeof(float) * inputElements);
    out = (float*) malloc(sizeof(float*) * outputElements);

    // assigning values
    for (int i=0; i < inputElements; i++) {
        in[i] = i;
    }
  
    parallelReduction(in, out, inputElements,outputElements);
    cout << "Reduced Sum from GPU = " << out[0];

    free(in);
    free(out);

    return 0;
}
