// Rishabh Agarwal - 18JE0676
#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

// kernel function

__global__ void kernelFunction(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs)/2;
    }
}


int main( void ) {

    cudaDeviceProp  prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    
    if (!prop.deviceOverlap) {
        cout << "Device will not handle overlaps, so no speed up from streams\n";
        return 0;
    }    
    if(prop.concurrentKernels == 0) {
        cout << "> GPU does not support concurrent kernel execution\n";
        cout << "  CUDA kernel runs will be serialized\n";
    }
    if(prop.asyncEngineCount == 0) {
        cout << "GPU does not support concurrent Data transer and overlaping of kernel execution & data transfer\n";
        cout << "Mem copy call will be blocking calls\n";
    }

    cudaEvent_t start, stop;
    float elapsedTime;

    int n = 1024*1024;
    int maxsize = n*20;

    int *ha, *hb, *hc;
    int *da0, *db0, *dc0, *da1, *db1, *dc1;
    cudaStream_t    stream0, stream1;

    // start the timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // initialize the streams
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // allocate the memory on the GPU
    cudaMalloc(&da0, n * sizeof(int));
    cudaMalloc(&da1, n * sizeof(int));
    cudaMalloc(&db0, n * sizeof(int));
    cudaMalloc(&db1, n * sizeof(int));
    cudaMalloc(&dc0, n * sizeof(int));
    cudaMalloc(&dc1, n * sizeof(int));

    // allocate host locked memory, used to stream
    cudaHostAlloc((void**)&ha, maxsize * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&hb, maxsize * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&hc, maxsize * sizeof(int), cudaHostAllocDefault);

    for(int i=0; i < maxsize; i++) {
        ha[i] = i + 10;
        hb[i] = i + 200;
    }

    cudaEventRecord(start, 0);
    for(int i=0; i < maxsize; i += n*2) {
        
        // enqueue copies of a in stream0 and stream1
        cudaMemcpyAsync(da0, ha + i, n * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(da1, ha + i + n, n * sizeof(int), cudaMemcpyHostToDevice, stream1);
        
        // enqueue copies of b in stream0 and stream1
        cudaMemcpyAsync(db0, hb + i, n * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(db1, hb + i + n, n * sizeof(int), cudaMemcpyHostToDevice, stream1);

        // enqueue kernels in stream0 and stream1   
        kernelFunction <<< n/256, 256, 0, stream0 >>> (da0, db0, dc0, n);
        kernelFunction <<< n/256, 256, 0, stream1 >>> (da1, db1, dc1, n);

        // enqueue copies of c from device to locked memory
        cudaMemcpyAsync(hc + i, dc0, n * sizeof(int), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(hc + i + n, dc1, n * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    }

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Time taken in ms: " <<  elapsedTime << "\n\n";


    // we are printing only upto 20 elements
    cout << "Vector A: \n";
    for(int i=0; i < 20; i++) {
        cout << ha[i] << " ";
    }
    cout << "\n\n";

    cout << "Vector B: \n";
    for(int i=0; i < 20; i++) {
        cout << hb[i] << " ";
    }
    cout << "\n\n";

    cout <<"After performing operation: C[i] = ((A[i] + A[i+1] + A[i+2]) / 3 + (B[i] + B[i+1] + B[i+2]) / 3) / 2\n";
    cout << "Vector C: \n";
    for(int i=0; i < 20; i++) {
        cout << hc[i] << " ";
    }
    cout << "\n\n";

    cudaFreeHost(ha);
    cudaFreeHost(hb);
    cudaFreeHost(hc);
    
    cudaFree(da0);
    cudaFree(da1);
    cudaFree(db0);
    cudaFree(db1);
    cudaFree(dc0);
    cudaFree(dc1);
    
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    return 0;
}
