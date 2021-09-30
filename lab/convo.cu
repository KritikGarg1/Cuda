// Rishabh Agarwal - 18je0676
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
using namespace std;

#define check(statement) do {\
        cudaError_t error = statement;\
        if (error != cudaSuccess) {\
            cout << "Failed to run stmt " << __LINE__ << "\n";\
            cout << "Got CUDA error ...  " << cudaGetErrorString(error) << "\n";\
            return -1;\
        }\
    } while(0)

// kernel function
__global__ void convolutionKernel(float *a,float *b,float *c,int maskWidth,int width) {
    
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    float cvalue=0.0;
    int start_point=i-(maskWidth/2);
    
    for(int j = 0;j < maskWidth; j++) {
        if((start_point + j) >= 0 && (start_point+j) < width) {
            cvalue += a[start_point + j] * b[j];
        }
    }
    c[i]=cvalue;
}

// main function
int main() {
    
    float * input;
    float * mask;
    float * output;
    float * dinput;
    float * dmask;
    float * doutput;
    
    int maskWidth=3;
    int width=5;

    // allocating memomry to input, mask and output
    input = (float *)malloc(sizeof(float) * width);
    mask = (float *)malloc(sizeof(float) * maskWidth);
    output = (float *)malloc(sizeof(float) * width);
 
    // assigning values to input, mask and output
    for(int i=0;i<width;i++) {
        input[i]=1.0;
    }
    for(int i=0;i < maskWidth;i++) {
        mask[i]=1.0;
    }

    cout << "\nInput: \n";
    for(int i=0; i<width; i++) {
        cout << input[i] << " ";
    }
    cout << "\n";

    cout << "\nMask: \n";
    for(int i=0; i < maskWidth; i++) {
        cout << mask[i] << " ";
    }
    cout << "\n";

    // allocating device memory
    check(cudaMalloc((void **)&dinput, sizeof(float) * width));
    check(cudaMalloc((void **)&dmask, sizeof(float) * maskWidth));
    check(cudaMalloc((void **)&doutput, sizeof(float) * width));

    // copying memory from host to device
    check(cudaMemcpy(dinput, input, sizeof(float) * width, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dmask, mask, sizeof(float) * maskWidth, cudaMemcpyHostToDevice));

    // kernel dimensions
    dim3 dimGrid(((width-1)/maskWidth) + 1, 1,1);
    dim3 dimBlock(maskWidth,1, 1);

    // calling kernel
    convolutionKernel<<<dimGrid,dimBlock>>>(dinput, dmask, doutput, maskWidth, width);
    cudaDeviceSynchronize();

    // copying memory back from device to host
    check(cudaMemcpy(output, doutput, sizeof(float) * width, cudaMemcpyDeviceToHost));

    cout << "\nOutput: \n";
    for(int i=0; i < width; i++) {
        cout << output[i] << " ";
    }

    cudaFree(dinput);
    cudaFree(dmask);
    cudaFree(doutput);

    free(input);
    free(output);
    free(mask);

    return 0;
}
