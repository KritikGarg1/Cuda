#include <stdio.h>
//using namespace std;
//#typedef n 100

// Kernel Definition

__global__ void VecAddKernel(float *d_A, float *d_B, float *d_C, int n){
  int i=blockDim.x*blockIdx.x+threadIdx.x;
  if(i<n) d_C[i]=d_A[i]+d_B[i];
}
void vecAdd(float *A, float *B, float *C, int n){
  float *d_A, *d_B, *d_C;
  int size=n*sizeof(float);

  // Device Memory Allocation
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  // Host to Device data transfer
  cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
  cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);
  
  // Calling Kernel
  VecAddKernel<<< ceil(n/16),16>>> (d_A,d_B,d_C,n);
  cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);

}


int main(){
  
  float *A, *B, *C;
  int n;
  
  printf("Enter the size of Vector");
  scanf("%d",&n);
  
  A = (float*)malloc(n*sizeof(float));
  B = (float*)malloc(n*sizeof(float));
  C = (float*)malloc(n*sizeof(float));

  for(int i=0;i<n;i++){
    A[i]=i;
    B[i]=i*i;
  }
  
  vecAdd(A,B,C,n);
  printf("The value of A+B .i.e  C = \n{");
  
  for(int i=0;i<n;i++){
    printf("%f, ",C[i]);
  }
  
  printf("}\n");

  return 0;
}
