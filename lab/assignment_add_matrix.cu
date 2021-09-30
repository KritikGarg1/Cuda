#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

__global__ void MatrixAddKernel(float *da, float *db, float* dc, int n, int m) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < n*m) {
      dc[i] = da[i] + db[i];
  }
}

void MatrixAdd(float *a, float *b, float *c, int n, int m) {

  float *da, *db, *dc;
  int size = n*m * sizeof(sizeof(float));

  // memory allocation
  cudaMalloc((void**)&da, size);
  cudaMalloc((void**)&db, size);
  cudaMalloc((void**)&dc, size);

  // transfer memory from host to device
  cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);
  
  // calling cuda kernel to add
  int blockSize = 128;
  int gridSize = (int)ceil((float)n/blockSize);
  MatrixAddKernel<<<gridSize, blockSize>>>(da, db, dc, n, m);

  // transfer memory from device to host 
  cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  return;
}

int main() {
  
  int n, m;
  cout << "Enter n: ";
  cin >> n;
  cout << "Enter m: ";
  cin >> m;
  
  // allocating memories
  float *a, *b, *c;
  a = (float*)malloc(n*m * sizeof(float));
  b = (float*)malloc(n*m * sizeof(float));
  c = (float*)malloc(n*m * sizeof(float));

  int lower = 10, upper = 20;
  for(int i=0; i<n*m; i++) {
	int a1 = (rand() % (upper - lower + 1)) + lower;
	int b1 = (rand() % (upper - lower + 1)) + lower;
        a[i] = a1;
        b[i] = b1;
  }
  
  cout << "A is : \n";
  int t = 0;
  for(int i=0; i<n; i++) {
    for(int j=0; j<m; j++) {
        cout << a[t] << " ";
        t++;
    }
    cout << "\n";
  }
  
  t = 0;
  cout << "\nB is : \n";
  for(int i=0; i<n; i++) {
    for(int j=0; j<m; j++) {
        cout << b[t] << " ";
        t++;
    }
    cout << "\n";
  }

  cout << "\nAfter adding...\n";
  MatrixAdd(a, b, c, n, m);
  
  t = 0;
  cout << "C is : \n";
  for(int i=0; i<n; i++) {
    for(int j=0; j<m; j++) {
        cout << c[t] <<  " ";
        t++;
    }
    cout << "\n";
  }

  return 0;
}
