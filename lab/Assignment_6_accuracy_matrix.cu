// Rishabh Agarwal - 18JE0676
#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

// kernel functions

// single precision addition
__global__ void MatrixAddFloatKernel(float *temp_ha, float *temp_hb, float *temp_hd, int n, int m) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n*m) {
        temp_hd[i] = temp_ha[i] + temp_hb[i];
    }
}

__global__ void MatrixAddFloatKernel(double *temp_had, double *temp_hbd, double *temp_hdd, int n, int m) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n*m) {
        temp_hdd[i] = temp_had[i] + temp_hbd[i];
    }
}

// main()

int main() {

    int n, m;
    cout << "Enter n: ";
    cin >> n;
    cout << "\nEnter m: ";
    cin >> m;
    float totalf = (float)n*m;
    double totald = (double)n*m;

    // Single point precision matrix addition
    cout << "Single point precision matrix addition\n\n";
    
    // allocating memories
    float *ha, *hb, *hc, *temp_ha, *temp_hb, *temp_hd, *hd;
    ha = (float*)malloc(n*m * sizeof(float));
    hb = (float*)malloc(n*m * sizeof(float));
    hc = (float*)malloc(n*m * sizeof(float));
    hd = (float*)malloc(n*m * sizeof(float));

    cudaMalloc((void**)&temp_ha, n*m * sizeof(float));
    cudaMalloc((void**)&temp_hb, n*m * sizeof(float));
    cudaMalloc((void**)&temp_hd, n*m * sizeof(float));

    // assigning values
    for(int i=0; i<n*m; i++) {
        ha[i] = ((float)i + totalf) / totalf;
        hb[i] = ((float)i*i + totalf) / totalf;
        hc[i] = ha[i] + hb[i];
    }

    // copying memory from host to device
    cudaMemcpy(temp_ha, ha, n*m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(temp_hb, hb, n*m*sizeof(float), cudaMemcpyHostToDevice);

    // calling kernel
    int blockSize = 1024;
    int gridSize = (int)ceil((float)n/blockSize);
    MatrixAddFloatKernel<<<gridSize, blockSize>>>(temp_ha, temp_hb, temp_hd, n, m);
    
    // copying memory to host
    cudaMemcpy(hd, temp_hd, n*m*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(temp_ha);
    cudaFree(temp_hb);
    cudaFree(temp_hd);

    cout << "Result matrix after addition of single point precision numbers in host:\n";
    int t = 0;
    for(int i=0; i<n; i++) {
      for(int j=0; j<m; j++) {
          cout << hc[t] <<  " ";
          t++;
      }
      cout << "\n";
    }

    cout << "\nResult matrix after addition of single point precision numbers in device:\n";
    t = 0;
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            cout << hd[t] <<  " ";
            t++;
        }
        cout << "\n";
    }

    t = 0;
    bool ans = true;
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            if(hc[t] != hd[t]) {
                ans = false;
                break;
            }
            t++;
        }
    }
    if(ans) {
        cout << "\n Matrices are equal\n";
    } else {
        cout << "\n Matrices are unequal\n";
    }

    // Double point precision matrix addition
    cout << "\n\nDouble point precision matrix addition\n\n";

    // allocating memories
    double *had, *hbd, *hcd, *temp_had, *temp_hbd, *temp_hdd, *hdd;
    had = (double*)malloc(n*m * sizeof(double));
    hbd = (double*)malloc(n*m * sizeof(double));
    hcd = (double*)malloc(n*m * sizeof(double));
    hdd = (double*)malloc(n*m * sizeof(double));

    cudaMalloc((void**)&temp_had, n*m * sizeof(double));
    cudaMalloc((void**)&temp_hbd, n*m * sizeof(double));
    cudaMalloc((void**)&temp_hdd, n*m * sizeof(double));

    // assigning values
    for(int i=0; i<n*m; i++) {
        had[i] = ((double)i + totald) / totald;
        hbd[i] = ((double)i*i + totald) / totald;
        hcd[i] = had[i] + hbd[i];
    }

    // copying memory from host to device
    cudaMemcpy(temp_had, had, n*m*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(temp_hbd, hbd, n*m*sizeof(double), cudaMemcpyHostToDevice);

    // calling kernel
    blockSize = 1024;
    gridSize = (int)ceil((double)n/blockSize);
    MatrixAddFloatKernel<<<gridSize, blockSize>>>(temp_had, temp_hbd, temp_hdd, n, m);
    
    // copying memory to host
    cudaMemcpy(hdd, temp_hdd, n*m*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(temp_had);
    cudaFree(temp_hbd);
    cudaFree(temp_hdd);

    cout << "Result matrix after addition of double point precision numbers in host:\n";
    t = 0;
    for(int i=0; i<n; i++) {
      for(int j=0; j<m; j++) {
          cout << hcd[t] <<  " ";
          t++;
      }
      cout << "\n";
    }

    cout << "\nResult matrix after addition of double point precision numbers in device:\n";
    t = 0;
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            cout << hdd[t] <<  " ";
            t++;
        }
        cout << "\n";
    }

    t = 0;
    ans = true;
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            if(hcd[t] != hdd[t]) {
                ans = false;
                break;
            }
            t++;
        }
    }
    if(ans) {
        cout << "\n Matrices are equal\n";
    } else {
        cout << "\n Matrices are unequal\n";
    }
    return 0;
}
