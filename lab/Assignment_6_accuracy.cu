// Rishabh Agarwal - 18JE0676
#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

// kernel functions

// compare single point and double point
__global__ void CompareKernel(float *da, double *db) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        *da = 20.234;
        *db = 20.234;
    }
}

// single precision operations
__global__ void SinglePrecisionOperationsKernel(float *fres_add, float *fres_sub, float *fres_mul, float *fres_div) {
    float f1 = 45.238; 
    float f2 = 26.547;
    *fres_add = f1 + f2;
    *fres_sub = f1 - f2;
    *fres_mul = f1 * f2;
    *fres_div = f1 / f2;
}

// double precision operations
__global__ void DoublePrecisionOperationsKernel(double *dres_add, double *dres_sub, double *dres_mul, double *dres_div) {
    double d1 = 74.112;
    double d2 = 55.656;
    *dres_add = d1 + d2;
    *dres_sub = d1 - d2;
    *dres_mul = d1 * d2;
    *dres_div = d1 / d2;
}

// main()

int main() {

    float host_f = 20.234, device_f;
    double host_d = 20.234, device_d;

    float *da;
    double *db;

    cudaMalloc(&da, sizeof(float));
    cudaMalloc(&db, sizeof(double));

    CompareKernel<<<1, 32>>>(da, db);
    cudaMemcpy(&device_f, da, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&device_d, db, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);

    cout << "Single point float precision number in host: " << host_f << "\n";
    cout << "Single point float precision number in device: " << device_f << "\n";
    cout << ((host_f == device_f) ? "Equal\n" : "Unequal\n");
    cout << "Double point float precision number in host: " << host_d << "\n";
    cout << "Double point float precision number in device: " << device_d << "\n";
    cout << ((host_d == device_d) ? "Equal\n" : "Unequal\n");
    
    // host and device storing variables
    float f1 = 45.238, f2 = 26.547;
    float device_f_add, device_f_sub, device_f_mul, device_f_div;
    double d1 = 74.112, d2 = 55.656;
    double device_d_add, device_d_sub, device_d_mul, device_d_div;

    // pointers to send in kernel
    float *fres_add, *fres_sub, *fres_mul, *fres_div;
    double *dres_add, *dres_sub, *dres_mul, *dres_div;
    
    // allocation memories
    cudaMalloc(&fres_add, sizeof(float));
    cudaMalloc(&fres_sub, sizeof(float));
    cudaMalloc(&fres_mul, sizeof(float));
    cudaMalloc(&fres_div, sizeof(float));
    cudaMalloc(&dres_add, sizeof(double));
    cudaMalloc(&dres_sub, sizeof(double));
    cudaMalloc(&dres_mul, sizeof(double));
    cudaMalloc(&dres_div, sizeof(double));
    
    // calling kernel
    SinglePrecisionOperationsKernel<<<1, 32>>>(
        fres_add, fres_sub, fres_mul, fres_div
    );
    DoublePrecisionOperationsKernel<<<1, 32>>>(
        dres_add, dres_sub, dres_mul, dres_div
    );

    // copying back to host    
    cudaMemcpy(&device_f_add, fres_add, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&device_f_sub, fres_sub, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&device_f_mul, fres_mul, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&device_f_div, fres_div, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&device_d_add, dres_add, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&device_d_sub, dres_sub, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&device_d_mul, dres_mul, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&device_d_div, dres_div, sizeof(double), cudaMemcpyDeviceToHost);

    // freeing memory
    cudaFree(fres_add);
    cudaFree(fres_sub);
    cudaFree(fres_mul);
    cudaFree(fres_div);
    cudaFree(dres_add);
    cudaFree(dres_sub);
    cudaFree(dres_mul);
    cudaFree(dres_div);
    
    cout << "\nNow performing single point precision operations:\n\n";
    
    cout << "Single point precision addition in host: " << f1+f2 << "\n";
    cout << "Single point precision addition in device: " << device_f_add << "\n";
    if(device_f_add == f1 + f2) {
        cout << "Equal\n\n";
    } else {
        cout << "Unequal\n\n";
    }

    cout << "Single point precision subtraction in host: " << f1-f2 << "\n";
    cout << "Single point precision subtraction in device: " << device_f_sub << "\n";
    if(device_f_sub == f1 - f2) {
        cout << "Equal\n\n";
    } else {
        cout << "Unequal\n\n";
    }
    
    cout << "Single point precision multiplication in host: " << f1*f2 << "\n";
    cout << "Single point precision multiplication in device: " << device_f_mul << "\n";
    if(device_f_mul == f1 * f2) {
        cout << "Equal\n\n";
    } else {
        cout << "Unequal\n\n";
    }
    
    cout << "Single point precision division in host: " << f1/f2 << "\n";
    cout << "Single point precision division in device: " << device_f_div << "\n"; 
    if(device_f_div == f1 / f2) {
        cout << "Equal\n\n";
    } else {
        cout << "Unequal\n\n";
    }

    cout << "Now performing double point precision operations:\n\n";
    
    cout << "Double point precision addition in host: " << d1+d2 << "\n";
    cout << "Double point precision addition in device: " << device_d_add << "\n";
    if(device_d_add == d1 + d2) {
        cout << "Equal\n\n";
    } else {
        cout << "Unequal\n\n";
    }

    cout << "Double point precision subtraction in host: " << d1-d2 << "\n";
    cout << "Double point precision subtraction in device: " << device_d_sub << "\n";
    if(device_d_sub == d1 - d2) {
        cout << "Equal\n\n";
    } else {
        cout << "Unequal\n\n";
    }

    cout << "Double point precision multiplication in host: " << d1*d2 << "\n";
    cout << "Double point precision multiplication in device: " << device_d_mul << "\n";
    if(device_d_mul == d1 * d2) {
        cout << "Equal\n\n";
    } else {
        cout << "Unequal\n\n";
    }

    cout << "Double point precision division in host: " << d1/d2 << "\n";
    cout << "Double point precision division in device: " << device_d_div << "\n";
    if(device_d_div == d1 / d2) {
        cout << "Equal\n\n";
    } else {
        cout << "Unequal\n\n";
    }

    return 0;
}
