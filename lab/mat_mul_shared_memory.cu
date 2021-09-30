#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define Tile_size 2

int numARows, numAColumns;
int numBRows, numBColumns;
int numCRows, numCColumns;

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
    __shared__ float sA[Tile_size][Tile_size];   // Tile size to store elements in shared memory
    __shared__ float sB[Tile_size][Tile_size];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (((numAColumns - 1) / Tile_size) + 1); k++) {
        //Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        if ((row < numARows) && (threadIdx.x + (k * Tile_size)) < numAColumns) {
            sA[threadIdx.y][threadIdx.x] = A[(row * numAColumns) + threadIdx.x + (k * Tile_size)];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        //Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        if (col < numBColumns && (threadIdx.y + k * Tile_size) < numBRows) {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k * Tile_size) * numBColumns + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < Tile_size; ++j)//Multiplying Elements present in tile
        {
            cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (row < numCRows && col < numCColumns)//Saving Final result into Matrix C
    {
        C[row * numCColumns + col] = cvalue;
    }
}

void printMat(int row, int col, float *Mat) {
    for (int i = 0; i < row * col; i++) {
        printf("%f  ", *(Mat + i));
        if ((i % col) == 0 && i != 0) {
            printf("\n");
        }
    }
}

void matMultiplyOnHost(float *A, float *B, float *C, int numARows,
                       int numAColumns, int numBRows, int numBColumns,
                       int numCRows, int numCColumns) {
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numAColumns; j++) {
            C[i * numCColumns + j] = 0.0;
            for (int k = 0; k < numCColumns; k++) {
                C[i * numCColumns + j] += A[i * numAColumns + k] * B[k * numBColumns + j];
            }
        }
    }
}

int main(int argc, char **argv) {
    float *hostA, *hostB, *hostC;
    float *hostComputedC;
    float *deviceA, *deviceB, *deviceC;

    printf("\nEnter Rows and Columns of A:");
    scanf("%d %d", &numARows, &numAColumns);

    printf("\nEnter Rows and Columns of B:");
    scanf("%d %d", &numBRows, &numBColumns);

    hostA = (float *) malloc(sizeof(float) * numARows * numAColumns);
    hostB = (float *) malloc(sizeof(float) * numBRows * numBColumns);

    int lower = 10.0, upper = 20.0;
    for (int i = 0; i < numARows * numAColumns; i++)
        hostA[i] = (rand() % (upper - lower + 1)) + lower;
    for (int i = 0; i < numBRows * numBColumns; i++)
        hostB[i] = (rand() % (upper - lower + 1)) + lower;

    printf("\nMatrix A Values:\n");
    printMat(numARows, numAColumns, hostA);

    printf("\n\nMatrix B Values:\n");
    printMat(numBRows, numBColumns, hostB);

    numCRows = numARows;
    numCColumns = numBColumns;

    hostC = (float *) malloc(sizeof(float) * numCRows * numCColumns);
    hostComputedC = (float *) malloc(sizeof(float) * numCRows * numCColumns);

    // Allocating GPU memory
    cudaMalloc((void **) &deviceA, sizeof(float) * numARows * numAColumns);
    cudaMalloc((void **) &deviceB, sizeof(float) * numBRows * numBColumns);
    cudaMalloc((void **) &deviceC, sizeof(float) * numCRows * numCColumns);

    // Copy memory to the GPU
    cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);

    // Initialize the grid and block dimensions
    dim3 dimGrid((numCColumns / Tile_size) + 1, (numCRows / Tile_size) + 1, 1);//Number of Blocks required
    dim3 dimBlock(Tile_size, Tile_size, 1);//Number of threads in each block

    matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

    cudaDeviceSynchronize();//To synchronize the device

    // Copy the results in GPU memory back to the CPU
    cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost);

    printf("\nMatrix C From Device\n");
    printMat(numCRows, numCColumns, hostC);//Function Call

    matMultiplyOnHost(hostA, hostB, hostComputedC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    printf("\nMatrix C From Host\n");
    printMat(numCRows, numCColumns, hostComputedC);
    printf("\n\n");

    for (int i = 0; i < numCColumns *
                        numCRows; i++)//Compare both the result matrices 1. MatrixMultiplyonHost 2. MatrixMultiplyonDevice
    {
        if (hostComputedC[i] != hostC[i]) {
            printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / numCColumns,
                   i % numCColumns, hostComputedC[i], hostC[i]);
            break;
        }
    }

    printf("\nNumber of Blocks Created:%d \n", ((numCColumns / Tile_size) + 1) * ((numCColumns / Tile_size) + 1));
    printf("\nNumber of Threads Per Block: %d \n", (Tile_size * Tile_size));

    // Free the GPU memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    //Free the Pointer Memory
    free(hostA);
    free(hostB);
    free(hostC);
    free(hostComputedC);
    return 0;
}
