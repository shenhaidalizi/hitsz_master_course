#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define EPS 1e-6
#define BLOCK_SIZE 32

__global__ void transShare(float *matrix, const int row, const int col, float *result)
{
    int bx = blockDim.x * blockIdx.x;
    int by = blockDim.y * blockIdx.y;
    int x1 = bx + threadIdx.x;
    int y1 = by + threadIdx.y;
    __shared__ float smem_matrix[BLOCK_SIZE][BLOCK_SIZE];
    smem_matrix[threadIdx.y][threadIdx.x] = x1 < col && y1 < row ? matrix[y1 * col + x1] : 0;
    __syncthreads();
    int x2 = bx + threadIdx.y;
    int y2 = by + threadIdx.x;
    if (x2 < col && y2 < row)
    {
        result[x2 * row + y2] = smem_matrix[threadIdx.x][threadIdx.y];
    }
}

__global__ void trans(float *matrix, const int row, const int col, float *result)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < row && y < col)
    {
        result[x + y * row] = matrix[y + x * col];
    }
}

void calOnMemorySharedGPU(float *matrix, int row, int col, float *result)
{
    dim3 grid, block;
    block.x = BLOCK_SIZE;
    block.y = BLOCK_SIZE;
    grid.x = col / block.x + 1; // x is for the col
    grid.y = row / block.y + 1; // y is for the row
    float *d_m, *d_r;
    int matrixLength = row * col * sizeof(float);
    cudaMalloc((void **)&d_m, matrixLength);
    cudaMalloc((void **)&d_r, matrixLength);
    cudaMemcpy(d_m, matrix, matrixLength, cudaMemcpyHostToDevice);
    timeval start, end;
    gettimeofday(&start, NULL);
    transShare<<<grid, block>>>(d_m, row, col, d_r);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    float elapsed_time = 1000 * (end.tv_sec - start.tv_sec) + (float)(end.tv_usec - start.tv_usec) / 1000;
    printf("elapsed time of memory shared GPU: %.2f ms\n", elapsed_time);
    cudaMemcpy(result, d_r, matrixLength, cudaMemcpyDeviceToHost);
    cudaFree(d_m);
    cudaFree(d_r);
}

void calOnGPU(float *matrix, int row, int col, float *result)
{
    dim3 grid, block;
    block.x = BLOCK_SIZE;
    block.y = BLOCK_SIZE;
    // make sure threads are sufficient
    grid.x = row / block.x + 1;
    grid.y = col / block.y + 1;
    float *d_m, *d_r;
    int matrixLength = row * col * sizeof(float);
    cudaMalloc((void **)&d_m, matrixLength);
    cudaMalloc((void **)&d_r, matrixLength);
    cudaMemcpy(d_m, matrix, matrixLength, cudaMemcpyHostToDevice);
    timeval start, end;
    gettimeofday(&start, NULL);
    trans<<<grid, block>>>(d_m, row, col, d_r);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    float elapsed_time = 1000 * (end.tv_sec - start.tv_sec) + (float)(end.tv_usec - start.tv_usec) / 1000;
    printf("elapsed time of GPU: %.2f ms\n", elapsed_time);
    cudaMemcpy(result, d_r, matrixLength, cudaMemcpyDeviceToHost);
    cudaFree(d_m);
    cudaFree(d_r);
}

void calOnCPU(float *matrix, int row, int col, float *result)
{
    timeval start, end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            result[i + j * row] = matrix[j + i * col];
        }
    }
    gettimeofday(&end, NULL);
    float elapsed_time = 1000 * (end.tv_sec - start.tv_sec) + (float)(end.tv_usec - start.tv_usec) / 1000;
    printf("elapsed time of CPU: %.2f ms\n", elapsed_time);
}

int judgeResult(float *a, float *b, int row, int col)
{
    int isSame = 1;
    for (int i = 0; i < row; i++)
    {
        if (fabs(a[i] - b[i]) >= EPS)
        {
            isSame = 0;
            break;
        }
    }
    return isSame;
}

int main()
{
    int row, col;
    printf("Input (row) (col): ");
    scanf("%d %d", &row, &col);
    int matrixLength = row * col * sizeof(float);
    float *matrix = (float *)malloc(matrixLength);
    for (int i = 0; i < row * col; i++)
    {
        matrix[i] = 1.0 * rand() / RAND_MAX;
    }
    float *resultCPU = (float *)malloc(matrixLength);
    float *resultGPU = (float *)malloc(matrixLength);
    float *resultGPUShared = (float *)malloc(matrixLength);
    calOnCPU(matrix, row, col, resultCPU);
    calOnGPU(matrix, row, col, resultGPU);
    calOnMemorySharedGPU(matrix, row, col, resultGPUShared);
    int judge1 = judgeResult(resultCPU, resultGPU, row, col);
    printf("GPU and CPU is%sthe same.\n", judge1 == 1 ? " " : " NOT ");
    int judge2 = judgeResult(resultCPU, resultGPUShared, row, col);
    printf("Memory shared GPU and CPU is%sthe same.\n", judge2 == 1 ? " " : " NOT ");
    free(matrix);
    free(resultCPU);
    free(resultGPU);
    free(resultGPUShared);
    return 0;
}
