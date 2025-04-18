#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define EPS 1e-6

__global__ void mykernel(float *matrix, float *vector, const int row, const int col, float *result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < row)
    {
        for (int j = 0; j < col; j++)
        {
            result[idx] += matrix[idx * col + j] * vector[j];
        }
        idx += gridDim.x * blockDim.x;
    }
}

void calOnGPU(float *matrix, float *vector, int row, int col, float *result)
{
    float *d_m, *d_v, *d_r;
    int threadsPerBlock = 1024;
    int maybeBlocks = row / threadsPerBlock;
    int blocksPerGrid = maybeBlocks > 128 ? 128 : (maybeBlocks == 0 ? 1 : maybeBlocks);
    cudaMalloc((void **)&d_m, row * col * sizeof(float));
    cudaMalloc((void **)&d_v, col * sizeof(float));
    cudaMalloc((void **)&d_r, row * sizeof(float));
    cudaMemcpy(d_m, matrix, row * col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, vector, col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_r, 0, row * sizeof(float));
    timeval start, end;
    gettimeofday(&start, NULL);
    mykernel<<<blocksPerGrid, threadsPerBlock>>>(d_m, d_v, row, col, d_r);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    float elapsed_time = 1000 * (end.tv_sec - start.tv_sec) + (float)(end.tv_usec - start.tv_usec) / 1000;
    printf("elapsed time of GPU: %.2f ms\n", elapsed_time);
    cudaMemcpy(result, d_r, row * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_m);
    cudaFree(d_v);
    cudaFree(d_r);
}

void calOnCPU(float *matrix, float *vector, int row, int col, float *result)
{
    timeval start, end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            result[i] += matrix[i * col + j] * vector[j];
        }
    }
    gettimeofday(&end, NULL);
    float elapsed_time = 1000 * (end.tv_sec - start.tv_sec) + (float)(end.tv_usec - start.tv_usec) / 1000;
    printf("elapsed time of CPU: %.2f ms\n", elapsed_time);
}

int judgeResult(float *resultCPU, float *resultGPU, int row)
{
    int isSame = 1;
    for (int i = 0; i < row; i++)
    {
        if (fabs(resultCPU[i] - resultGPU[i]) >= EPS)
        {
            isSame = 0;
            printf("%f %f Diff: %lf\n", resultCPU[i], resultGPU[i], resultCPU[i] - resultGPU[i]);
        }
    }
    return isSame;
}

int main()
{
    int row, col;
    printf("input (row) (col): ");
    scanf("%d %d", &row, &col);
    float *matrix = (float *)malloc(row * col * sizeof(float));
    float *vector = (float *)malloc(col * sizeof(float));
    for (int i = 0; i < row * col; i++)
    {
        matrix[i] = 1.0 * rand() / RAND_MAX;
    }
    for (int i = 0; i < col; i++)
    {
        vector[i] = 1.0 * rand() / RAND_MAX;
    }
    float *resultCPU = (float *)malloc(row * sizeof(float));
    float *resultGPU = (float *)malloc(row * sizeof(float));
    calOnCPU(matrix, vector, row, col, resultCPU);
    calOnGPU(matrix, vector, row, col, resultGPU);
    int judge = judgeResult(resultCPU, resultGPU, row);
    printf("%sSame.\n", judge == 1 ? "" : "Not ");
    free(matrix);
    free(vector);
    free(resultCPU);
    free(resultGPU);
    return 0;
}
