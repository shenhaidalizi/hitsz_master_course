#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define EPS 1e-6
#define BLOCK_SIZE 32

__global__ void mykernel(int *matrix, int matrixRow, int matrixCol, int *kernal, int k, int *result, int resultRow, int resultCol)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < resultRow && j < resultCol)
    {
        int *kp = kernal;
        int total = 0;
        for (int x = 0; x < k; x++)
        {
            for (int y = 0; y < k; y++)
            {
                total += matrix[(i + x) * matrixCol + (j + y)] * (*kp);
                kp++;
            }
        }
        result[i * resultCol + j] = total;
    }
}

void calOnGPU(int *matrix, int matrixRow, int matrixCol, int *kernal, int k, int *result, int resultRow, int resultCol)
{
    dim3 grid, block;
    block.x = BLOCK_SIZE;
    block.y = BLOCK_SIZE;
    // make sure threads are sufficient
    grid.x = resultRow / block.x + 1;
    grid.y = resultCol / block.y + 1;
    int *d_m, *d_k, *d_r;
    cudaMalloc((void **)&d_m, matrixRow * matrixCol * sizeof(int));
    cudaMalloc((void **)&d_k, k * k * sizeof(int));
    cudaMalloc((void **)&d_r, resultRow * resultCol * sizeof(int));
    cudaMemcpy(d_m, matrix, matrixRow * matrixCol * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, kernal, k * k * sizeof(int), cudaMemcpyHostToDevice);
    timeval start, end;
    gettimeofday(&start, NULL);
    mykernel<<<grid, block>>>(d_m, matrixRow, matrixCol, d_k, k, d_r, resultRow, resultCol);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    float elapsed_time = 1000 * (end.tv_sec - start.tv_sec) + (float)(end.tv_usec - start.tv_usec) / 1000;
    printf("elapsed time of GPU: %.2f ms\n", elapsed_time);
    cudaMemcpy(result, d_r, resultRow * resultCol * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_m);
    cudaFree(d_k);
    cudaFree(d_r);
}

void calOnCPU(int *matrix, int matrixRow, int matrixCol, int *kernal, int k, int *result, int resultRow, int resultCol)
{
    memset(result, 0, resultRow * resultCol * sizeof(int));
    timeval start, end;
    gettimeofday(&start, NULL);
    int *rp = result;
    for (int i = 0; i < resultRow; i++)
    {
        for (int j = 0; j < resultCol; j++)
        {
            int *kp = kernal;
            for (int x = 0; x < k; x++)
            {
                for (int y = 0; y < k; y++)
                {
                    (*rp) += matrix[(i + x) * matrixCol + (j + y)] * (*kp);
                    kp++;
                }
            }
            rp++;
        }
    }
    gettimeofday(&end, NULL);
    float elapsed_time = 1000 * (end.tv_sec - start.tv_sec) + (float)(end.tv_usec - start.tv_usec) / 1000;
    printf("elapsed time of CPU: %.2f ms\n", elapsed_time);
}

int judgeResult(int *a, int *b, int row, int col)
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
    int row, col, k;
    printf("input image (row) (col): ");
    scanf("%d %d", &row, &col);
    printf("input convolution kernal size: ");
    scanf("%d", &k);
    int *matrix = (int *)malloc(row * col * sizeof(int));
    int *kernal = (int *)malloc(k * k * sizeof(int));
    for (int i = 0; i < row * col; i++)
    {
        matrix[i] = rand() % 256;
    }
    // assume the kernal has been flipped
    for (int i = 0; i < k * k; i++)
    {
        kernal[i] = rand() % 10;
    }
    int resultRow = row - k + 1;
    int resultCol = col - k + 1;
    int *resultCPU = (int *)malloc(resultRow * resultCol * sizeof(int));
    int *resultGPU = (int *)malloc(resultRow * resultCol * sizeof(int));
    calOnCPU(matrix, row, col, kernal, k, resultCPU, resultRow, resultCol);
    calOnGPU(matrix, row, col, kernal, k, resultGPU, resultRow, resultCol);
    int judge = judgeResult(resultCPU, resultGPU, resultRow, resultCol);
    printf("%sSame.\n", judge == 1 ? "" : "Not ");
    free(matrix);
    free(kernal);
    free(resultCPU);
    free(resultGPU);
    return 0;
}
