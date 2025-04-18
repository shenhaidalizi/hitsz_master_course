#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "MMM.h"

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("Need Three parameters!\n");
        return -1;
    }
    int m = strtol(argv[1], NULL, 10);
    int k = strtol(argv[2], NULL, 10);
    int n = strtol(argv[3], NULL, 10);
    time_t startTime = clock();
    float *A = (float *)malloc(m * k * sizeof(float));
    float *B = (float *)malloc(k * n * sizeof(float));
    float *C = (float *)malloc(m * n * sizeof(float));
    for (int i = 0; i < m * k; i++)
    {
        A[i] = 1.0 * rand() / RAND_MAX;
    }
    for (int i = 0; i < k * n; i++)
    {
        B[i] = 1.0 * rand() / RAND_MAX;
    }
    matrixMatrixMul(A, B, C, m, k, n);
    time_t endTime = clock();
    float res = sumArray(C, m, n);
    printf("sum: %f, time: %lfs\n", res, (double)(endTime - startTime) / CLOCKS_PER_SEC);
    free(A);
    free(B);
    free(C);
    return 0;
}