#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
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
    double startTime = MPI_Wtime();
    int pid, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    float *A, *B, *C;
    int srow = m / np;
    if (pid == 0)
    {
        // init
        A = (float *)malloc(m * k * sizeof(float));
        B = (float *)malloc(k * n * sizeof(float));
        C = (float *)malloc(m * n * sizeof(float));
        for (int i = 0; i < m * k; i++)
        {
            A[i] = 1.0 * rand() / RAND_MAX;
        }
        for (int i = 0; i < k * n; i++)
        {
            B[i] = 1.0 * rand() / RAND_MAX;
        }
        // send
        MPI_Bcast(B, k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        for (int i = 1; i < np; i++)
        {
            MPI_Send(A + i * srow * k, srow * k, MPI_FLOAT, i,
                     0, MPI_COMM_WORLD);
        }
        // cal
        matrixMatrixMul(A, B, C, srow, k, n);
        // collect
        for (int i = 1; i < np; i++)
        {
            MPI_Recv(C + i * srow * n, srow * n, MPI_FLOAT, i, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        A = (float *)malloc(srow * k * sizeof(float));
        B = (float *)malloc(k * n * sizeof(float));
        C = (float *)malloc(srow * n * sizeof(float));
        MPI_Bcast(B, k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Recv(A, srow * k, MPI_FLOAT, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        matrixMatrixMul(A, B, C, srow, k, n);
        MPI_Send(C, srow * n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
    // output
    if (pid == 0)
    {
        double endTime = MPI_Wtime();
        float res = sumArray(C, m, n);
        printf("sum: %f, time: %lfs\n", res, endTime - startTime);
    }
    free(A);
    free(B);
    free(C);
    MPI_Finalize();
    return 0;
}