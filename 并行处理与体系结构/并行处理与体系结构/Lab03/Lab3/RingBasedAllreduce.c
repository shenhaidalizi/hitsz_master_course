#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

void customRingBasedAllreduce(float local_sum, float *global_sum, int op, int pid, int np)
{
    float sum = local_sum;
    float tmp;
    int nextPid = (pid + 1) % np;
    int prePid = (pid - 1 + np) % np;
    // scatter
    if (pid == 0)
    {
        MPI_Send(&sum, 1, MPI_FLOAT, 1, 666, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Recv(&tmp, 1, MPI_FLOAT, prePid, 666, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (op == 1)
        {
            sum += tmp;
        }
        else
        {
            sum = fmax(tmp, sum);
        }
        MPI_Send(&sum, 1, MPI_FLOAT, nextPid, 666, MPI_COMM_WORLD);
    }
    // gather
    if (pid < np - 2)
    {
        MPI_Recv(&tmp, 1, MPI_FLOAT, prePid, 666, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sum = tmp;
        MPI_Send(&sum, 1, MPI_FLOAT, nextPid, 666, MPI_COMM_WORLD);
    }
    else if (pid == np - 2)
    {
        MPI_Recv(&tmp, 1, MPI_FLOAT, prePid, 666, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sum = tmp;
    }
    *global_sum = sum;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Need two parameters!\n");
        return -1;
    }
    int n = strtol(argv[1], NULL, 10);
    int op = strtol(argv[2], NULL, 10);
    if (op != 1 && op != 2)
    {
        printf("The second parameter must be 1 or 2.\n");
        printf("1: SUM, 2: MAX\n");
        return -1;
    }
    float *arr = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
    {
        arr[i] = 1.0 * rand() / RAND_MAX;
    }
    int pid, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int elements_per_process = ceil(n * 1.0 / np);
    int i, init = elements_per_process * pid;
    float local_sum = 0;
    for (i = init; i < init + elements_per_process && i < n; i++)
    {
        local_sum += arr[i];
    }
    { // MPI_Allreduce
        if (pid == 0)
        {
            printf("MPI_Allreduce:\n");
        }
        float global_sum;
        double startTime = MPI_Wtime();
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT,
                      op == 1 ? MPI_SUM : MPI_MAX, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double endTime = MPI_Wtime();
        printf("[MPI] pid = %d, %s = %f\n", pid, op == 1 ? "sum" : "max", global_sum);
        if (pid == 0)
        {
            printf("--------------MPI_Allreduce time = %lfs\n",
                   endTime - startTime);
        }
    }
    { // custom ring-based Allreduce
        if (pid == 0)
        {
            printf("\ncustom ring-based Allreduce:\n");
        }
        float global_sum;
        double startTime = MPI_Wtime();
        customRingBasedAllreduce(local_sum, &global_sum, op, pid, np);
        double endTime = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        printf("[DIY] pid = %d, %s = %f\n", pid, op == 1 ? "sum" : "max", global_sum);
        if (pid == 0)
        {
            printf("custom ring-based Allreduce time = %lfs\n", endTime - startTime);
        }
    }
    free(arr);
    MPI_Finalize();
    return 0;
}