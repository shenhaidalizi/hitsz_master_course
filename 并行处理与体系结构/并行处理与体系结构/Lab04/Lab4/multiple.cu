#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

void SequentialCalculation(const int &n,
                           const int &m,
                           const std::vector<std::vector<int>> &A,
                           const std::vector<std::vector<int>> &B,
                           std::vector<std::vector<int>> *C)
{
    std::vector<std::vector<int>> B_power, next_B_power;
    std::vector<std::vector<int>> D;
    (*C) = A;
    B_power = B;
    int tmp;
    for (int t = 1; t <= m; t++)
    {
        D = std::vector<std::vector<int>>(n, std::vector<int>(n, 0));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    D[i][j] = (D[i][j] + A[i][k] * B_power[k][j]) % 2;
                }
            }
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                (*C)[i][j] = ((*C)[i][j] + D[i][j]) % 2;
            }
        }
        if (t == m)
            break;
        next_B_power = std::vector<std::vector<int>>(n, std::vector<int>(n, 0));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                    next_B_power[i][j] = (next_B_power[i][j] + B_power[i][k] * B[k][j]) % 2;
            }
        }
        B_power = next_B_power;
    }
}

bool LoadFile(const std::string &input_file_path, int *n, int *m, std::vector<std::vector<int>> *A,
              std::vector<std::vector<int>> *B)
{
    std::ifstream fin(input_file_path.c_str());
    if (!fin.is_open())
    {
        return false;
    }
    fin >> (*n) >> (*m);
    *A = std::vector<std::vector<int>>(*n, std::vector<int>(*n, 0));
    *B = std::vector<std::vector<int>>(*n, std::vector<int>(*n, 0));
    for (int i = 0; i < (*n); i++)
        for (int j = 0; j < (*n); j++)
            fin >> (*A)[i][j];
    for (int i = 0; i < (*n); i++)
        for (int j = 0; j < (*n); j++)
            fin >> (*B)[i][j];
    fin.close();
    return true;
}

void TestAnswerCorrectness(const std::vector<std::vector<int>> &sequential_answer,
                           const std::vector<std::vector<int>> &parallel_answer)
{
    if (sequential_answer.size() != parallel_answer.size())
    {
        std::cout << "Error! The number of sequential_answer and parallel_answer "
                     "is not the same"
                  << std::endl;
        return;
    }
    long long sum_sequential_answer = 0;
    long long sum_parallel_answer = 0;
    int sum_error = 0;
    for (uint i = 0; i < sequential_answer.size(); i++)
    {
        if (sequential_answer[i].size() != parallel_answer[i].size())
        {
            std::cout << "Error! The number of sequential_answer and parallel_answer "
                         "is not the same"
                      << std::endl;
            return;
        }
        for (uint j = 0; j < sequential_answer[i].size(); j++)
        {
            sum_error += abs(sequential_answer[i][j] - parallel_answer[i][j]);
            sum_sequential_answer += sequential_answer[i][j];
            sum_parallel_answer += parallel_answer[i][j];
        }
    }
    std::cout << "sum_sequential_answer = " << sum_sequential_answer << std::endl;
    std::cout << "sum_parallel_answer = " << sum_parallel_answer << std::endl;
    if (sum_error > 0)
    {
        std::cout << "Wrong Answer" << std::endl;
    }
    else
    {
        std::cout << "Correct!!!" << std::endl;
    }
}

// ==============================================================
// ====    Write your functions below this line    ====
// ==============================================================
// ==============================================================

__global__ void matrixAddKernal(int *d_C, int *d_B,
                                int n, int m, int step)
{ // n*m矩阵的%加法，按行划分线程
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n)
    {
        for (int j = 0; j < m; j++)
        {
            d_C[idx * m + j] = (d_C[idx * m + j] + d_B[idx * m + j]) % 2;
        }
        idx += step;
    }
}

__global__ void matrixMultipleKernal(int *d_C, int *d_A, int *d_B,
                                     int n, int m, int w, int step)
{ // n*m和m*w矩阵的%乘法，按行划分线程
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n)
    {
        for (int j = 0; j < w; j++)
        {
            for (int k = 0; k < m; k++)
            {
                d_C[idx * w + j] =
                    (d_C[idx * w + j] + d_A[idx * m + k] * d_B[k * w + j]) % 2;
            }
        }
        idx += step;
    }
}

void ParallelCalculationWithCuda(const int *A, const int *B, int *CPart,
                                 int n, int m, int startCol, int endCol,
                                 int blockPerGrid, int threadPerBlock, int rank)
{
    int colomnLength = endCol - startCol + 1;
    int partMatrixSize = n * colomnLength * sizeof(int);
    int fullMatrixSize = n * n * sizeof(int);
    int BArrayPart[n][colomnLength];
    // 初始化C的片段为A，初始化B的片段
    for (int i = 0; i < n; i++)
    {
        for (int j = startCol; j <= endCol; j++)
        {
            CPart[i * colomnLength + j - startCol] = A[i * n + j];
            BArrayPart[i][j - startCol] = B[i * n + j];
        }
    }
    int *d_CPart, *d_BArrayPart, *d_A, *d_B, *d_D, *d_nextBPartPower;
    cudaMalloc((void **)&d_CPart, partMatrixSize);
    cudaMalloc((void **)&d_BArrayPart, partMatrixSize);
    cudaMalloc((void **)&d_nextBPartPower, partMatrixSize);
    cudaMalloc((void **)&d_A, fullMatrixSize);
    cudaMalloc((void **)&d_B, fullMatrixSize);
    cudaMalloc((void **)&d_D, partMatrixSize);
    cudaMemcpy(d_CPart, CPart, partMatrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_BArrayPart, *BArrayPart, partMatrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, fullMatrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, fullMatrixSize, cudaMemcpyHostToDevice);
    int step = blockPerGrid * threadPerBlock;
    for (int t = 1; t <= m; t++)
    {
        cudaMemset(d_D, 0, partMatrixSize);
        matrixMultipleKernal<<<blockPerGrid, threadPerBlock>>>(
            d_D, d_A, d_BArrayPart, n, n, colomnLength, step);
        cudaDeviceSynchronize();
        // 把D加到C
        matrixAddKernal<<<blockPerGrid, threadPerBlock>>>(
            d_CPart, d_D, n, colomnLength, step);
        if (t == m)
        {
            break;
        }
        // 通过左乘B来迭代B
        cudaMemset(d_nextBPartPower, 0, partMatrixSize);
        matrixMultipleKernal<<<blockPerGrid, threadPerBlock>>>(
            d_nextBPartPower, d_B, d_BArrayPart, n, n, colomnLength, step);
        cudaDeviceSynchronize();
        cudaMemcpy(d_BArrayPart, d_nextBPartPower, partMatrixSize, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(CPart, d_CPart, partMatrixSize, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
    cudaFree(d_BArrayPart);
    cudaFree(d_nextBPartPower);
    cudaFree(d_CPart);
}

void ParallelCalculation(const int *A, const int *B, int *CPart,
                         int n, int m, int startCol, int endCol, int rank)
{
    int colomnLength = endCol - startCol + 1;
    int partMatrixSize = n * colomnLength * sizeof(int);
    int BArrayPart[n][colomnLength];
    // 初始化C的片段为A，初始化B的片段
    for (int i = 0; i < n; i++)
    {
        for (int j = startCol; j <= endCol; j++)
        {
            CPart[i * colomnLength + j - startCol] = A[i * n + j];
            BArrayPart[i][j - startCol] = B[i * n + j];
        }
    }
    for (int t = 1; t <= m; t++)
    {
        int D[n][colomnLength]; // D为 A x B^t
        memset(D, 0, partMatrixSize);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < colomnLength; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    D[i][j] = (D[i][j] + A[i * n + k] * BArrayPart[k][j]) % 2;
                }
            }
        }
        // 把D加到C
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < colomnLength; j++)
            {
                CPart[i * colomnLength + j] = (CPart[i * colomnLength + j] + D[i][j]) % 2;
            }
        }
        if (t == m)
        {
            break;
        }
        // 通过左乘B来迭代B
        int nextBPartPower[n][colomnLength];
        memset(nextBPartPower, 0, partMatrixSize);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < colomnLength; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    nextBPartPower[i][j] = (nextBPartPower[i][j] + B[i * n + k] * BArrayPart[k][j]) % 2;
                }
            }
        }
        memcpy(BArrayPart, nextBPartPower, partMatrixSize);
    }
}

// ==============================================================
// ====    Write your functions above this line    ====
// ==============================================================
// ==============================================================

int main(int argc, char **argv)
{
    int number_of_processes, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double parallel_start_time;
    int number_of_block_in_a_grid;
    int number_of_thread_in_a_block;
    int n, m;
    std::vector<std::vector<int>> A;
    std::vector<std::vector<int>> B;
    if (rank == 0)
    {
        if (argc < 4)
        {
            std::cout << "Error! Please use \"mpiexec -n [process number] "
                         "[--hostfile hostfile] multiple [number_of_block_in_a_grid] [number_of_thread_in_a_block] [data_file_name]\"\n";
            return 1;
        }
        else
        {
            number_of_block_in_a_grid = std::atoi(argv[1]);
            number_of_thread_in_a_block = std::atoi(argv[2]);
            std::string input_file_path = std::string(argv[3]);
            std::cout << "number_of_block_in_a_grid:" << number_of_block_in_a_grid << std::endl;
            std::cout << "number_of_thread_in_a_block:" << number_of_thread_in_a_block << std::endl;
            if (!LoadFile(input_file_path, &n, &m, &A, &B))
            {
                std::cout << "Error! Please check the format of input file\n";
                return 1;
            }
        }
    }
    std::vector<std::vector<int>> parallel_answer;
    if (rank == 0)
    {
        parallel_start_time = MPI_Wtime();
    }

    // ==============================================================
    // ====    Write your implementation below this line    ====
    // ==============================================================
    // ==============================================================
    MPI_Barrier(MPI_COMM_WORLD);
    // 参数在主进程中读取，其他进程需要被广播才有参数值
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&number_of_block_in_a_grid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&number_of_thread_in_a_block, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int *Aarray = (int *)malloc(n * n * sizeof(int));
    int *Barray = (int *)malloc(n * n * sizeof(int));
    int *Carray;
    if (rank == 0)
    {
        // vector转array
        int *AarrayTemp = Aarray, *BarrayTemp = Barray;
        for (int i = 0; i < n; i++)
        {
            memcpy(AarrayTemp, &A[i][0], n * sizeof(int));
            memcpy(BarrayTemp, &B[i][0], n * sizeof(int));
            AarrayTemp += n;
            BarrayTemp += n;
        }
        // 分配结果数组的空间
        Carray = (int *)malloc(n * n * sizeof(int));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(Aarray, n * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Barray, n * n, MPI_INT, 0, MPI_COMM_WORLD);
    // 计算各个进程计算的起始终止列
    int scatterSize = max(n / number_of_processes, 1);
    int startCol = rank * scatterSize;
    int endCol = rank == number_of_processes - 1 ? n - 1 : startCol + scatterSize - 1;
    int realSize = endCol - startCol + 1; // 只有最后一列可能不一样
    int *cPart = (int *)malloc(realSize * n * sizeof(int));
    ParallelCalculationWithCuda(Aarray, Barray, cPart, n, m,
                                startCol, endCol,
                                number_of_block_in_a_grid, number_of_thread_in_a_block, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    // Gather是按行存，所以这里按列每次取一层
    for (int i = 0; i < n; i++)
    {
        MPI_Gather(cPart + i * realSize, scatterSize, MPI_INT,
                   Carray + i * n, scatterSize, MPI_INT, 0, MPI_COMM_WORLD);
    }
    // 处理没有整除的最后几列
    int lastProcessRank = number_of_processes - 1;
    if (rank == lastProcessRank || rank == 0)
    {
        int leftCol = n - scatterSize * number_of_processes;
        if (leftCol > 0)
        {
            if (rank == lastProcessRank)
            {
                int *leftMatrix = (int *)malloc(n * leftCol * sizeof(int));
                for (int i = 0; i < n; i++)
                {
                    for (int j = scatterSize; j < realSize; j++)
                    {
                        leftMatrix[i * leftCol + j - scatterSize] = cPart[i * realSize + j];
                    }
                }
                MPI_Send(leftMatrix, n * leftCol, MPI_INT, 0, 0, MPI_COMM_WORLD);
                free(leftMatrix);
            }
            else if (rank == 0)
            {
                int *leftMatrix = (int *)malloc(n * leftCol * sizeof(int));
                MPI_Recv(leftMatrix, n * leftCol, MPI_INT, lastProcessRank, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0, k = n - leftCol; j < leftCol; j++)
                    {
                        Carray[i * n + k + j] = leftMatrix[i * leftCol + j];
                    }
                }
                free(leftMatrix);
            }
        }
    }
    if (rank == 0)
    { // array转vector
        parallel_answer = std::vector<std::vector<int>>(n, std::vector<int>(n));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                parallel_answer[i][j] = Carray[i * n + j];
            }
        }
    }
    free(cPart);
    free(Aarray);
    free(Barray);
    if (rank == 0)
    {
        free(Carray);
    }

    // ==============================================================
    // ====    Write your implementation above this line    ====
    // ==============================================================
    // ==============================================================
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        double parallel_end_time = MPI_Wtime();
        double parallel_running_time = parallel_end_time - parallel_start_time;
        std::cout << "parallel running time:" << parallel_running_time << std::endl;
        std::vector<std::vector<int>> sequential_answer;
        double sequential_start_time = MPI_Wtime();
        SequentialCalculation(n, m, A, B, &sequential_answer);
        double sequential_end_time = MPI_Wtime();
        double sequential_running_time =
            sequential_end_time - sequential_start_time;
        std::cout << "sequential running time:" << sequential_running_time
                  << std::endl;
        std::cout << "speed up:" << sequential_running_time / parallel_running_time
                  << std::endl;
        TestAnswerCorrectness(sequential_answer, parallel_answer);
    }
    MPI_Finalize();
    return 0;
}