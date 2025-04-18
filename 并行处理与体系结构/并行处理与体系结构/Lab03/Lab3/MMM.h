void matrixMatrixMul(float *A, float *B, float *C, int m, int k, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float temp = 0;
            for (int a = 0; a < k; a++)
            {
                temp += A[i * k + a] * B[a * n + j];
            }
            C[i * n + j] = temp;
        }
    }
}

float sumArray(float *C, int m, int n)
{
    float res = 0;
    for (int i = 0; i < m * n; i++)
    {
        res += C[i];
    }
    return res;
}