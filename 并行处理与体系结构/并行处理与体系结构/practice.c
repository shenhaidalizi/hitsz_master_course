int Trap(double a, double b, double n, double * global_result_p){
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    h = (b - a) / n;
    approx = (f(a) + f(b)) / 2.0;
    double local_n = n / thread_count;
    double local_a = a + my_rank * local_n * h;
    double local_b = local_a + local_n * h;
    for(int i = 1; i <= n - 1; i++){
        x_i = a + i * h;
        approx += f(x_i);
    }
    approx =- h * approx;
}

int main(){
    
    #pragma omp parallel num_threads(thread_count)
    Trap(a, b, n, &global_result);
}

double my_pi(){
    double factor = 1.0;
    double sum = 0.0;
    #pragma omp parallel for reduce(+:sum) private(factor) shared(n)
    for(int k = 0; k < n; k++){
        if(k % 2 == 0){
            sum += factor / (2 * k + 1);
        }
        else{
            sum += -factor / (2 * k + 1);
        }
    }
}

int matrix_mul(int ** A, int * B, int n, int m){
    #pragma omp parallel for num_threads(thread_count) default(none) private(i, j) shared(A, x, y, m, n)
    for(int i = 0; i < m; i++){
        y[i] = 0.0;
        for(int j = 0; j < n; j++){
            y[i] += A[i][j] * x[j];
        }
    }
}

#pragma omp parallel for collapse(2)
for(int i = 0; i < 4; i++){
    for(int j = 0; j < 5; j++){
        omp_get_thread_num();
    }
}

int a = 1; //shared
void foo(){
    int b = 2, c = 3; // c shared
    #pragma omp parallel private(b)
    { //b firstprivate
        int d = 4; //d firstprivate
        #pragma omp task
        {
            int e = 5; //e private
        }
    }
}

int fibo(int n){
    if(n < 2)return n;
    if(n < 30)return serial_fibo(n);

    #pragma omp task shared(x) if(n > 30)
    {
        x = fibo(n - 1);
    }
    #pragma omp task shared(y) if(n > 30)
    {
        y = fibo(n - 2);
    }
    #pragma omp taskwait
    return x + y;
}

my_pointer = listhead;
#pragma omp parallel
{
    #pragma omp single
    {
        while(my_pointer){
            #pragma omp task firstprivate(my_pointer)
            {
                do_independent_work(my_pointer);
            }
        }
        
    }
    my_pointer = my_pointer->next;
}

__global__ float my_kernel(){

}

__global__ void vecAdd(float *A, float *B, float *C, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n)C[idx] = A[idx] + B[idx];
    return;
}

void vecAdd(float *A, float *B, float *C, int n){
    int size = n * sizeof(float);
    float *a_d, *b_d, *c_d;

    //STEP 1
    cudaMalloc((void**) &a_d, size);
    cudaMalloc((void**) &b_d, size);
    cudaMalloc((void**) &c_d, size);

    cudaMemcpy(a_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, B, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(c_d, C, size, cudaMemcpyHostToDevice);

    //STEP 2
    vecAdd<<<ceil(n/block.x), block>>>(a_d, b_d, c_d, n);

    //step 3
    cudaMemcpy(C, c_d, size, cudaMemcpyDeviceToHost);
}

int main(){
    int n;
    int *c;
    int *g;
    c = (int*)malloc(sizeof(int) * n);
    cudaMalloc((void**) &g, sizeof(int) * n);
    cudaMemset(g, 0, sizeof(int) * n);
    cudaMemcpy(g, c, sizeof(int) * n, cudaMemcpyDeviceToHost);

    free(c);
    cudaFree(g);

}

__global__ void mykernel(int *a){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] = threadIdx.x;
}

int main(){
    int dimx = 16;
    int numbytes = dimx * sizeof(int);
    int *c, *g;
    c = (int*) malloc(numbytes);
    cudaMalloc((void**) &g, numbytes);
    cudaMemset(g, 0, numbytes);
    dim3 grid, block;
    block.x = 4;
    grid = dimx / block.x;
    mykernel<<<grid, block>>>(g);
    cudaMemcpy(c, g, numbytes, cudaMemcpyDeviceToHost);

}

__global__ void PictureKernel(float *d_Pin, float *d_Pout, int n, int m){
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(Row < m && Col < n){
        d_Pout[Row * n + col] = 2 * d_Pin[Row * n + col];
    }
}

int main(){
    
}

__global__ void MulOnGPU(float *M, float *N, float *P, int width){
    int local_i = blockIdx.y * blockDim.y + threadIdx.y;
    int local_j = blockIdx.x * blockDim.x + threadIdx.x;

    if(local_i < width && local_j < width){
        float pvalue = 0.0;
        for(int k = 0; k < width; k++){
            pvalue += M[local_i * width + k] * N[k * width + local_j];
            P[local_i * width + local_j] = pvalue;
        }
    }
}


#define TILE_WIDTH 16
__global__ void mulkernel(float *md, float *nd, float *pd, int width){
    __shared__ float mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float pv = 0.0;
    for(int m = 0; m < width / TILE_WIDTH; m++){
        mds[ty][tx] = md[row * width + m * TILE_WIDTH + tx];
        nds[ty][tx] = nd[(m * TILE_WIDTH + ty) * width + col];
        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++){
            pv += mds[ty][k] * nd[k][tx];
        }
        __syncthreads();
    }
    pd[row * width + col] = pv;
}


int main(){
    int n, myid, nump, i;
    double mypi, pi, h, sum, x;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_CPMM_WORLD, &nump);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    while(1){
        if(myid == 0){
            scanf("%d", &n);
        }
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(n == 0)break;
        else{
            h = 1.0 / (double)n;
            sum = 0.0;
            for(i = myid + 1; i <= n; i+= nump){
                x = h *((double)i - 0.5);
                sum += (4.0 / (1.0 + x * x));
            }
            mypi = h * sum;
            MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if(myid == 0){
                printf("%f", pi);
            }
        }
    }
}

int main(){
    int m, n, myid, nump, srow, i;
    double *a, *x, *y;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nump);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if(myid == 0){
        scanf("%d%d", &m, &n);
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    srow = m / nump;
    if(myid == 0){
        a = (double*)malloc(sizeof(double) * m * n);
        x = (double*)malloc(sizeof(double) * n);
        y = (double*)malloc(sizeof(double) * m);

        MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for(i = 1; i < nump; i++){
            MPI_Send(a + i * srow * n, srow * n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }

        for(i = 1; i < nump; i++){
            MPI_Recv(y + i * srow, srow, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else{
        a = (double*)malloc(sizeof(double) * srow * n);
        x = (double*)malloc(sizeof(double) * n);
        y = (double*)malloc(sizeof(double) * srow);

        MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Recv(a, srow * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MAT_MUL();

        MPI_Send(y, srow, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    for(i = myid; i < m; i += nump){
        double y[i] = 0.0;
        for(j = 0; j < n; j++){
            y[i] += 
        }
    }
}

#pragma omp parallel num_threads(thread_count) default(none) private(i, yp, j, sum) shared(A, x, y, m, n)
#pragma omp for
for(int i = myrank; i < m; i++){
    yp = 0.0;
    for(int j = 0; j < n; j++){
        yp += A[i][j] * x[j];
    }
    y[i] = yp;
}