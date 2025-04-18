#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int N;

void serialCompute(double C[][N]){
    double sum;
    double start, end;
    double A[N][N], B[N][N];
    double csum=0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = j*1;
            B[i][j] = i*j+2;
            C[i][j] = j-i*2;
        }
    }
    start = omp_get_wtime(); //start time measurement
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum = 0;
            for (int k=0; k < N; k++) {
                sum += A[i][k]*B[k][j];
            }
            C[i][j] = sum;
        }
    }
    end = omp_get_wtime(); //end time measurement
    printf("Time of serial computation: %lf seconds\n",end-start);
}

void parallelCompute(double C[][N],int thread_count){
    double sum;
    double start, end;
    double A[N][N], B[N][N];
    double csum=0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = j*1;
            B[i][j] = i*j+2;
            C[i][j] = j-i*2;
        }
    }
    start = omp_get_wtime(); //start time measurement
    # pragma omp parallel for num_threads(thread_count) private(sum)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum = 0;
            for (int k=0; k < N; k++) {
                sum += A[i][k]*B[k][j];
            }
            C[i][j] = sum;
        }
    }
    end = omp_get_wtime(); //end time measurement
    printf("Time of parallel computation: %lf seconds\n",end-start);
}

int main(int argc, char *argv[]) {
    if(argc!=3){
	printf("Parameters are required!\n");
    	return -1;
    }
    int thread_count = strtol(argv[1],NULL,10);
    N = strtol(argv[2],NULL,10);
    printf("thread_count: %d, N: %d\n",thread_count,N);
    double C1[N][N],C2[N][N];
    serialCompute(C1);
    parallelCompute(C2,thread_count);
    int isEqual=1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if(C1[i][j]!=C2[i][j]){
            	isEqual=0;
            	break;
	    }
        }
    }
    printf("%sEqual.\n",isEqual?"":"Not ");
    return 0;
}
