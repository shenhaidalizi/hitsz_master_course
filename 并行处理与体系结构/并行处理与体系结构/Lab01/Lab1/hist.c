#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void parallel_histogram(float *array, int n, int *bins, int num_bins, int thread_count)
{
    int i;
    /* Initialize the bins as zero */
    for (i = 0; i < num_bins; i++) {
        bins[i] = 0;
    }
    /* Counting */
    int idx;
#pragma omp parallel for reduction(+:bins[:num_bins]) private(idx)
    for (i = 0; i < n; i++) {
        int val = (int)array[i];
        if (val == num_bins) { /* Ensure 10 numbers go to the last bin */
            idx = num_bins - 1;
        } else {
            idx = val % num_bins;
        }
        bins[idx]++;
    }
}


void serial_histogram(float *array, int n, int *bins, int num_bins)
{
    int i;
    /* Initialize the bins as zero */
    for (i = 0; i < num_bins; i++) {
        bins[i] = 0; 
    }
    /* Counting */
    int idx;
    for (i = 0; i < n; i++) {
        int val = (int)array[i];
        if (val == num_bins) { /* Ensure 10 numbers go to the last bin */
            idx = num_bins - 1;
        } else {
            idx = val % num_bins;
        }
        bins[idx]++;
    }
}

void generate_random_numbers(float *array, int n) 
{
    int i;
    float a = 10.0;
    for(i=0; i<n; ++i)
        array[i] = ((float)rand()/(float)(RAND_MAX)) * a;
}

int main(int argc, char* argv[])
{    
    if(argc!=3){
    	printf("need argument!\n");
	return -1;
    }
    int n = strtol(argv[1], NULL, 10);
    int thread_count = strtol(argv[2], NULL, 10);
    int num_bins = 10;
    float *array;
    int *bins;
    array = (float *)malloc(sizeof(float) * n);
    bins = (int*)malloc(sizeof(int) * num_bins);
    generate_random_numbers(array, n);
    double start,end;
    start = omp_get_wtime();
    serial_histogram(array, n, bins, num_bins);
    end = omp_get_wtime();
    int i;
    printf("Results\n");
    for (i = 0; i < num_bins; i++) {
        printf("bins[%d]: %d\n", i, bins[i]);
    }
    printf("Serial Running time: %f seconds\n", end - start);
    free(bins);
    // parallel
    bins = (int*)malloc(sizeof(int) * num_bins);
    start = omp_get_wtime();
    parallel_histogram(array, n, bins, num_bins, thread_count);
    end = omp_get_wtime();
    printf("Results\n");
    for (i = 0; i < num_bins; i++) {
        printf("bins[%d]: %d\n", i, bins[i]);
    }
    printf("Parallel Running time: %f seconds\n", end - start);
    free(array);
    free(bins);
    return 0;
}

