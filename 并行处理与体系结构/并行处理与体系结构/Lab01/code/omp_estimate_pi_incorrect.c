#include <stdio.h>
#include <omp.h>


#define N 3
void test(int **a ){
	int i,j;
	for( i=0;i<N;i++){
		for(j=0;j<N;j++){
			printf("%d %d \n",a[i*N+j]);
		}
	}
}

int main(int argc, char* argv[]) {
  	int a[3][3]={1,2,3,4,5,6,7,8,9};
	   
	test(&a);
    return 0;
}
