#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <time.h>
typedef double KalmanReal;

void mm_mul(KalmanReal *matA,KalmanReal *matB,KalmanReal *matC,int rowsA,int colsA,int colsB);
void mm_mul2(KalmanReal *matA,KalmanReal *matB,KalmanReal *matC,int rowsA,int colsA,int colsB);

void gsl_matrix_print(gsl_matrix *m){	
	int i,j;
	for(i=0;i<m->size1;i++){
		for(j=0;j<m->size2;j++){
			printf("%.10lf ",gsl_matrix_get(m,i,j));
		}
		puts("");
	}
}
#define DIMENTION_ROWS 400
#define DIMENTION_COLS 400
#define NUM_LOOP 10
int main(void){
	int i,j;
	KalmanReal matA[DIMENTION_ROWS*DIMENTION_COLS];
	KalmanReal matB[DIMENTION_ROWS*DIMENTION_COLS];
	KalmanReal matC[DIMENTION_ROWS*DIMENTION_COLS];
	gsl_matrix *gMatA=gsl_matrix_alloc(DIMENTION_ROWS,DIMENTION_COLS);
	gsl_matrix *gMatB=gsl_matrix_alloc(DIMENTION_ROWS,DIMENTION_COLS);
	gsl_matrix *gMatC=gsl_matrix_alloc(DIMENTION_ROWS,DIMENTION_COLS);

	for(i=0;i<DIMENTION_ROWS;i++){
		for(j=0;j<DIMENTION_COLS;j++){
			matA[i*DIMENTION_COLS+j]=i*DIMENTION_COLS+j;
			matB[i*DIMENTION_COLS+j]=-(10+i*DIMENTION_COLS+j);
			matC[i*DIMENTION_COLS+j]=-(50+i*DIMENTION_COLS+j);
		}
	}
	/*for(i=0;i<DIMENTION_ROWS;i++){
		for(j=0;j<DIMENTION_COLS;j++){
			printf("%lf ",matA[i*DIMENTION_COLS+j]);
		}
		puts("");
	}*/

	for(i=0;i<DIMENTION_ROWS;i++){
		for(j=0;j<DIMENTION_COLS;j++){
			gsl_matrix_set(gMatA,i,j,matA[i*DIMENTION_COLS+j]);
			gsl_matrix_set(gMatB,i,j,matB[i*DIMENTION_COLS+j]);
			gsl_matrix_set(gMatC,i,j,matC[i*DIMENTION_COLS+j]);
		}
	}
	clock_t t1 = clock();
	for(i=0;i<NUM_LOOP;i++){
		mm_mul2(matA,matB,matC,DIMENTION_ROWS,DIMENTION_COLS,DIMENTION_ROWS);
	}
	printf("1:%lf\n",(double)(clock()-t1) / CLOCKS_PER_SEC);

	/*for(i=0;i<DIMENTION_ROWS;i++){
		for(j=0;j<DIMENTION_COLS;j++){
			printf("%lf ",matC[i*DIMENTION_COLS+j]);
		}
		puts("");
	}*/
	t1 = clock();
	for(i=0;i<NUM_LOOP;i++){
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, gMatA, gMatB, 0.0, gMatC);
	}
	printf("2:%lf\n",(double)(clock()-t1) / CLOCKS_PER_SEC);
	//puts("gMatC");
	//gsl_matrix_print(gMatC);

}


