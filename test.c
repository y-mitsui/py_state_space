#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <time.h>
#include <cublas.h>
typedef float KalmanReal;

void mm_mul(KalmanReal *matA,KalmanReal *matB,KalmanReal *matC,int rowsA,int colsA,int colsB);
void mm_mul2(KalmanReal *matA,KalmanReal *matB,KalmanReal *matC,int rowsA,int colsA,int colsB);

void gsl_matrix_print(gsl_matrix_float *m){	
	int i,j;
	for(i=0;i<m->size1;i++){
		for(j=0;j<m->size2;j++){
			printf("%.10lf ",gsl_matrix_float_get(m,i,j));
		}
		puts("");
	}
}
#define DIMENTION_ROWS 11
#define DIMENTION_COLS 11
#define NUM_LOOP 5000
int main(void){
	int i,j;
	cublasStatus stat;
	float *devPtrA,*devPtrB,*devPtrC;
	
	KalmanReal matA[DIMENTION_ROWS*DIMENTION_COLS];
	KalmanReal matB[DIMENTION_ROWS*DIMENTION_COLS];
	KalmanReal matC[DIMENTION_ROWS*DIMENTION_COLS];
	gsl_matrix_float *gMatA=gsl_matrix_float_alloc(DIMENTION_ROWS,DIMENTION_COLS);
	gsl_matrix_float *gMatB=gsl_matrix_float_alloc(DIMENTION_ROWS,DIMENTION_COLS);
	gsl_matrix_float *gMatC=gsl_matrix_float_alloc(DIMENTION_ROWS,DIMENTION_COLS);

	for(i=0;i<DIMENTION_ROWS;i++){
		for(j=0;j<DIMENTION_COLS;j++){
			matA[i*DIMENTION_COLS+j]=i*DIMENTION_COLS+j;
			matB[i*DIMENTION_COLS+j]=-(10+i*DIMENTION_COLS+j);
			matC[i*DIMENTION_COLS+j]=0.0;
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
			gsl_matrix_float_set(gMatA,i,j,matA[i*DIMENTION_COLS+j]);
			gsl_matrix_float_set(gMatB,i,j,matB[i*DIMENTION_COLS+j]);
			gsl_matrix_float_set(gMatC,i,j,matC[i*DIMENTION_COLS+j]);
		}
	}
	clock_t t1 = clock();
	for(i=0;i<NUM_LOOP;i++){
		mm_mul(matA,matB,matC,DIMENTION_ROWS,DIMENTION_COLS,DIMENTION_ROWS);
	}
	printf("1:%lf\n",(double)(clock()-t1) / CLOCKS_PER_SEC);
	/*for(i=0;i<DIMENTION_ROWS;i++){
		for(j=0;j<DIMENTION_COLS;j++){
			printf("%lf ",matC[i*DIMENTION_COLS+j]);
		}
		puts("");
	}*/
	
	cublasInit();
	
	stat = cublasAlloc (DIMENTION_ROWS*DIMENTION_ROWS, sizeof(*gMatA->data), (void**)&devPtrA);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("device memory allocation failed");
        cublasShutdown();
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (DIMENTION_ROWS, DIMENTION_ROWS, sizeof(*gMatA->data), matA, DIMENTION_ROWS, devPtrA, DIMENTION_ROWS);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cublasFree (devPtrA);
        cublasShutdown();
        return EXIT_FAILURE;
    }
    
    stat = cublasAlloc (DIMENTION_ROWS*DIMENTION_ROWS, sizeof(*gMatB->data), (void**)&devPtrB);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("device memory allocation failed");
        cublasShutdown();
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (DIMENTION_ROWS, DIMENTION_ROWS, sizeof(*gMatB->data), matB, DIMENTION_ROWS, devPtrB, DIMENTION_ROWS);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cublasFree (devPtrB);
        cublasShutdown();
        return EXIT_FAILURE;
    }
    
    stat = cublasAlloc (DIMENTION_ROWS*DIMENTION_ROWS, sizeof(*gMatC->data), (void**)&devPtrC);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("device memory allocation failed");
        cublasShutdown();
        return EXIT_FAILURE;
    }
    /*for(i=0;i<DIMENTION_ROWS;i++){
		for(j=0;j<DIMENTION_COLS;j++){
			matC[i*DIMENTION_COLS+j]=0.0;
		}
	}*/
    /*stat = cublasSetMatrix (DIMENTION_ROWS, DIMENTION_ROWS, sizeof(*matC), matC, DIMENTION_ROWS, devPtrC, DIMENTION_ROWS);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cublasFree (devPtrC);
        cublasShutdown();
        return EXIT_FAILURE;
    }*/
    float alpha=1.0;
    float beta=0.0;
	t1 = clock();
	for(i=0;i<NUM_LOOP;i++){
	    cublasSgemm ( 'T',  'T', DIMENTION_ROWS, DIMENTION_ROWS, DIMENTION_ROWS, alpha, devPtrA, DIMENTION_ROWS, devPtrB, DIMENTION_ROWS, beta, devPtrC, DIMENTION_ROWS);
    }
	printf("2:%lf\n",(double)(clock()-t1) / CLOCKS_PER_SEC);

    stat = cublasGetMatrix (DIMENTION_ROWS, DIMENTION_ROWS, sizeof(*matC), devPtrC, DIMENTION_ROWS, matC, DIMENTION_ROWS);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cublasFree (devPtrA);
        cublasShutdown();
        return EXIT_FAILURE;
    }
    
    // Shutdown CUBLAS
    cublasFree (devPtrA);
    cublasFree (devPtrB);
    cublasFree (devPtrC);
    cublasShutdown();
    
	for(i=0;i<DIMENTION_ROWS;i++){
		for(j=0;j<DIMENTION_COLS;j++){
			printf("%lf ",matC[j*DIMENTION_COLS+i]);
		}
		puts("");
	}
	t1 = clock();
	for(i=0;i<NUM_LOOP;i++){
		gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1.0, gMatA, gMatB, 0.0, gMatC);
	}
	printf("3:%lf\n",(double)(clock()-t1) / CLOCKS_PER_SEC);
	puts("gMatC");
	gsl_matrix_print(gMatC);

}


