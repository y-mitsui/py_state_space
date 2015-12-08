
void matrix_mul(KalmanReal *matA,KalmanReal *matB,KalmanReal *matC,int rowsA,int colsA,int colsB){
    int i,j;
    KalmanReal val; 
    
    for (i=0;i < rowsA; i++) {
        for (j=0; j < colsB; j++) {
            val = 0.;
            for (k=0; k<colsA; k++) {
                val += matA[i*colsA+k] * matB[k*colsB+j];
            }
            matC[i,j] = val;
        }
    }
}
