
void mm_mul(KalmanReal *matA,KalmanReal *matB,KalmanReal *matC,int rowsA,int colsA,int colsB){
    int i,j,k;
    KalmanReal val; 
    
    for (i=0;i < rowsA; i++) {
        for (j=0; j < colsB; j++) {
            val = 0.;
            for (k=0; k<colsA; k++) {
                val += matA[i*colsA+k] * matB[k*colsB+j];
            }
            matC[i*colsB+j] = val;
        }
    }
}

void mm_mul2(KalmanReal *matA,KalmanReal *matB,KalmanReal *matC,int rowsA,int colsA,int colsB){
    int i,j,k,ib,jb,kb;
    int ibl = 40;
    KalmanReal val; 
    int N=rowsA;
    for (ib=0; ib<N; ib+=ibl)
        for (jb=0; jb<N; jb+=ibl)
            for (kb=0; kb<N; kb+=ibl)
                for (i=ib;i < ib+ibl; i++) {
                    for (j=jb; j < jb+ibl; j++) {
                        val = 0.;
                        for (k=kb; k<kb+ibl; k++) {
                            val += matA[i*colsA+k] * matB[k*colsB+j];
                        }
                        matC[i*colsB+j] = val;
                    }
                }
}
