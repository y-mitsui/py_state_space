import sys

cdef extern from "kalman_filter.h":
    cdef struct KalmanFilter:
        pass
        
    KalmanFilter *KalmanFilterInit(double *sample_y, int n_y, int y_dimention, double *arg_A, double *arg_b, double *arg_c, double *arg_x0, int state_dimention, int m)
    double KalmanFilterSolve(KalmanFilter *kalman_filter, double *arg_sigma_Q, double *arg_sigma_w);
    
cdef extern from "stdlib.h":
    void *malloc(size_t size)
    void free(void *ptr)
    
cdef class KalmanFilterWrap:
    cdef KalmanFilter *kalman_filter
    
    def __init__(self, sample, coef_A, coef_b, coef_c, x0):
        #print coef_A,coef_b,coef_c, x0
        
        cdef int n_sample = sample.shape[0]
        cdef int n_dimention = sample.shape[1]
        cdef int i
        cdef int j
        cdef double *sample_ptr = <double *>malloc(sizeof(double) * n_sample * n_dimention)
        cdef double *coef_A_ptr = <double *>malloc(sizeof(double) * coef_A.shape[0] * coef_A.shape[1])
        cdef double *coef_b_ptr = <double *>malloc(sizeof(double) * coef_b.shape[0] * coef_b.shape[1])
        cdef double *coef_c_ptr = <double *>malloc(sizeof(double) * coef_c.shape[0] * coef_c.shape[1])
        cdef double *x0_ptr = <double *>malloc(sizeof(double) * x0.shape[0])
        
        for i in range(n_sample):
            for j in range(n_dimention):
                sample_ptr[i * n_dimention + j] = sample[i][j]
        
        for i in range(coef_A.shape[0]):
            for j in range(coef_A.shape[1]):
                coef_A_ptr[i * coef_A.shape[1] + j] = coef_A[i,j]
                
        for i in range(coef_b.shape[0]):
            for j in range(coef_b.shape[1]):
                coef_b_ptr[i * coef_b.shape[1] + j] = coef_b[i,j]

        for i in range(coef_c.shape[0]):
            for j in range(coef_c.shape[1]):
                coef_c_ptr[i * coef_c.shape[1] + j] = coef_c[i,j]
                
        for i in range(x0.shape[0]):
            x0_ptr[i] = x0[i]
            
        self.kalman_filter = KalmanFilterInit(sample_ptr, n_sample, n_dimention, coef_A_ptr, coef_b_ptr, coef_c_ptr, x0_ptr, coef_A.shape[0], coef_b.shape[1])
        
        free(sample_ptr)
        free(coef_A_ptr)
        free(coef_b_ptr)
        free(coef_c_ptr)
        free(x0_ptr)
    
    def filter(self, sigma_Q, sigma_w):
        cdef double *sigma_Q_ptr = <double *>malloc(sizeof(double) * sigma_Q.shape[0] * sigma_Q.shape[1])
        cdef double *sigma_w_ptr = <double *>malloc(sizeof(double) * sigma_w.shape[0] * sigma_w.shape[1])
        
        for i in range(sigma_Q.shape[0]):
            for j in range(sigma_Q.shape[1]):
                sigma_Q_ptr[i * sigma_Q.shape[1] + j] = sigma_Q[i, j]
        
        for i in range(sigma_w.shape[0]):
            for j in range(sigma_w.shape[1]):
                sigma_w_ptr[i * sigma_w.shape[1] + j] = sigma_w[i, j]
                
        r = KalmanFilterSolve(self.kalman_filter, sigma_Q_ptr, sigma_w_ptr)
        
        free(sigma_Q_ptr)
        free(sigma_w_ptr)
        return r
        
        
