#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>

typedef struct {
    gsl_vector **state_filter, **state_predictor;
    gsl_matrix **covariance_filter, **covariance_predictor;
    gsl_vector **y_forecast;
    gsl_matrix **cov_forecast;
    gsl_matrix *matrix_km[1];
    gsl_matrix *matrix_lk[1];
    gsl_matrix *matrix_ll[1];
    gsl_matrix *matrix_kl[1];
    
    gsl_vector *vector_l[1];
    gsl_vector *vector_k[1];
    
    gsl_vector *y_clone;
    
    gsl_matrix *tmp_matrix[5];
    gsl_vector *tmp_vector[2];
    gsl_matrix *gain;
    int state_dimention;
    int y_dimention;
    int n_y;
    int m;
    gsl_vector **y;
    gsl_matrix *A;
    gsl_matrix *b;
    gsl_matrix *c;
    gsl_vector *x0;
    
}KalmanFilter;

KalmanFilter *KalmanFilterInit(double *sample_y, int n_y, int y_dimention, double *arg_A, double *arg_b, double *arg_c, double *arg_x0, int state_dimention, int m);
double KalmanFilterSolve(KalmanFilter *kalman_filter, double *arg_sigma_Q, double *arg_sigma_w);

