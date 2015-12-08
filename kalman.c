#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>
#include <Python.h>

typedef double KalmanReal;

#define dmalloc(size) debug_malloc(size,__FILE__,__LINE__)

void* debug_malloc(const size_t size, const char *file,const int line){
    void *r = malloc(size);
    if (r == NULL) {
        fprintf(stderr, "out of memory %s:%d", file, line);
    }
    return r;
}

void gsl_vector_print(gsl_vector *v){	
	int i;
	for(i=0;i<v->size;i++){
		printf("%.10lf ",gsl_vector_get(v,i));
	}
	puts("");
	
}
void gsl_matrix_print(gsl_matrix *m){	
	int i,j;
	for(i=0;i<m->size1;i++){
		for(j=0;j<m->size2;j++){
			printf("%.10lf ",gsl_matrix_get(m,i,j));
		}
		puts("");
	}
}
gsl_vector *gsl_vector_clone(const gsl_vector *src){
	gsl_vector *r=gsl_vector_alloc(src->size);
	gsl_vector_memcpy(r,src);
	return r;
}
gsl_matrix *gsl_matrix_clone(const gsl_matrix *src){
	gsl_matrix *r=gsl_matrix_alloc(src->size1,src->size2);
	gsl_matrix_memcpy(r,src);
	return r;
}
void gsl_matrix_mul_constant(gsl_matrix *a,const double x){
	int i,j;
	for(i=0;i<a->size1;i++){
		for(j=0;j<a->size2;j++){
			gsl_matrix_set(a,i,j,gsl_matrix_get(a,i,j)*x);
		}
	}
}

gsl_matrix *gsl_inverse(gsl_matrix *m){
	int s=0;
	gsl_permutation * p = gsl_permutation_alloc (m->size1);
	gsl_matrix *inv=gsl_matrix_clone(m);
	gsl_matrix *r=gsl_matrix_alloc(inv->size1,inv->size2);
	gsl_linalg_LU_decomp(inv,p,&s);
	if(gsl_linalg_LU_invert(inv,p,r) != GSL_SUCCESS){
	    return NULL;
	}
	gsl_matrix_free(inv);
	gsl_permutation_free(p);
	return r;
}

int gsl_inverse_cholesky(gsl_matrix *src,gsl_matrix *dist){
	gsl_matrix_memcpy(dist,src);
	int r = gsl_linalg_cholesky_decomp(dist);
	if( r == GSL_SUCCESS){
		gsl_linalg_cholesky_invert(dist);
	}
	return r;
}

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
    gsl_vector **y;
    gsl_matrix *A;
    gsl_matrix *b;
    gsl_matrix *c;
    gsl_vector *x0;
    
}Kalman;


gsl_vector *readArgVec(PyObject *arg){
    int num,i;
    gsl_vector *vec;
    PyObject *row;
    
    if((num = PyList_Size(arg)) < 0) return NULL;
    
    vec = gsl_vector_alloc(num);
    for (i=0; i<num; i++){
        row = PyList_GetItem(arg, i);
        double tmp=PyFloat_AsDouble(row);
        gsl_vector_set(vec,i,tmp);
    }
    return vec;
}

gsl_vector **readArgVecs(PyObject *arg){
    int num[2],i;
    gsl_vector **vecs;
    PyObject *row;
    
    if((num[0] = PyList_Size(arg)) < 0) return NULL;
    if((row = PyList_GetItem(arg, 0)) == NULL) return NULL;
    if((num[1] = PyList_Size(row)) < 0) return NULL;
    vecs = dmalloc(sizeof(gsl_vector *) * num[0]);
    for (i=0; i<num[0]; i++){
        row = PyList_GetItem(arg, i);
        vecs[i] = readArgVec(row);
    }
    return vecs;
}



Kalman *kalmanInit(gsl_vector **y,int n_y, int y_dimention, int state_dimention,int m,gsl_matrix *A, gsl_matrix *b, gsl_matrix *c, gsl_vector *x0){
    Kalman *kalman = dmalloc(sizeof(Kalman));
    int i;
    
    kalman->state_filter = dmalloc(sizeof(gsl_vector *) * n_y);
    kalman->covariance_filter = dmalloc(sizeof(gsl_matrix *) * n_y);
    kalman->state_predictor = dmalloc(sizeof(gsl_vector *) * n_y);
    kalman->covariance_predictor = dmalloc(sizeof(gsl_matrix *) * n_y);
    kalman->y_forecast = dmalloc(sizeof(gsl_vector *) * n_y);
    kalman->cov_forecast = dmalloc(sizeof(gsl_matrix *) * n_y);
    
    for(i=0;i<n_y;i++){
        kalman->state_filter[i] = gsl_vector_alloc(state_dimention);
        kalman->state_predictor[i] = gsl_vector_alloc(state_dimention);
        kalman->covariance_filter[i] = gsl_matrix_alloc(state_dimention,state_dimention);
        kalman->covariance_predictor[i] = gsl_matrix_alloc(state_dimention,state_dimention);
        kalman->y_forecast[i] = gsl_vector_alloc(y_dimention);
        kalman->cov_forecast[i] = gsl_matrix_alloc(y_dimention,y_dimention);
    }
    kalman->matrix_km[0]=gsl_matrix_alloc(state_dimention, m);
    kalman->matrix_lk[0]=gsl_matrix_alloc(y_dimention, state_dimention);
    kalman->matrix_ll[0]=gsl_matrix_alloc(y_dimention, y_dimention);
    kalman->matrix_kl[0]=gsl_matrix_alloc(state_dimention, y_dimention);
    
    kalman->vector_l[0] = gsl_vector_alloc(y_dimention);
    kalman->vector_k[0] = gsl_vector_alloc(state_dimention);
    
    kalman->tmp_matrix[0]=gsl_matrix_alloc(state_dimention,state_dimention);
    kalman->tmp_matrix[1]=gsl_matrix_alloc(state_dimention,state_dimention);
    kalman->tmp_matrix[2]=gsl_matrix_alloc(state_dimention,state_dimention);
    kalman->tmp_matrix[3]=gsl_matrix_alloc(state_dimention,state_dimention);
    kalman->tmp_matrix[4]=gsl_matrix_alloc(state_dimention,state_dimention);
    kalman->tmp_vector[0]=gsl_vector_alloc(state_dimention);
    kalman->tmp_vector[1]=gsl_vector_alloc(state_dimention);
    
    kalman->gain=gsl_matrix_alloc(state_dimention,y_dimention);
    kalman->state_dimention = state_dimention;
    kalman->y_dimention = y_dimention;
    kalman->n_y = n_y;
    kalman->y = y;
    kalman->A = A;
    kalman->b = b;
    kalman->c = c;
    kalman->x0 = x0;
    kalman->y_clone = gsl_vector_alloc(y_dimention);
    
    gsl_set_error_handler_off();
    return kalman;
}

void kalmanPredictionStep(Kalman *kalman,gsl_vector *state_filter, gsl_matrix *covariance_filter,gsl_vector *state_predictor, gsl_matrix *covariance_predictor,
                          gsl_matrix *A, gsl_matrix *constant_covariance){
                          
    gsl_blas_dgemv(CblasNoTrans, 1.0, A, state_filter, 0.0, state_predictor);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, covariance_filter, 0.0, kalman->tmp_matrix[0]);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans , 1.0, kalman->tmp_matrix[0], A, 0.0, covariance_predictor);
    
    gsl_matrix_add(covariance_predictor,constant_covariance);
}

int kalmanFilteringStep(Kalman *kalman,gsl_vector *state_predictor, gsl_matrix *covariance_predictor,gsl_vector *state_filter, gsl_matrix *covariance_filter,
                         gsl_vector *each_y, gsl_matrix *c, gsl_matrix *sigma_w){

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, c, covariance_predictor, 0.0, kalman->matrix_lk[0]);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, kalman->matrix_lk[0], c, 0.0, kalman->matrix_ll[0]);
    gsl_matrix_add(kalman->matrix_ll[0], sigma_w);
    gsl_matrix *tmp = gsl_inverse(kalman->matrix_ll[0]);
    if(tmp==NULL){
        fprintf(stderr,"error tmp\n");
        return -1;
    }
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, covariance_predictor, c, 0.0, kalman->matrix_kl[0]);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, kalman->matrix_kl[0], tmp, 0.0, kalman->gain);
    gsl_matrix_free(tmp);
    gsl_blas_dgemv(CblasTrans , 1.0, c, state_predictor, 0.0, kalman->vector_l[0]);
    gsl_vector_memcpy(kalman->y_clone,each_y);
    gsl_vector_sub(kalman->y_clone, kalman->vector_l[0]);
    gsl_blas_dgemv(CblasNoTrans, 1.0, kalman->gain, kalman->y_clone, 0.0, kalman->vector_k[0]);
    gsl_vector_memcpy(state_filter,state_predictor);
    gsl_vector_add(state_filter, kalman->vector_k[0]);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, kalman->gain, c, 0.0, kalman->tmp_matrix[4]);
    gsl_matrix *tmp2 = gsl_matrix_alloc(kalman->state_dimention, kalman->state_dimention);
    gsl_matrix_set_identity(tmp2);
    gsl_matrix_sub(tmp2,kalman->tmp_matrix[4]);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp2, covariance_predictor, 0.0, covariance_filter);
    gsl_matrix_free(tmp2);
    return 1;
}

int kalmanFilter(Kalman *kalman,gsl_vector **y, int n_y, gsl_matrix *A, gsl_matrix *b, gsl_matrix *c, gsl_matrix *sigma_Q, gsl_matrix *sigma_w, gsl_vector *x0){
    int i;
    
    gsl_matrix *covariance_filter0 = gsl_matrix_alloc(kalman->state_dimention, kalman->state_dimention);
    gsl_matrix_set_identity(covariance_filter0);
    gsl_matrix_mul_constant(covariance_filter0, gsl_matrix_get(sigma_w, 0, 0));
    gsl_matrix *constant_covariance = gsl_matrix_alloc(kalman->state_dimention, kalman->state_dimention);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, b, sigma_Q, 0.0, kalman->matrix_km[0]);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, kalman->matrix_km[0], b, 0.0, constant_covariance);
    kalmanPredictionStep(kalman, x0, covariance_filter0, kalman->state_predictor[0], kalman->covariance_predictor[0], A, constant_covariance);
    
    gsl_matrix_free(covariance_filter0);
    gsl_blas_dgemv(CblasTrans, 1.0, c, kalman->state_predictor[0], 0.0, kalman->y_forecast[0]);
    gsl_matrix *tmp=gsl_matrix_alloc(kalman->y_dimention,kalman->state_dimention);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, c, kalman->covariance_predictor[0], 0.0, tmp);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp, c, 0.0, kalman->cov_forecast[0]);
    gsl_matrix_free(tmp);
    gsl_matrix_add(kalman->cov_forecast[0],sigma_w);
    if(kalmanFilteringStep(kalman, kalman->state_predictor[0], kalman->covariance_predictor[0], kalman->state_filter[0], kalman->covariance_filter[0], y[0], c, sigma_w) < 0)
        return -1;
    
    for(i=1;i < n_y;i++) {
        kalmanPredictionStep(kalman, kalman->state_filter[i-1], kalman->covariance_filter[i-1], kalman->state_predictor[i], kalman->covariance_predictor[i], A, constant_covariance);
        //printf("kalman->state_predictor[%d]:%lf\n",i,gsl_vector_get(kalman->state_predictor[i],0));
        gsl_blas_dgemv(CblasTrans, 1.0, c, kalman->state_predictor[i], 0.0, kalman->y_forecast[i]);
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, c, kalman->covariance_predictor[i], 0.0, kalman->matrix_lk[0]);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, kalman->matrix_lk[0], c, 0.0, kalman->cov_forecast[i]);
        gsl_matrix_add(kalman->cov_forecast[i],sigma_w);
        if(kalmanFilteringStep(kalman, kalman->state_predictor[i], kalman->covariance_predictor[i], kalman->state_filter[i], kalman->covariance_filter[i], y[i], c, sigma_w) < 0)
            return -1;
    }
    gsl_matrix_free(constant_covariance);
    
    return 0;
}

gsl_matrix *readArg(PyObject *arg){
    int num[2],i,j;
    gsl_matrix *mat;
    PyObject *row,*col;
    
    if((num[0] = PyList_Size(arg)) < 0) return NULL;
    if((row = PyList_GetItem(arg, 0)) == NULL) return NULL;
    if((num[1] = PyList_Size(row)) < 0) return NULL;
    mat = gsl_matrix_alloc(num[0],num[1]);
    for (i=0; i<num[0]; i++){
        row = PyList_GetItem(arg, i);
        for (j=0; j<num[1]; j++){
            col = PyList_GetItem(row, j);
            double tmp=PyFloat_AsDouble(col);
            gsl_matrix_set(mat,i,j,tmp);
        }
    }
    return mat;
}




PyListObject *convertPyObjectFromMatries(gsl_matrix **mats,int n){
    int i,j,k;
    //puts("f1");
    PyListObject *list = (PyListObject *) PyList_New(n);
    for(i=0; i < n; i++){
        PyListObject *list2 = (PyListObject *) PyList_New(mats[i]->size1);
        for(j=0; j<mats[i]->size1; j++){
            PyListObject *list3 = (PyListObject *) PyList_New(mats[i]->size2);
            for(k=0; k<mats[i]->size2; k++){
                PyList_SET_ITEM(list3, k, Py_BuildValue("f", gsl_matrix_get(mats[i], j, k)));
            }
            PyList_SET_ITEM(list2,j,Py_BuildValue("O",list3));
        }
        PyList_SET_ITEM(list,i,Py_BuildValue("O",list2));
    }
    return list;
}
PyListObject *convertPyObjectFromVectors(gsl_vector **vecs,int n){
    int i,j;
    PyListObject *list = (PyListObject *) PyList_New(n);
    for(i=0; i < n; i++){
        PyListObject *list2 = (PyListObject *) PyList_New(vecs[i]->size);
        for(j=0; j<vecs[i]->size; j++){
            PyList_SET_ITEM(list2, j, Py_BuildValue("f", gsl_vector_get(vecs[i], j)));
        }
        PyList_SET_ITEM(list,i,Py_BuildValue("O",list2));
    }
    return list;
}
double loglikelihood2(int n_y,gsl_vector **y,gsl_vector **y_forecast,gsl_matrix **cov_forecast){
    int i;
    double r=0.0,pipi=2*M_PI;
    
    
    for(i=0;i<n_y;i++){
        double sigma = gsl_matrix_get(cov_forecast[i],0,0);
        double sample = gsl_vector_get(y[i],0);
        double estimate_sample = gsl_vector_get(y_forecast[i],0);
        r += log(1.0/sqrt(pipi*sigma)) - 0.5 * (sample - estimate_sample) * (sample - estimate_sample) / sigma;
    }
    return r;
}
double gsl_det(gsl_matrix *m){
	gsl_permutation * p = gsl_permutation_alloc (m->size1);
	gsl_matrix *lu=gsl_matrix_clone(m);
	int s=0;
	gsl_linalg_LU_decomp (lu,p,&s);           // LU分解
	double n = gsl_linalg_LU_det (lu,s);    // 行列式
	gsl_matrix_free(lu);
	gsl_permutation_free(p);
	
	return n;
}
double loglikelihood(int n_y,gsl_vector **y,gsl_vector **y_forecast,gsl_matrix **cov_forecast){
    int i;
    double r=0.0,pipi=2*M_PI,subr;
    
    gsl_vector *diffX = gsl_vector_alloc(y[0]->size);
	gsl_vector *tmp = gsl_vector_alloc(y[0]->size);
	gsl_matrix *covI = gsl_matrix_alloc(cov_forecast[0]->size1,cov_forecast[0]->size2);
    for(i=0;i<n_y;i++){
        gsl_vector_memcpy(diffX,y[i]);
		gsl_vector_sub(diffX,y_forecast[i]);
		gsl_inverse_cholesky(cov_forecast[i],covI);
        //gsl_matrix *covI = gsl_inverse(cov_forecast[i]);
        
        gsl_blas_dgemv (CblasTrans, 1.0, covI, diffX,0.0,tmp);
    	gsl_blas_ddot (tmp, diffX,&subr);

        r += log(1.0 / (pow(sqrt(pipi),y[0]->size) * sqrt(gsl_det(cov_forecast[i])))) - 0.5 * subr;
        //printf("r:%lf %lf %lf\n",r,gsl_det(cov_forecast[i]),sqrt(gsl_det(cov_forecast[i])));
    }
    return r;
}

static PyObject *kalmanFilterInitInterface(PyObject *self, PyObject *args){
    int n_y, state_dimention, y_dimention, m;
    PyObject *vecY, *arg_A, *arg_b, *arg_c, *arg_x0;
    gsl_vector **y,*x0;
    gsl_matrix *A,*b,*c;
    
    if (! PyArg_ParseTuple( args, "OOOOOiii", &vecY, &arg_A, &arg_b, &arg_c, &arg_x0, &y_dimention, &state_dimention, &m)) return NULL;
    
    if((n_y = PyList_Size(vecY)) < 0) return NULL;
    y = readArgVecs(vecY);
    A = readArg(arg_A);
    b = readArg(arg_b);
    c = readArg(arg_c);
    x0 = readArgVec(arg_x0);
    
    Kalman *kalman = kalmanInit(y, n_y, y_dimention, state_dimention, m, A, b, c, x0);
    return Py_BuildValue("O", PyCapsule_New(kalman,NULL,NULL));
}

static PyObject *kalmanFilterInterface(PyObject *self, PyObject *args){
    Kalman *kalman;
    PyObject *arg_kalman,*arg_sigma_Q, *arg_sigma_w;
    gsl_matrix *sigma_Q, *sigma_w;
    
    if (! PyArg_ParseTuple( args, "OOO", &arg_kalman, &arg_sigma_Q, &arg_sigma_w)) return NULL;
    kalman=PyCapsule_GetPointer(arg_kalman,NULL);
    
    sigma_Q = readArg(arg_sigma_Q);
    sigma_w = readArg(arg_sigma_w);
    
    
    if(kalmanFilter(kalman,kalman->y,kalman->n_y,kalman->A,kalman->b,kalman->c,sigma_Q,sigma_w,kalman->x0) < 0)
        return Py_BuildValue("f", 1.0);
    // memory leak
    /*PyListObject *list = (PyListObject *) PyList_New(2);
    PyList_SET_ITEM(list,0,Py_BuildValue("O",convertPyObjectFromVectors(kalman->state_predictor, kalman->n_y)));
    PyList_SET_ITEM(list,1,Py_BuildValue("O",convertPyObjectFromMatries(kalman->covariance_predictor, kalman->n_y)));
    PyList_SET_ITEM(list,2,Py_BuildValue("O",convertPyObjectFromVectors(kalman->state_filter, kalman->n_y)));
    PyList_SET_ITEM(list,3,Py_BuildValue("O",convertPyObjectFromMatries(kalman->covariance_filter, kalman->n_y)));
    PyList_SET_ITEM(list,0,Py_BuildValue("O",convertPyObjectFromVectors(kalman->y_forecast, kalman->n_y)));
    PyList_SET_ITEM(list,1,Py_BuildValue("O",convertPyObjectFromMatries(kalman->cov_forecast, kalman->n_y)));*/
    /*for(i=0;i<n_y;i++){
        gsl_vector_free(y[i]);
    }
    free(y);*/
    
    gsl_matrix_free(sigma_Q);
    gsl_matrix_free(sigma_w);
    
    double r = loglikelihood2(kalman->n_y,kalman->y,kalman->y_forecast,kalman->cov_forecast);
    //printf("r:%lf\n",r);
    return Py_BuildValue("f", r);
    //return Py_BuildValue("O", list);
}
static PyMethodDef kalmanmethods[] = {
    {"filter", kalmanFilterInterface, METH_VARARGS},
    {"init", kalmanFilterInitInterface, METH_VARARGS},
    {NULL},
};

void initkalman(void)
{
    Py_InitModule("kalman", kalmanmethods);
}

