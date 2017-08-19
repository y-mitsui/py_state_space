#! /usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from  scipy import optimize
import matplotlib.pyplot as plt
import sys
from kalman_filter_wrap import KalmanFilterWrap
import datetime

class StateSpaceModel:
    def __init__(self, mat_A, mat_b, mat_c):
        self.mat_A = mat_A
        self.mat_b = mat_b
        self.mat_c = mat_c
        
    def __add__(self, right_obj):
        new_mat_A = np.matrix(np.zeros((self.mat_A.shape[0] + right_obj.mat_A.shape[0], self.mat_A.shape[1] + right_obj.mat_A.shape[1])))
        new_mat_b = np.matrix(np.zeros((self.mat_b.shape[0] + right_obj.mat_b.shape[0], self.mat_b.shape[1] + right_obj.mat_b.shape[1])))
        new_mat_c = np.matrix(np.zeros((self.mat_c.shape[0] + right_obj.mat_c.shape[0], 1)))
    
        new_mat_A[:self.mat_A.shape[0], :self.mat_A.shape[1]] = self.mat_A
        new_mat_A[self.mat_A.shape[0]:, self.mat_A.shape[1]:] = right_obj.mat_A
        new_mat_b[:self.mat_b.shape[0], :self.mat_b.shape[1]] = self.mat_b
        new_mat_b[self.mat_b.shape[0]:, self.mat_b.shape[1]:] = right_obj.mat_b
        new_mat_c[:self.mat_c.shape[0], :] = self.mat_c
        new_mat_c[self.mat_c.shape[0]:, :] = right_obj.mat_c
        
        return StateSpaceModel(new_mat_A, new_mat_b, new_mat_c)
        

def StateSpaceTrend(order):
    mat_A = np.matrix(np.zeros((order, order)))
    if order == 1:
        mat_A[0] = 1
    elif order == 2:
        mat_A[0:2,0:2] = np.matrix([[2,-1],[1,0]])
    elif order == 3:
        mat_A[0:3,0:3] = np.matrix([[3, -3, -1],[1, 0, 0],[0, 1, 0]])
        
    mat_b = np.matrix(np.zeros((order, 1)))
    mat_b[0, 0] = 1
    
    mat_c = np.matrix(np.zeros((order, 1)))
    mat_c[0, 0] = 1
    
    return StateSpaceModel(mat_A, mat_b, mat_c)
    
def StateSpacePriod(priod):
    if priod < 1:
        raise ValueError("invalid priod ", priod)
        
    period_rank = priod - 1
    mat_A = np.matrix(np.zeros((period_rank, period_rank)))
    mat_A[0, 0:period_rank] = np.ones(period_rank) * -1
    mat_A[1:1 + period_rank - 1, :period_rank - 1] = np.diag(np.ones(period_rank - 1))
        
    mat_b = np.matrix(np.zeros((period_rank, 1)))
    mat_b[0, 0] = 1
    
    mat_c = np.matrix(np.zeros((period_rank, 1)))
    mat_c[0, 0] = 1
    
    return StateSpaceModel(mat_A, mat_b, mat_c)
    
u""" 状態空間モデルを解析 """
        
class PyStateSpace:
    u"""
    [CLASSES] 以下の状態空間モデルを取り扱うクラス
    x(k+1) = Ax(k) + bv(k)
    y(k)   = c'x(k) + w(k)
    v ~ N(0,sigma[0])
    w ~ N(0,sigma[1])
    A: m X m numpy.matrix
    b: m X n numpy.matrix
    c: l X 1 numpy.matrix
    """
    def __init__(self, model, opt_min=1e-8, opt_max=1e+8, sigma=None, debug=False):
        u""" [CLASSES] 
        Keyword arguments:
        A -- numpy.matrix 
        b -- numpy.matrix
        c -- numpy.matrix
        sigma -- numpy.ndarray
        """
        
        self.A = model.mat_A
        self.b = model.mat_b
        self.c = model.mat_c
        
        self.sigma = sigma
        self.model = model
        self.opt_min = opt_min
        self.opt_max = opt_max
        
    def filter(self, y, sigma_Q, sigma_w , x0, gamma=100):
        u""" [CLASSES] カルマンフィルタによる状態平均ベクトル・分散共分散の推定
        Return value:
        """
        n_dimention = x0.shape[0]
        self.n_sample = y.shape[0]
        self.state_filter, self.state_predictor = [], []
        self.covariance_filter, self.covariance_predictor = [], []
        self.y_forecast, self.cov_forecast = [], []
        
        self.state_filter.append(x0)
        gamma = sigma_w[0,0]
        self.covariance_filter.append(np.eye(n_dimention) * gamma)
        #constant_coveriance = self.b * sigma_Q * self.b.T
        constant_coveriance = sigma_Q
        for each_y in y:
            #1期先予測  
            self.state_predictor.append(self.A * self.state_filter[-1])
            self.covariance_predictor.append(self.A * self.covariance_filter[-1] * self.A.T + constant_coveriance)
            self.y_forecast.append(self.c.T * self.state_predictor[-1])
            self.cov_forecast.append(self.c.T * self.covariance_predictor[-1] * self.c + sigma_w)
            
            #フィルタリング
            if each_y[0,0] != None:
                gain = self.covariance_predictor[-1] * self.c * ( self.c.T * self.covariance_predictor[-1] * self.c  + sigma_w).I
                self.state_filter.append(self.state_predictor[-1] + gain * ( each_y - self.c.T * self.state_predictor[-1]))
                self.covariance_filter.append((np.matrix(np.eye(n_dimention)) - gain * self.c.T) * self.covariance_predictor[-1])
            else:
                self.state_filter.append(self.state_predictor[-1])
                self.covariance_filter.append(self.covariance_predictor[-1])
        self.state_filter, self.covariance_filter = self.state_filter[1:], self.covariance_filter[1:]
 
        return (self.state_predictor, self.covariance_predictor, self.state_filter, self.covariance_filter, self.y_forecast, self.cov_forecast)

    def AIC(self):
        return -2 * self.log_ff + 2 * (self.A.shape[0])
        
    def logLikelyfood(self, theta, y, x0, context, method):
        
        sigma_Q = np.matrix(np.diag(theta[:self.b.shape[1]]))
        sigma_w = np.matrix(theta[self.b.shape[1]:])

        r = context.filter(sigma_Q, sigma_w)
        if r > 0:
            raise Exception('out!!')
                
        if r != r:
            print sigma_Q
            print sigma_w
            print np.linalg.det(sigma_Q)
            print np.linalg.det(sigma_w)
            #return 1e+100
            #sys.exit(1)
            #print "r_log:{}".format(r_log)
            #print "cov_forecast:{}".format(cov_forecast)
            #print "y_forecast:{}".format(y_forecast)
            #print cov_forecast
            #sys.exit(1)
            raise Exception('r is NaN')
        if r == float("inf"):
            raise Exception('r is inf')
        
        return -r

    def em(self, sample_y, max_iter=50):
        n_sample = len(sample_y)
        n_dimentions = sample_y[0].shape[0]
        x0 = np.matrix(np.random.rand(self.b.shape[0], 1))
        sigma_Q = np.matrix(np.diag(np.random.rand(self.b.shape[0])))
        sigma_w = np.matrix(np.random.rand(n_dimentions))
        for i in range(max_iter):
            print "i", i
            print "sigma_w", sigma_w
            
            print "x0", x0
            self.filter(sample_y, sigma_Q, sigma_w , x0)
            self.smooth()
            
            sigma_Q = np.matrix(np.zeros(sigma_Q.shape))
            coef = 1.0 / (sample_y.shape[0] - 1)
            for j in range(1, n_sample):
                e_zz = self.covariance_smooth[j] + self.state_smooth[j] * self.state_smooth[j].T
                e_zz1 = self.covariance_smooth[j] * self.smooth_gain[j - 1].T + self.state_smooth[j] * self.state_smooth[j - 1].T
                #e_z1z = self.smooth_gain[j - 1].T * self.covariance_smooth[j] + self.state_smooth[j - 1].T * self.state_smooth[j]
                e_z1z = self.covariance_smooth[j - 1] * self.smooth_gain[j].T + self.state_smooth[j -1 ] * self.state_smooth[j].T
                e_z1z1 = self.covariance_smooth[j - 1] + self.state_smooth[j - 1] * self.state_smooth[j - 1].T
                """
                print "self.covariance_smooth[j]", self.covariance_smooth[j]
                print "self.state_smooth[j]", self.state_smooth[j]
                print "self.smooth_gain[j - 1]", self.smooth_gain[j - 1]
                print "e_zz", e_zz
                print "e_zz1", e_zz1
                print "e_z1z", e_z1z
                print "e_z1z1", e_z1z1
                """
                sigma_Q += (e_zz - self.A * e_z1z - e_zz1 * self.A.T + self.A * e_z1z1 * self.A.T) * coef
                #sigma_Q /= sample_y.shape[0] - 1
                #print "sigma_Q", sigma_Q
                #if j == 1:
                #    sys.exit(1)
            print "sigma_Q", sigma_Q
            #sigma_Q = sigma_Q * coef
            #if i == 10:
            #    break
            
            sigma_w = np.zeros(sigma_w.shape)
            for j, each_y in enumerate(sample_y):
                e_zz = self.covariance_smooth[j] + self.state_smooth[j] * self.state_smooth[j].T
                sigma_w += each_y * each_y.T - self.c.T * self.state_smooth[j] * each_y.T - each_y * self.state_smooth[j].T * self.c + self.c.T * e_zz * self.c
            sigma_w /= sample_y.shape[0]
            
            x0 = self.state_smooth[j]
            
        return (sigma_Q, sigma_w)
        
    def mle(self, sample_y, x0, method, init_theta=[]):
        context = KalmanFilterWrap(sample_y, self.A, self.b, self.c, x0)
        
        n_cover2 = (sample_y[0].shape[0] ** 2 - sample_y[0].shape[0]) / 2

        
        args = (sample_y, x0, context, method)
        
        if method == "differential_evolution":
            n_cover2 = 0
            bounds1 = [(self.opt_min, self.opt_max)] * (self.b.shape[1] + sample_y[0].shape[0])
            bounds2 = [(self.opt_min, self.opt_max)] * n_cover2 
            r = optimize.differential_evolution(self.logLikelyfood, args=args,bounds=bounds1 + bounds2,popsize=150,mutation=0.25,recombination=0.25,tol=0.001)
        elif method == "Nelder-Mead" or method == "L-BFGS-B":
            theta = np.random.random(self.b.shape[1] + sample_y[0].shape[0] + n_cover2) * 100
            r = optimize.minimize(self.logLikelyfood, theta, method=method, args=args, bounds=[(self.opt_min, self.opt_max)] * (self.b.shape[1] + 1))
        elif method == "Powell":
            theta = np.random.random(self.b.shape[1] + sample_y[0].shape[0] + n_cover2) * 20
            r = optimize.minimize(self.logLikelyfood,theta,method='Powell',args=args)
        if r.success == False:
            raise Exception('optimize failed', r)
            
        return (r.x,r.fun)
        
    def smooth(self):
        
        self.state_smooth, self.covariance_smooth = [0.] * self.n_sample, [0.] * self.n_sample
        self.state_smooth[-1] = self.state_filter[-1]
        self.covariance_smooth[-1] = self.covariance_filter[-1]
        self.smooth_gain = [None] * self.n_sample
        for idx in range(self.n_sample - 2, -1, -1):
            gain = self.covariance_filter[idx] * self.A.T * self.covariance_predictor[idx + 1].I
            self.state_smooth[idx] = self.state_filter[idx] + gain * ( self.state_smooth[idx+1] - self.state_predictor[idx+1] )
            self.covariance_smooth[idx] = self.covariance_filter[idx] + gain * ( self.covariance_smooth[idx+1] - self.covariance_predictor[idx+1] ) * gain.T
            self.smooth_gain[idx] = gain
            
        return np.array(self.state_smooth), np.array(self.covariance_smooth)

    def fit(self, y, x0=None, repeat=5, mle_method='differential_evolution'):
        if x0 == None:
            x0 = np.matrix(np.zeros((self.b.shape[0],1)))
            x0[0, 0] = np.average(y)
            self.x0 = x0
            
        if True:
            self.sigma_Q, self.sigma_w = self.em(y)
        else:
            parameters = []
            
            parameters.append(self.mle(y, x0, method=mle_method))
            
            for i in range(repeat):
                try:
                    parameters.append(self.mle(y, x0, method=mle_method))
                except Exception as exc:
                    print "Exception Error: {}".format(exc)
                    continue
                #print "i:{0} result:{1}".format(i,parameters[-1])
            if parameters == []:
                 raise ValueError
                 
            sort_index = np.argmin(map(lambda x: x[1], parameters))
            theta, self.log_ff = parameters[sort_index]
            self.log_ff = -self.log_ff
            
            self.sigma_Q = np.matrix(np.diag(theta[:self.b.shape[1]]))
            self.sigma_w = np.matrix(theta[self.b.shape[1]:])
        
        self.filter(y, self.sigma_Q, self.sigma_w, x0)
        return self.smooth()
    
    def forecast(self, n_ahead):
        y_forecast = []
        #state = self.state_predictor[-1]
        state = self.state_smooth[-1]
        
        #print self.sigma_w
        for i in range(n_ahead):
            state = self.A * state
            y_forecast.append(self.c.T * state)
            
        return y_forecast


if __name__ == "__main__":
    u"""
    Example
    """
    def createModel(trend, priod):
        return StateSpaceTrend(trend) + StateSpacePriod(priod)
        
    def createTrendModel(trend):
        return StateSpaceTrend(trend)
    
    np.random.seed(12345)
    model = createModel(2, 4)
    print model.mat_A
    print model.mat_b
    print model.mat_c
    
    np.random.seed(1)
    
    def generateRandomWalk(n_sample, start_date):
        samples = [np.random.randn()]
        date_indexes = [pd.to_datetime(start_date)]
        for i in range(n_sample):
            #samples.append(np.sin(i / 10))
            #samples.append(1.002 * samples[-1] + np.random.randn() + i % 7 * 1e-2)
            samples.append(0.99 * samples[-1] + np.random.randn() * 1.5 + i % 7 * 1e-0)
            date_indexes.append(date_indexes[-1] + datetime.timedelta(days=1))
            
        return pd.TimeSeries(samples, index=date_indexes)
        
    sample = generateRandomWalk(200, "2017-01-01")
    sample.plot()
    plt.show()
    #sys.exit(1)
    sample = np.matrix(sample.as_matrix().reshape(-1, 1))
    sample = (sample - np.average(sample)) / np.std(sample)
    #ar_rank = 3
    #model = ar_model.AR(sample)
    #ar_result = model.fit(ar_rank)
    #ar_coef = ar_result.params[1:ar_rank+1].as_matrix()
    """y_forecast = ar_result.predict(100)
    plt.plot(y_forecast)
    plt.show()
    sys.exit(1)"""

    #model = getModelAR(ar_coef)
    """
    min_aic = float('inf')
    for n_trend in range(1, 4):
        for n_priod in range(2, 14):
            #model = getModel(n_trend, n_priod)
            #model = getModel(2, 4)
            model = createModel(n_trend, n_priod)
            state_space = PyStateSpace(model)
            try:
                state_space.fit(sample, repeat=0, mle_method='L-BFGS-B')
            except Exception as e:
                print e
                continue
            print n_trend, n_priod, " AIC:", state_space.AIC()
            if min_aic > state_space.AIC():
                min_aic = state_space.AIC()
                min_param = (n_trend, n_priod)
    print "min_param", min_param
    print "min_aic", min_aic
    """
    #model = createModel(min_param[0], min_param[1])
    model = createModel(2, 7)
    print "model.mat_b", model.mat_b
    state_space = PyStateSpace(model)
    state_smooth, _ = state_space.fit(sample, repeat=2, mle_method='L-BFGS-B')
    sample_predict = state_space.forecast(20)
    state_smooth = np.array([(model.mat_c.T * ss)[0, 0] for ss in state_smooth])
    sample_predict = np.array([sp[0,0] for sp in sample_predict])

    plt.plot(range(state_smooth.shape[0]),state_smooth, c='g')
    plt.plot(range(sample.shape[0]),[x[0,0] for x in sample], c='b')
    plt.plot(range(sample.shape[0],sample.shape[0]+len(sample_predict)),sample_predict, c='r')
    plt.show()




