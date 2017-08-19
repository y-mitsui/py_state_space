#! /usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import datetime
from scipy import stats

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
    z(n) = Az(n) + w(n)
    x(n) = Cx(n) + v(n)
    z(1) = mu(0) + u
    w ~ N(0, kai)
    v ~ N(0, sigma)
    u ~ N(0, P(0))
    
    A: m X m numpy.matrix
    C: l X 1 numpy.matrix
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
        self.C = model.mat_c.T
        
        self.sigma = sigma
        self.model = model
        self.opt_min = opt_min
        self.opt_max = opt_max
        
    def filter(self, sample_x, kai, sigma , mu0, P0):
        u""" [CLASSES] カルマンフィルタによる状態平均ベクトル・分散共分散の推定
        Return value:
        """
        self.n_sample = len(sample_x)
        n_dimention = mu0.shape[0]
        mu = []
        P = []
        kalman_gain = []
        V = []
        
        kalman_gain.append(P0 * self.C.T * (self.C * P0 * self.C.T + sigma).I)
        mu.append(mu0 + kalman_gain[0] * (sample_x[0] - self.C * mu0))
        mat_I = np.matrix(np.eye(n_dimention))
        V.append((mat_I - kalman_gain[0] * self.C) * P0)
        for each_x in sample_x[1:]:
            P.append(self.A * V[-1] * self.A.T + kai)
            
            kalman_gain.append(P[-1] * self.C.T * (self.C * P[-1] * self.C.T + sigma).I)
            mu.append(self.A * mu[-1] + kalman_gain[-1] * (each_x - self.C * self.A * mu[-1]))
            V.append((mat_I - kalman_gain[-1] * self.C) * P[-1])
            
        self.mu, self.P, self.kalman_gain, self.V = mu, P, kalman_gain, V
          
    def smooth(self):
        self.state_smooth, self.covariance_smooth = [0.] * self.n_sample, [0.] * self.n_sample
        self.state_smooth[-1] = self.mu[-1]
        self.covariance_smooth[-1] = self.V[-1]
        self.smooth_J = [None] * self.n_sample
        self.smooth_J[-1] = self.V[-1] * self.A.T * self.P[-1].I
        for idx in range(self.n_sample - 2, -1, -1):
            self.smooth_J[idx] = self.V[idx] * self.A.T * self.P[idx].I
            self.state_smooth[idx] = self.mu[idx] + self.smooth_J[idx] * ( self.state_smooth[idx + 1] - self.A * self.mu[idx] )
            self.covariance_smooth[idx] = self.V[idx] + self.smooth_J[idx] * ( self.covariance_smooth[idx + 1] - self.P[idx] ) * self.smooth_J[idx].T
        return self.state_smooth

    def logLikelyfood(self, sample_x, sigma, show):
        r = 0.
        for each_x, state in zip(sample_x, self.state_smooth):
            if show:
                print "state", state,
                print "x", each_x[0]
            r += stats.norm.logpdf(each_x[0], (self.C * state)[0, 0], np.sqrt(sigma[0, 0]))
        return r
        
    def em(self, sample_y, max_iter=1000):
        n_sample = len(sample_y)
        n_dimentions = sample_y[0].shape[0]
        
        kai = np.matrix(np.diag(np.random.rand(self.A.shape[0])))
        print kai
        sigma = np.matrix(np.diag(np.random.rand(n_dimentions)))
        sigma[0, 0] = 1
        mu0 = np.matrix(np.random.rand(self.A.shape[0], 1))
        P0 = np.matrix(np.diag(np.random.rand(kai.shape[0])))
        for i in range(max_iter):
            print "i", i
            #print "sigma", sigma
            #print "mu0", mu0
            self.filter(sample_y, kai, sigma , mu0, P0)
            self.smooth()
            
            kai = np.matrix(np.zeros(kai.shape))
            for j in range(1, n_sample):
                e_zz = self.covariance_smooth[j] + self.state_smooth[j] * self.state_smooth[j].T
                e_zz1 = self.covariance_smooth[j] * self.smooth_J[j - 1].T + self.state_smooth[j] * self.state_smooth[j - 1].T
                e_z1z = self.smooth_J[j - 1] * self.covariance_smooth[j].T + self.state_smooth[j - 1] * self.state_smooth[j].T
                e_z1z1 = self.covariance_smooth[j - 1] + self.state_smooth[j - 1] * self.state_smooth[j - 1].T
                """
                print "self.covariance_smooth[j]", self.covariance_smooth[j]
                print "self.state_smooth[j]", self.state_smooth[j]
                print "self.smooth_J[j - 1]", self.smooth_J[j - 1]
                print "e_zz", e_zz
                print "e_zz1", e_zz1
                print "e_z1z", e_z1z
                print "e_z1z1", e_z1z1
                """
                kai += e_zz - self.A * e_z1z - e_zz1 * self.A.T + self.A * e_z1z1 * self.A.T
            kai /= sample_y.shape[0] - 1
            kai = np.minimum(kai, 100.)
            if i % 10 == 0:
                print "kai", kai
                print np.diag(kai)
            
            """
            sigma = np.zeros(sigma.shape)
            for j, each_y in enumerate(sample_y):
                e_zz = self.covariance_smooth[j] + self.state_smooth[j] * self.state_smooth[j].T
                sigma += each_y * each_y.T - self.C * self.state_smooth[j] * each_y.T - each_y * self.state_smooth[j].T * self.C.T + self.C * e_zz * self.C.T
            sigma /= sample_y.shape[0]
            sigma[0, 0] = 1
            """
            
            mu0 = self.state_smooth[0]
            e_zz = self.covariance_smooth[0] + self.state_smooth[0] * self.state_smooth[0].T
            P0 = e_zz - self.state_smooth[0] * self.state_smooth[0].T
            print "logLikelyfood", self.logLikelyfood(sample_y, sigma, i == 200)
            
        return (kai, sigma, mu0, P0)
        
    def fit(self, y, x0=None, repeat=5, mle_method='differential_evolution'):
        self.kai, self.sigma, self.mu0, self.P0 = self.em(y)
        self.filter(y, self.kai, self.sigma, self.mu0, self.P0)
        return self.smooth()
    
    def forecast(self, n_ahead):
        y_forecast = []
        #state = self.state_predictor[-1]
        state = self.state_smooth[-1]
        
        #print self.sigma_w
        for i in range(n_ahead):
            state = self.A * state
            y_forecast.append(self.C * state)
            
        return y_forecast


if __name__ == "__main__":
    u"""
    Example
    """
    def createModel(trend, priod):
        return StateSpaceTrend(trend) + StateSpacePriod(priod)
        
    def createTrendModel(trend):
        return StateSpaceTrend(trend)
    
    model = createModel(2, 4)
    print model.mat_A
    print model.mat_b
    print model.mat_c
    
    np.random.seed(12345)
    
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
    #model = createTrendModel(2)
    print "model.mat_b", model.mat_b
    state_space = PyStateSpace(model)
    state_smooth = state_space.fit(sample, repeat=2, mle_method='L-BFGS-B')
    sample_predict = state_space.forecast(100)
    state_smooth = np.array([(model.mat_c.T * ss)[0, 0] for ss in state_smooth])
    sample_predict = np.array([sp[0,0] for sp in sample_predict])

    plt.plot(range(state_smooth.shape[0]),state_smooth, c='g')
    plt.plot(range(sample.shape[0]),[x[0,0] for x in sample], c='b')
    plt.plot(range(sample.shape[0],sample.shape[0]+len(sample_predict)),sample_predict, c='r')
    plt.show()




