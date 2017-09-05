#! /usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import datetime
from scipy import stats
    
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
    
    def __init__(self, n_state_dimentions):
        u""" [CLASSES] 
        Keyword arguments:
        """
        
        self.n_state_dimentions = n_state_dimentions
        
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
            if any(np.isnan(np.asarray(self.state_smooth[idx]).reshape(-1))):
                print "nan", self.state_smooth[idx]
                sys.exit(1)        
        return self.state_smooth

    def logLikelyfood(self, sample_x, kai, sigma, mu0, show):
        r = 0.
        r += stats.multivariate_normal.logpdf(np.asarray(self.state_smooth[0]).reshape(-1), np.asarray(mu0).reshape(-1) , kai)
        r += stats.norm.logpdf(sample_x[0, 0], (self.C * self.state_smooth[0])[0, 0], np.sqrt(sigma[0, 0]))
        state_prev = self.state_smooth[0]
        for each_x, state in zip(sample_x, self.state_smooth):
            if show:
                print "state", state,
                print "x", each_x[0]
            r += stats.multivariate_normal.logpdf(np.asarray(state).reshape(-1), np.asarray(self.A * state_prev).reshape(-1) ,  kai)
            r += stats.norm.logpdf(each_x[0], (self.C * state)[0, 0], np.sqrt(sigma[0, 0]))
            state_prev = state
        return r
        
    def em(self, sample_y, max_iter=600):
        n_sample = len(sample_y)
        n_dimentions = sample_y[0].shape[0]
        
        self.A = np.matrix(np.random.rand(self.n_state_dimentions, self.n_state_dimentions))
        self.C = np.matrix(np.random.rand(n_dimentions, self.n_state_dimentions))
        kai = np.matrix(np.diag(np.random.rand(self.A.shape[0])))
        sigma = np.matrix(np.diag(np.random.rand(n_dimentions)))
        mu0 = np.matrix(np.random.rand(self.A.shape[0], 1))
        P0 = np.matrix(np.diag(np.random.rand(kai.shape[0])))
        
        for i in range(max_iter):
            print "i", i
            #print "sigma", sigma
            #print "mu0", mu0
            self.filter(sample_y, kai, sigma , mu0, P0)
            self.smooth()
            
            e_z, e_zz, e_zz1, e_z1z, e_z1z1 = [], [], [], [], []
            for j in range(0, n_sample):
                e_z.append(self.state_smooth[j])
                e_zz.append(self.covariance_smooth[j] + self.state_smooth[j] * self.state_smooth[j].T)
                if j != 0:
                    e_zz1.append(self.covariance_smooth[j] * self.smooth_J[j - 1].T + self.state_smooth[j] * self.state_smooth[j - 1].T)
                    e_z1z.append(self.smooth_J[j - 1] * self.covariance_smooth[j].T + self.state_smooth[j - 1] * self.state_smooth[j].T)
                    e_z1z1.append(e_zz[-2])
                else:
                    e_zz1.append(None)
                    e_z1z.append(None)
                    e_z1z1.append(None)

            A_left, A_right = np.matrix(np.zeros(self.A.shape)), np.matrix(np.zeros(self.A.shape))
            for j in range(1, n_sample):
                A_left += e_zz1[j]
                A_right += e_z1z1[j]
            self.A = A_left * A_right.I

            kai = np.matrix(np.zeros(kai.shape))
            for j in range(1, n_sample):
                kai += e_zz[j] - self.A * e_z1z[j] - e_zz1[j] * self.A.T + self.A * e_z1z1[j] * self.A.T
            kai /= sample_y.shape[0] - 1
            
            C_left, C_right = np.matrix(np.zeros(self.C.shape)), np.matrix(np.zeros(e_zz[0].shape))
            for j, each_y in enumerate(sample_y):
                C_left += each_y * e_z[j].T
                C_right += e_zz[j]
            self.C = C_left * C_right.I

            sigma = np.zeros(sigma.shape)
            for j, each_y in enumerate(sample_y):
                sigma += each_y * each_y.T - self.C * self.state_smooth[j] * each_y.T - each_y * self.state_smooth[j].T * self.C.T + self.C * e_zz[j] * self.C.T
            sigma /= sample_y.shape[0]

            mu0 = self.state_smooth[0]
            e_zz0 = self.covariance_smooth[0] + self.state_smooth[0] * self.state_smooth[0].T
            P0 = e_zz0 - self.state_smooth[0] * self.state_smooth[0].T
            #print "logLikelyfood"
            #print self.logLikelyfood(sample_y, kai, sigma, mu0, i == 200)
            if i % 10 == 0:
                print "kai", kai
                print  "sigma", sigma
                print  "A", self.A
                print  "C", self.C
                print "self.state_smooth[-1]", self.state_smooth[-1]

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
    
    np.random.seed(123)
    
    def generateRandomWalk(n_sample, start_date):
        samples = [np.random.randn()]
        date_indexes = [pd.to_datetime(start_date)]
        for i in range(n_sample):
            samples.append(0.99 * samples[-1] + np.random.randn() * 1e-0+ i % 3 * 1)
            date_indexes.append(date_indexes[-1] + datetime.timedelta(days=1))
            
        return pd.Series(samples, index=date_indexes)
        
    sample = generateRandomWalk(200, "2017-01-01")
    sample.plot()
    plt.show()
    sample = np.matrix(sample.as_matrix().reshape(-1, 1))
    state_space = PyStateSpace(4)
    state_smooth = state_space.fit(sample, repeat=2, mle_method='L-BFGS-B')
    sample_predict = state_space.forecast(100)
    state_smooth = np.array([(state_space.C * ss)[0, 0] for ss in state_smooth])
    sample_predict = np.array([sp[0,0] for sp in sample_predict])

    plt.plot(range(state_smooth.shape[0]),state_smooth, c='g')
    plt.plot(range(sample.shape[0]),[x[0,0] for x in sample], c='b')
    plt.plot(range(sample.shape[0],sample.shape[0]+len(sample_predict)),sample_predict, c='r')
    plt.show()




