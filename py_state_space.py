#! /usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from  scipy import optimize
import math
import matplotlib.pyplot as plt

u""" 状態空間モデルを解析 """

class PyStateSpace:
    u"""
    [CLASSES] 以下の状態空間モデルを取り扱うクラス
    x(k+1) = Ax(k) + bv(k)
    y(k)   = c'x(k) + w(k)
    v ~ N(0,sigma[0])
    w ~ N(0,sigma[1])
    A: m X m numpy.matrix class
    b: m X 1 numpy.matrix class
    c: l X 1 numpy.matrix class
    """
    def __init__(self, A, b, c, debug=False):
        u""" [CLASSES] 
        Keyword arguments:
        A -- numpy.matrix 
        b -- numpy.matrix
        c -- numpy.matrix
        sigma -- numpy.ndarray
        """
        self.A = A
        self.b = b
        self.c = c

    def filter(self, y, sigma_v, sigma_w ,x0):
        u""" [CLASSES] カルマンフィルタによる状態平均ベクトル・分散共分散の推定
        Return value:
        """
        
        n_dimention = x0.shape[0]
        self.state_filter, self.state_predictor = [], []
        self.covariance_filter, self.covariance_predictor = [], []
        self.y_forecast, self.cov_forecast = [], []
        
        self.state_filter.append(x0)
        self.covariance_filter.append(np.ones((n_dimention, n_dimention)) * sigma_w)
        
        for each_y in y:
            #1期先予測
            self.state_predictor.append(self.A * self.state_filter[-1])
            self.covariance_predictor.append(self.A * self.covariance_filter[-1] * self.A.T + sigma_v * (self.b * self.b.T))
            self.y_forecast.append(self.c.T * self.state_predictor[-1])
            self.cov_forecast.append(self.c.T * self.covariance_predictor[-1] * self.c + sigma_w)
            #フィルタリング
            gain = self.covariance_predictor[-1] * self.c / ( self.c.T * self.covariance_predictor[-1] * self.c  + sigma_w)
            self.state_filter.append(self.state_predictor[-1] + gain * ( each_y - self.c.T * self.state_predictor[-1]))
            self.covariance_filter.append((np.eye(n_dimention) - gain * self.c.T) * self.covariance_predictor[-1])
        
        self.state_filter, self.covariance_filter = self.state_filter[1:], self.covariance_filter[1:]
        return (self.state_predictor, self.covariance_predictor, self.state_filter, self.covariance_filter, self.y_forecast, self.cov_forecast)

    def logLikelyfood(self, theta, y, x0):
        sigma_v, sigma_w = np.exp(theta)
        #sigma_v, sigma_w = (theta)
        _, _, _, _, y_forecast, cov_forecast = self.filter(y, sigma_v, sigma_w, x0)
        
        #print np.array(y_forecast)
        y_forecast = np.array([yf[0,0] for yf in y_forecast])
        cov_forecast = np.array([yf[0,0] for yf in cov_forecast])
        
        r = np.sum(np.log(1. / (2. * math.pi * cov_forecast)) - (y - y_forecast)  ** 2 / (2 * cov_forecast))
        print "theta:{} loglike:{}".format(theta,r)
        return -r

    def mle(self, y, x0):
        r = optimize.minimize(self.logLikelyfood,[1,1],method='Powell',args=(y, x0))
        return np.exp(r.x) if r.success else False
        
    def mleBrute(self, y, x0):
        rranges = (slice(1, 1000, 100), slice(1, 10000, 100))
        resbrute = optimize.brute(self.logLikelyfood, rranges, args=(y, x0), full_output=True, finish=optimize.fmin)
        return resbrute[0]
        
    def smooth(self, y):
        self.state_smooth, self.covariance_smooth = [0.] * y.shape[0], [0.] * y.shape[0]
        self.state_smooth[-1] = self.state_filter[-1]
        self.covariance_smooth[-1] = self.covariance_filter[-1]

        for idx in range(y.shape[0]-2,-1,-1):
            gain = self.covariance_filter[idx] * self.A.T * self.covariance_predictor[idx + 1].I
            self.state_smooth[idx] = self.state_filter[idx] + gain * ( self.state_smooth[idx+1] - self.state_predictor[idx+1] )
            self.covariance_smooth[idx] = self.covariance_filter[idx] + gain * ( self.covariance_smooth[idx+1] - self.covariance_predictor[idx+1] ) * gain.T
            
        return np.array(self.state_smooth), np.array(self.covariance_smooth)

    def fit(self, y, x0):
        sigma_v, sigma_w = self.mle(y, x0)
        self.filter(y, sigma_v, sigma_w, x0)
        return self.smooth(y)
    
    def forecast(self, n_ahead):
        y_forecast = []
        state = self.state_predictor[-1]
        for i in range(n_ahead):
            state = self.A * state
            y_forecast.append(self.c.T * state)
            
        return y_forecast[1:]


if __name__ == "__main__":
    u"""
    Example
    """
    df1 = pd.read_csv("ts.txt")
    sample = df1["Sale"].as_matrix()

    A = np.matrix([[2,-1],[1,0]])
    b = np.matrix([[1.],[0]])
    c = np.matrix([[1.],[0]])
    x0 = np.matrix([[0.],[0.]])

    state_space = PyStateSpace(A,b,c)
    state_smooth, _ = state_space.fit(sample, x0)
    sample_predict = state_space.forecast(100)

    state_smooth = np.array([ss[0,0] for ss in state_smooth])
    sample_predict = np.array([sp[0,0] for sp in sample_predict])

    plt.plot(range(state_smooth.shape[0]),state_smooth)
    plt.plot(range(state_smooth.shape[0]),sample)
    plt.plot(range(state_smooth.shape[0],state_smooth.shape[0]+len(sample_predict)),sample_predict)
    plt.show()




