#! /usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from  scipy import optimize
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
import sys
from statsmodels.tsa.stattools import acf
import kalman
import copy

def getModel(trend=1,period=7,ar=3):
    A = np.matrix(np.zeros((period+1,period+1)))
    A[2,2:] = np.ones(period - 1) * -1
    A[3:,2:-1] = np.diag(np.ones(period-2))
    A[0:2,0:2] = np.matrix([[2,-1],[1,0]])
    b = np.matrix(np.zeros((period-1+2,2)))
    b[0,0] = 1
    b[2,1] = 1
    c = np.matrix(np.zeros((period-1+2,1)))
    c[0,0] = 1
    c[2,0] = 1
    return {'A':A,'b':b,'c':c}

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
    def __init__(self, model, sigma=None, debug=False):
        u""" [CLASSES] 
        Keyword arguments:
        A -- numpy.matrix 
        b -- numpy.matrix
        c -- numpy.matrix
        sigma -- numpy.ndarray
        """
        self.A = model['A']
        self.b = model['b']
        self.c = model['c']
        self.sigma = sigma
    def filter(self, y, sigma_Q, sigma_w ,x0):
        u""" [CLASSES] カルマンフィルタによる状態平均ベクトル・分散共分散の推定
        Return value:
        """
        
        n_dimention = x0.shape[0]
        self.n_sample = y.shape[0]
        self.state_filter, self.state_predictor = [], []
        self.covariance_filter, self.covariance_predictor = [], []
        self.y_forecast, self.cov_forecast = [], []
        
        self.state_filter.append(x0)
        print sigma_w
        self.covariance_filter.append(np.eye(n_dimention) * sigma_w[0,0])
        constant_coveriance = self.b * sigma_Q * self.b.T
        for each_y in y:
            #1期先予測  
            self.state_predictor.append(self.A * self.state_filter[-1])
            self.covariance_predictor.append(self.A * self.covariance_filter[-1] * self.A.T + constant_coveriance)
            """if any(np.diag(self.covariance_predictor[-1]) < 0.0):
                a=self.A * self.covariance_filter[-1] * self.A.T
                b= self.b * sigma_Q * self.b.T
                print "----------------- failed -------------------"
                print "np.diag(a):{}".format(np.diag(a))
                print "np.diag(b):{}".format(np.diag(b))
                print "np.diag(self.covariance_filter[-1]):{}".format(np.diag(self.covariance_filter[-1]))
                print "gain * self.c.T :{}".format(gain * self.c.T)
                print "np.diag(gain * self.c.T):{}".format(np.diag(gain * self.c.T))
                print "index:%d"%(i)
                print "fail sigma_Q:{}".format(sigma_Q)
                print "fail sigma_w:{}".format(sigma_w)
                raise Exception('covariance included nagative values')
            self.covariance_predictor[-1].I"""
            
            self.y_forecast.append(self.c.T * self.state_predictor[-1])
            self.cov_forecast.append(self.c.T * self.covariance_predictor[-1] * self.c + sigma_w)
            
            #フィルタリング
            if each_y != None:
                gain = self.covariance_predictor[-1] * self.c / ( self.c.T * self.covariance_predictor[-1] * self.c  + sigma_w)
                self.state_filter.append(self.state_predictor[-1] + gain * ( each_y - self.c.T * self.state_predictor[-1]))
                self.covariance_filter.append((np.matrix(np.eye(n_dimention)) - gain * self.c.T) * self.covariance_predictor[-1])
            else:
                self.state_filter.append(self.state_predictor[-1])
                self.covariance_filter.append(self.covariance_predictor[-1])
        self.state_filter, self.covariance_filter = self.state_filter[1:], self.covariance_filter[1:]
 
        return (self.state_predictor, self.covariance_predictor, self.state_filter, self.covariance_filter, self.y_forecast, self.cov_forecast)

    def logLikelyfood(self, theta, y, x0, context):
        
        #n_cover = (self.b.shape[1] ** 2 - self.b.shape[1]) / 2
        #sigma_Q = np.matrix(np.diag(np.exp(theta[:self.b.shape[1]])))#+squareform(theta[self.b.shape[1]:self.b.shape[1] + n_cover]))
        #sigma_w = np.matrix(np.exp(theta[self.b.shape[1]:]))
        
        sigma_Q = np.matrix(np.diag(theta[:self.b.shape[1]]))
        sigma_w = np.matrix(theta[self.b.shape[1]:])
        parameterA=sigma_w.tolist()
        #print parameterA
        sigma_w_arg =  copy.deepcopy(sigma_w.tolist())
        #print type(sigma_w_arg)
        _, _, _, _, y_forecast, cov_forecast = kalman.filter(context,y.tolist(),self.A.tolist(),self.b.tolist(),self.c.tolist(),sigma_Q.tolist(),parameterA,x0.tolist())
        sys.exit(1)
        #print parameterA
        #print sigma_w
        #_, _, _, _, y_forecast, cov_forecast = self.filter(y, sigma_Q, sigma_w, x0)
        #print np.array([y_forecast[-1][0]])
        y_forecast = np.array([float(yf[0]) for yf in y_forecast])
        cov_forecast = np.array([cf[0] for cf in cov_forecast])
        
        r = np.sum(np.log(1. / (2. * math.pi * cov_forecast)) - (y - y_forecast)  ** 2 / (2 * cov_forecast))
        #print "theta:{} loglike:{}".format(theta,r)
        
        return -r

    def mle(self, y, x0, method='differential_evolution'):
        context = kalman.init(y.shape[0],1,x0.shape[0], self.b.shape[1])
        #n_cover = (self.b.shape[1] ** 2 - self.b.shape[1]) / 2
        x_min=1e-5
        x_max=2e+7
        
        if method == "differential_evolution":
            r = optimize.differential_evolution(self.logLikelyfood, args=(y, x0, context),bounds=[(x_min,x_max)] * (self.b.shape[1] + 1),init='random',popsize=50)
        elif method == "slsqp":
            theta = np.random.random(self.b.shape[1] + 1) * x_max
            r = optimize.minimize(self.logLikelyfood,theta,method="L-BFGS-B",args=(y, x0),bounds=[(x_min,x_max)] * (self.b.shape[1] + 1))
            
        #r = optimize.minimize(self.logLikelyfood,theta,method='Powell',args=(y, x0))
        
        if r.success == False:
            raise Exception('optimize failed')
        return (r.x,r.fun)
        
    def smooth(self):
        
        self.state_smooth, self.covariance_smooth = [0.] * self.n_sample, [0.] * self.n_sample
        self.state_smooth[-1] = self.state_filter[-1]
        self.covariance_smooth[-1] = self.covariance_filter[-1]
        
        for idx in range(self.n_sample-2,-1,-1):
            #print idx
            #flag =  True    
            #try:
            #    self.covariance_predictor[idx].I
            #except Exception as e:
                #print e
                #print idx
            #    flag=False
            #    continue
            #if flag == False:
            #    print "flag is false" 
            #    continue
            
            gain = self.covariance_filter[idx] * self.A.T * self.covariance_predictor[idx + 1].I
            self.state_smooth[idx] = self.state_filter[idx] + gain * ( self.state_smooth[idx+1] - self.state_predictor[idx+1] )
            self.covariance_smooth[idx] = self.covariance_filter[idx] + gain * ( self.covariance_smooth[idx+1] - self.covariance_predictor[idx+1] ) * gain.T
            
        return np.array(self.state_smooth), np.array(self.covariance_smooth)

    def fit(self, y, x0=None,repeat=1,mle_method='differential_evolution'):
        parameters = []
        
        if x0 == None:
            x0 = np.matrix(np.zeros((self.b.shape[0],1)))
        parameters.append(self.mle(y, x0, method=mle_method))
        for i in range(repeat):
            try:
                parameters.append(self.mle(y, x0, method=mle_method))
            except Exception as exc:
                print "Exception Error: {}".format(exc)
                continue
            print "i:{0} result:{1}".format(i,parameters[-1])
        if parameters == []:
             raise ValueError
        theta = parameters[np.argmin(map(lambda x: x[1],parameters))][0]
        
        #n_cover = (self.b.shape[1] ** 2 - self.b.shape[1]) / 2
        sigma_Q = np.matrix(np.diag(theta[:self.b.shape[1]]))#+squareform(theta[self.b.shape[1]:self.b.shape[1] + n_cover]))
        sigma_w = np.matrix(theta[self.b.shape[1]:])
        
        self.filter(y, sigma_Q, sigma_w, x0)
        return self.smooth()
    
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
    sample = df1[df1["Sale"] > 0]["Sale"].as_matrix()
    #reverse
    sample = sample[-1::-1]
    sample = np.array([[x] for x in sample])
    lag_acf = acf(sample,nlags=20)
    print "自己相関関数:{}".format(lag_acf)

    model = getModel(1,12)
    state_space = PyStateSpace(model)
    state_smooth, _ = state_space.fit(sample,repeat=1,mle_method='differential_evolution')
    sample_predict = state_space.forecast(100)

    state_smooth = np.array([ss[0,0] for ss in state_smooth])
    sample_predict = np.array([sp[0,0] for sp in sample_predict])

    plt.plot(range(state_smooth.shape[0]),state_smooth)
    plt.plot(range(state_smooth.shape[0]),sample)
    #c = np.matrix([[1.],[0],[0.],[0],[0],[0],[0],[0]])
    #plt.plot(range(state_smooth.shape[0]),[x[0] for x in state_space.state_filter])
    plt.plot(range(state_smooth.shape[0],state_smooth.shape[0]+len(sample_predict)),sample_predict)
    plt.show()




