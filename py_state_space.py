#! /usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from  scipy import optimize
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime as dt
from scipy.spatial.distance import squareform
import sys
from statsmodels.tsa.stattools import acf
import copy
import kalman

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

def getModel2(trend=1,period=7,ar=3):
    A = np.matrix(np.zeros((period+1,period+1)))
    A[2,2:] = np.ones(period - 1) * -1
    A[3:,2:-1] = np.diag(np.ones(period-2))
    A[0:2,0:2] = np.matrix([[2,-1],[1,0]])
    b = np.matrix(np.zeros((period-1+2,2)))
    b[0,0] = 1
    b[2,1] = 1
    c = np.matrix(np.zeros((period-1+2,2)))
    c[0,0] = 1
    c[2,0] = 1
    c[0,1] = 1
    c[2,1] = 1
    return {'A':A,'b':b,'c':c}

def metropolis(fun,theta,args,maxiter=40000,step_size=1e-7):
    step_size=200
    n_params = theta.shape[0]
    userfun_cur = -fun(theta, *args)
    for iter_cur in range(maxiter):
        for i in range(n_params):
            theta_can = list(theta)
            
            tmp = np.random.randn() * step_size
            while theta_can[i] + tmp < 0.0:
                tmp = np.random.randn() * step_size
            theta_can[i] += tmp
            userfun_can = -fun(theta_can, *args)
            ratio = np.exp(userfun_can - userfun_cur);
            #print "ratio:{}".format(ratio)
            if np.random.random() < ratio:
                theta = theta_can
                userfun_cur = userfun_can
        if iter_cur % 100 == 0:
            print iter_cur
            print "theta2:{} logLikelyfood:{}".format(theta,userfun_cur)

    return (theta,-userfun_cur)

def metropolis2(fun,theta,args,maxiter=30000,step_size=1e-7):
    step_size=1
    n_params = theta.shape[0]
    userfun_cur = -fun(theta, *args)
    for iter_cur in range(maxiter):
        theta_can = theta
        tmp = np.random.randn(n_params) * step_size
        while any(theta_can + tmp < 0.0 ):
            tmp = np.random.randn(n_params) * step_size

        theta_can += tmp
        userfun_can = -fun(theta_can, *args)
        ratio = np.exp(userfun_can - userfun_cur);
        if np.random.random() < ratio:
            theta = theta_can
            userfun_cur = userfun_can

        if iter_cur % 100 == 0:
            print iter_cur
            print "theta:{} logLikelyfood:{}".format(theta,userfun_cur)
            print "like2:%.15f"%(userfun_cur)
        if iter_cur > 3000:
            step_size=0.1
    return theta

    
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
    def filter(self, y, sigma_Q, sigma_w ,x0,gamma=100):
        u""" [CLASSES] カルマンフィルタによる状態平均ベクトル・分散共分散の推定
        Return value:
        """
        n_dimention = x0.shape[0]
        self.n_sample = y.shape[0]
        y_dimention = y.shape[1]
        self.state_filter, self.state_predictor = [], []
        self.covariance_filter, self.covariance_predictor = [], []
        self.y_forecast, self.cov_forecast = [], []
        
        self.state_filter.append(x0)
        gamma = sigma_w[0,0]
        self.covariance_filter.append(np.eye(n_dimention) * gamma)
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
            if each_y[0,0] != None:
                #gain = self.covariance_predictor[-1] * self.c / ( self.c.T * self.covariance_predictor[-1] * self.c  + np.eye(y_dimention) * sigma_w[0,0])
                gain = self.covariance_predictor[-1] * self.c * ( self.c.T * self.covariance_predictor[-1] * self.c  + sigma_w).I
                self.state_filter.append(self.state_predictor[-1] + gain * ( each_y - self.c.T * self.state_predictor[-1]))
                self.covariance_filter.append((np.matrix(np.eye(n_dimention)) - gain * self.c.T) * self.covariance_predictor[-1])
            else:
                self.state_filter.append(self.state_predictor[-1])
                self.covariance_filter.append(self.covariance_predictor[-1])
        self.state_filter, self.covariance_filter = self.state_filter[1:], self.covariance_filter[1:]
 
        return (self.state_predictor, self.covariance_predictor, self.state_filter, self.covariance_filter, self.y_forecast, self.cov_forecast)

    def logLikelyfood(self, theta, y, x0, context, method):
        
        #n_cover = (self.b.shape[1] ** 2 - self.b.shape[1]) / 2
        n_cover2 = (y[0].shape[0]** 2 - y[0].shape[0]) / 2
        if method == 'Powell' or method == 'differential_evolution':
            sigma_Q = np.matrix(np.diag(np.exp(theta[:self.b.shape[1]])))#+squareform(theta[self.b.shape[1]:self.b.shape[1] + n_cover]))
            offset1 = self.b.shape[1] + y[0].shape[0]
            sigma_w = np.matrix(np.diag(np.exp(theta[self.b.shape[1]:self.b.shape[1] + y[0].shape[0]])))+squareform(theta[offset1:offset1 + n_cover2])
        else:
            sigma_Q = np.matrix(np.diag(theta[:self.b.shape[1]]))
            sigma_w = np.matrix(theta[self.b.shape[1]:])

        """a=np.sqrt(np.exp(theta[self.b.shape[1]:self.b.shape[1] + y[0].shape[0]]))
        b=theta[offset1:offset1 + n_cover2]
        for i in range(n_cover2):
            parent = int(i / (y[0].shape[0]-1))
            child = i % (y[0].shape[0]-1) + 1
            if abs(b[i]) > a[parent]*a[child]:
                print "fail"
                #sys.exit(1)
        #b=theta[offset1:offset1 + n_cover2]"""
        
        
        if True:
            #y_forecast, cov_forecast = kalman.filter(context,self.A.tolist(),self.b.tolist(),self.c.tolist(),sigma_Q.tolist(),sigma_w.tolist(),x0)
            #y_forecast = np.array([float(yf[0]) for yf in y_forecast])
            #cov_forecast = np.array([cf[0] for cf in cov_forecast])
            r = kalman.filter(context,sigma_Q.tolist(),sigma_w.tolist())
            if r > 0:
                raise Exception('out!!')
        else:
            _, _, _, _, y_forecast, cov_forecast = self.filter(y, sigma_Q, sigma_w, x0)
            r = np.matrix([[0.]])
            r_log = []
            y_forecast[0] = np.matrix(np.average(y_forecast,axis=0))
            #print cov_forecast
            #for x,mean,sigma in zip(y,y_forecast, cov_forecast):
            for i in range(len(y_forecast)):
                x = y[i]
                mean = y_forecast[i]
                sigma = cov_forecast[i]
                y_dimention = x.shape[0]
                #print "x:{}".format(x)
                #print "mean:{}".format(mean)
                #print "sigma:{}".format(sigma)
                
                #sys.exit(1)
                r += np.log(1. / (np.sqrt(2. * math.pi) ** y_dimention * np.sqrt(np.linalg.det(sigma)) ) ) - 0.5 * (x - mean).T * sigma.I * (x - mean)
                r_log.append(r)
            #y_forecast = np.array(y_forecast)

            #cov_forecast = np.array([cf[0,0] for cf in cov_forecast])
            #y_forecast = np.array([yf[0,0] for yf in y_forecast])
            #print y_forecast
            #print y_forecast.shape
            #y_dimention = y_forecast.shape[1]
            #r = np.sum(np.log(1. / np.sqrt(2. * math.pi * cov_forecast)) - (y - y_forecast)  ** 2 / (2 * cov_forecast))
            #r = np.sum(np.log(1. / (np.sqrt(2. * math.pi) ** y_dimention * np.sqrt(numpy.linalg.det(cov_forecast)) ) ) - 0.5 * (y - y_forecast).T * cov_forecast.I * (y - y_forecast))
            r = r[0,0]
        print "theta:{} loglike:{}".format(theta,r)
        if r != r:
            print sigma_Q
            print sigma_w
            print np.linalg.det(sigma_Q)
            print np.linalg.det(sigma_w)
            #return 1e+100
            sys.exit(1)
            print "r_log:{}".format(r_log)
            print "cov_forecast:{}".format(cov_forecast)
            print "y_forecast:{}".format(y_forecast)
            #print cov_forecast
            #sys.exit(1)
            raise Exception('r is NaN')
        return -r

    def mle(self, y, x0, method='Powell'):
        arg_x0 = [x[0] for x in x0]
        arg_y = [x.T.tolist()[0] for x in y]
        context = kalman.init(arg_y,self.A.tolist(),self.b.tolist(),self.c.tolist(), arg_x0, y[0].shape[0],x0.shape[0], self.b.shape[1])
        n_cover = (self.b.shape[1] ** 2 - self.b.shape[1]) / 2
        n_cover2 = (y[0].shape[0] ** 2 - y[0].shape[0]) / 2
        x_min=1e-6
        x_max=2e+8
        
        args = (y, x0, context, method)
        
        n_param = self.b.shape[1] + y[0].shape[0] + n_cover2
        
        if method == "differential_evolution":
            x_min=-15
            x_max=15
            n_cover2 = 0
            bounds1 = [(x_min,x_max)] * (self.b.shape[1] + y[0].shape[0])
            bounds2 = [(-1e+7,1e+7)] * n_cover2 
            r = optimize.differential_evolution(self.logLikelyfood, args=args,bounds=bounds1 + bounds2,popsize=150,mutation=0.25,recombination=0.25)
        elif method == "slsqp":
            theta = np.random.random(self.b.shape[1] + 1) * x_max
            r = optimize.minimize(self.logLikelyfood,theta,method="slsqp",args=args,bounds=[(x_min,x_max)] * (self.b.shape[1] + 1))
        elif method == "Powell":
            theta = np.random.random(self.b.shape[1] + y[0].shape[0] + n_cover2) * 20
            r = optimize.minimize(self.logLikelyfood,theta,method='Powell',args=args)
        else:
            theta = np.random.random(self.b.shape[1] + y[0].shape[0] + n_cover2) * 20
            return metropolis(self.logLikelyfood,theta,args)
            sys.exit(1)
        if r.success == False:
            raise Exception('optimize failed')
        return (r.x,r.fun)
        
    def smooth(self):
        
        self.state_smooth, self.covariance_smooth = [0.] * self.n_sample, [0.] * self.n_sample
        self.state_smooth[-1] = self.state_filter[-1]
        self.covariance_smooth[-1] = self.covariance_filter[-1]
        
        for idx in range(self.n_sample-2,-1,-1):
            gain = self.covariance_filter[idx] * self.A.T * self.covariance_predictor[idx + 1].I
            self.state_smooth[idx] = self.state_filter[idx] + gain * ( self.state_smooth[idx+1] - self.state_predictor[idx+1] )
            self.covariance_smooth[idx] = self.covariance_filter[idx] + gain * ( self.covariance_smooth[idx+1] - self.covariance_predictor[idx+1] ) * gain.T
            
        return np.array(self.state_smooth), np.array(self.covariance_smooth)

    def fit(self, y, x0=None,repeat=1,mle_method='differential_evolution'):
        parameters = []
        
        if x0 == None:
            x0 = np.matrix(np.zeros((self.b.shape[0],1)))

        for i in range(repeat):
            try:
                parameters.append(self.mle(y, x0, method=mle_method))
            except Exception as exc:
                print "Exception Error: {}".format(exc)
                continue
            print "i:{0} result:{1}".format(i,parameters[-1])
        if parameters == []:
             raise ValueError
        print "len(parameters):%d"%(len(parameters))
        theta = parameters[np.argmin(map(lambda x: x[1],parameters))][0]
        
        #n_cover = (self.b.shape[1] ** 2 - self.b.shape[1]) / 2
        
        if mle_method == 'Powell' or mle_method == 'differential_evolution':
            self.sigma_Q = np.matrix(np.diag(np.exp(theta[:self.b.shape[1]])))#+squareform(theta[self.b.shape[1]:self.b.shape[1] + n_cover]))
            offset1 = self.b.shape[1] + y[0].shape[0]
            n_cover2 = (y[0].shape[0] ** 2 - y[0].shape[0]) / 2
            self.sigma_w = np.matrix(np.diag(np.exp(theta[self.b.shape[1]:self.b.shape[1] + y[0].shape[0]])))+squareform(theta[offset1:offset1 + n_cover2])
            #sigma_w = np.matrix(np.exp(theta[self.b.shape[1]:]))
        else:
            self.sigma_Q = np.matrix(np.diag(theta[:self.b.shape[1]]))
            self.sigma_w = np.matrix(theta[self.b.shape[1]:])
        
        self.filter(y, self.sigma_Q, self.sigma_w, x0)
        return self.smooth()
    
    def forecast(self, n_ahead):
        y_forecast = []
        state = self.state_predictor[-1]
        for i in range(n_ahead):
            state = self.A * state
            y_forecast.append(self.c.T * state + np.random.randn() * np.sqrt(self.sigma_w[0,0]))
            
        return y_forecast[1:]


if __name__ == "__main__":
    u"""
    Example
    """
    
    df1 = pd.read_csv("ts2.txt")
    sample = df1[df1["ts1"] > 0][df1["ts2"] > 0].as_matrix()
    #sample = df1.as_matrix()
    sample = sample[-1::-1]
    mdates.strpdate2num('%Y-%m-%d')
    #plt.plot_date(x=[dt.strptime(x[0], '%Y-%m-%d') for x in sample], y=[[x[1],x[2]] for x in sample], fmt="-")
    #plt.show()
    #sys.exit(1)

    sample = np.array([np.matrix([x[1]]).T for x in sample])
    
    """
    sigma = np.matrix([[ 63807496.25393277,  63521784.61916471],[ 63521784.61916471,  61273850.59439901]])
    mean = np.matrix([[ 7003.98867307],[ 7003.98867307]])
    x = sample[0]
        
    print np.log(1. / (np.sqrt(2. * math.pi) ** 2 * np.sqrt(np.linalg.det(sigma)) ) ) - 0.5 * (x - mean).T * sigma.I * (x - mean)
    print 0.5 * (x - mean).T * sigma.I * (x - mean)
    print np.linalg.det(sigma)
    print np.sqrt(np.linalg.det(sigma))
    sys.exit(1)
    """
    
    """df1 = pd.read_csv("ts.txt")
    sample = df1[df1["Sale"] > 0]["Sale"].as_matrix()
    #reverse
    sample = sample[-1::-1]
    sample = np.array([[x] for x in sample])
    lag_acf = acf(sample,nlags=20)
    print "自己相関関数:{}".format(lag_acf)"""

    model = getModel(1,12)
    state_space = PyStateSpace(model)
    state_smooth, _ = state_space.fit(sample,repeat=8,mle_method='Powell')
    sample_predict = state_space.forecast(100)
    state_smooth = np.array([ss[0,0] for ss in state_smooth])
    sample_predict = np.array([sp[0,0] for sp in sample_predict])

    plt.plot(range(state_smooth.shape[0]),state_smooth)
    plt.plot(range(state_smooth.shape[0]),[x[0,0] for x in sample])
    plt.plot(range(state_smooth.shape[0],state_smooth.shape[0]+len(sample_predict)),sample_predict)
    plt.show()




