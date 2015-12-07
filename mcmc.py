def metropolis(fun,theta,args,maxiter=10000,step_size=1e-7):
    step_size=1
    n_params = theta.shape[0]
    userfun_cur = fun(theta, *args)
    for iter_cur in range(maxiter):
        for i in range(n_params):
            theta_can = theta
            
            tmp = np.random.randn() * step_size
            while theta_can[i] + tmp < 0.0:
                tmp = np.random.randn() * step_size
            theta_can[i] += tmp
            userfun_can = fun(theta_can, *args)
            ratio = np.exp(userfun_can - userfun_cur);
            print "ratio:{}".format(ratio)
            if np.random.random() < ratio:
                theta = theta_can
                userfun_cur = userfun_can
            if iter_cur > 100:
                step_size = 1
            if iter_cur > 500:
                step_size = 1
            if iter_cur % 100 == 0:
                print iter_cur
    return theta

def metropolis2(fun,theta,args,maxiter=10000,step_size=1e-7):
    n_params = theta.shape[0]
    userfun_cur = fun(theta, *args)
    for iter_cur in range(maxiter):
        tmp = np.random.randn(n_params) * step_size
        while any(theta + tmp < 0.0):
            tmp = np.random.randn(n_params) * step_size
        theta_can = theta + tmp
        userfun_can = fun(theta_can, *args)
        ratio = np.exp(userfun_can - userfun_cur);
        print "ratio:{}".format(ratio)
        if np.random.random() < ratio:
            theta = theta_can
            userfun_cur = userfun_can
        if iter_cur > 100:
            step_size = 1000
        if iter_cur > 500:
            step_size = 100
        if iter_cur % 100 == 0:
            print iter_cur
    return theta

def metropolis3(fun,theta,args,maxiter=10000,step_size=1e-7):
    step_size = findReasonableEpsilon(fun,theta,args)
    u = np.log(10 * step_size)
    
    n_params = theta.shape[0]
    userfun_cur = fun(theta, *args)
    
    ########   Dual Averaging ########
    target_accept = 0.8 #目標採択率
    t0 = 10
    gamma = 0.05
    average_H = 0.0
    #################################
    
    for iter_cur in range(maxiter):
        tmp = np.random.randn(n_params) * step_size
        while any(theta + tmp < 0.0):
            tmp = np.random.randn(n_params) * step_size
        theta_can = theta + tmp
        userfun_can = fun(theta_can, *args)
        ratio = np.exp(userfun_can - userfun_cur);
        #print "ratio:{}".format(ratio)
        H_t = target_accept - ratio
        w = 1. / ( iter_cur + t0 )
        average_H = (1 - w) * average_H + w * H_t
        step_size = min(1e+4,np.exp(u - (np.sqrt(iter_cur)/gamma)*average_H))
        if iter_cur % 10 == 0:
            # print dual averaging's parameters
            print "------ %d ------"%(iter_cur)
            print "average_H:{}".format(average_H)
            print "step_size:{}".format(step_size)
                    
        if np.random.random() < ratio:
            theta = theta_can
            userfun_cur = userfun_can
        
    return theta
    
 # step_sizeの初期化
def findReasonableEpsilon(fun,theta,args):
    n_params = theta.shape[0]
    step_size = 1e-20
    userfun_cur = fun(theta, *args)
    theta_can = theta + np.random.randn(n_params) * step_size
    userfun_can = fun(theta_can, *args)
    ratio = np.exp(userfun_can - userfun_cur);
    a = 2 * int( ratio > 0.5) - 1
    while ratio ** a > 2 **-a :
        
        step_size = 2. ** a * step_size
        theta_can = theta + np.random.randn(n_params) * step_size
        userfun_can = fun(theta_can, *args)
        ratio = np.exp(userfun_can - userfun_cur);
            
    return step_size
