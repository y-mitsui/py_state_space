library(dlm)

df_train = read.csv("ts.txt")

shop1 <- df_train[df_train$Sale > 0,]
sortlist <- order(shop1$Date)
data <- ts(shop1[sortlist,]$Sale,frequency = 365,start=c(2013,1,1))

build.4 <- function(theta){
    dlmModPoly(order=2,dV=exp(theta[1]),dW=c(0,0))+
    #dlmModTrig(s=301,q=1, dV=0,dW=0)+
    dlmModSeas(fr=12,dW=c(0,rep(0,10)),dV=0)
}

print("optimize parameters")
fit.4 <- dlmMLE(
data,
parm=dlmMLE(data,parm=c(1),build.4,method="Nelder-Mead")$par,
build.4,
method="BFGS"
)
print("--------------")

DLM.4 <- build.4(fit.4$par)

# Step3
# カルマンフィルター
Filt.4 <- dlmFilter(data, DLM.4)

# Step4
# スムージング
Smooth.4 <- dlmSmooth(Filt.4)
plot(data, col=1, type="l", lwd=1)
lines(dropFirst(Smooth.4$s)[, 1] , col=2, lwd=2)
lines(dropFirst(Smooth.4$s)[, 1] + dropFirst(Smooth.4$s)[, 3], col=3, lwd=2)

legend("bottomright", pch=c(1,NA,NA),col=c(1,2,4), lwd=c(1,2,2), legend=c("data","Filter","Smooth"))
    
Fore <- dlmForecast(Filt.4, nAhead=300, sampleNew=5)
lines(Fore$f, col=4, lwd=2)
data_a = rbind(matrix(shop1[sortlist,]$Sale), matrix(Fore$f))
data_b = ts(data_a,frequency = 365,start=c(2013,1,1))

#plot(data_a, col=1, type="l", lwd=1)
    
    
    
    
    
    
    
    
    
