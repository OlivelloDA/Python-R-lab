work_dir <- "/Users/claudioolivelli/Documents/GitHub/Python lab/R_studentt_fitting"
setwd(work_dir)

#install.packages("data.table")
#install.packages("mnormt")
#install.packages("MASS")
#install.packages("qrmtools")
#install.packages("mvtnorm")
library(data.table)
library(mnormt)
library(MASS)
library(qrmtools)
library(mvtnorm)

TSLAdata<-as.data.table(read.csv("TSLA.csv"))
BNGOdata<-as.data.table(read.csv("BNGO.csv"))
#limit yourself to the Adj.Close Prices
TSLAClose<-TSLAdata$Adj.Close
BNGOClose<-BNGOdata$Adj.Close
#Extract the final close prices (which should be the adjusted final close price)
#for the two stocks, to determine the dollar value 
#Print Apple and Verizon closing prices

TSLALast<-TSLAdata[Date =="2020-10-09",Adj.Close]
BNGOLast<-BNGOdata[Date =="2020-10-09",Adj.Close]

print(c("TSLA closing price is ",TSLAlast))

TSLAReturn<-diff(TSLAClose)/TSLAClose[-length(TSLAClose)]
BNGOReturn<-diff(BNGOClose)/BNGOClose[-length(BNGOClose)]
ReturnSet<-cbind(TSLAReturn,BNGOReturn)

ReturnCov<-as.matrix(cov(ReturnSet),header=FALSE)
MeansVector<-c(mean(TSLAReturn), mean(BNGOReturn))
dollarweights <- as.matrix(c(TSLALast*10,BNGOLast*100),nrow=2)

TotVar<-t(dollarweights)%*%ReturnCov%*%(dollarweights)
TotSD<-sqrt(TotVar)*sqrt(250)
TotMean<-MeansVector%*%dollarweights
ValueAtRisk<--qnorm(0.01,mean=TotMean,sd=TotSD)#4796.26 similar TotMean +(TotSD*qnorm(0.01))
print(ValueAtRisk)
valueP <- TSLALast*10+ BNGOLast*100
print(c("the value of the portfolio today is", valueP))
#I fit a multivariate T distribution on the two sets of return , and I choose the degree of freedom that gives me
#the highest profil log likelihood among all the possible set of v
df = seq(1,4,0.1)
n = length(df)
loglik = rep(0,n)
for(i in 1:n){
  fit = cov.trob(ReturnSet,nu=df[i])
  # dmt is the probability density function for multivariate t
  loglik[i] = sum(log(dmt(ReturnSet,mean=fit$center,
                          S=fit$cov,df=df[i])))
}
#round to 4 digits
options(digits=4)
#chooses the maximum from the computed sequence
nuhat = df[which.max(loglik)]
nuhat
cov.trob(ReturnSet,nu=nuhat)


n = 272; df=nuhat
set.seed(5600)
mu=cov.trob(ReturnSet,nu=nuhat)$center
S=cov.trob(ReturnSet,nu=nuhat)$cov

#multivariate t with zero correlation
# rmt = Random Number Generation For Multivariate T
x = rmt(n,mu,S,df=df)
par(mfrow=c(1,3))

plot(x[,1],x[,2],main="(a) Multivariate-t, Fitted",
     xlab=expression(Y[1]),ylab=expression(Y[2]),xlim=c(-0.15,0.15),ylim=c(-0.15,0.15))
abline(h=0); abline(v=0)

plot(ReturnSet[,1],ReturnSet[,2],main="Actual Data",
     xlab=expression(Y[1]),ylab=expression(Y[2]),xlim=c(-0.15,0.15),ylim=c(-0.15,0.15))
abline(h=0); abline(v=0)


df = seq(1,4,0.1)
maxLL = max(loglik)
plot(df,2*loglik - 2*maxLL,type="l",
     cex.axis=1.5,cex.lab=1.5,
     ylab="2*loglikelihood - max",lwd=2,
     main="Profile likelihood for df")
abline(h = - qchisq(.95,1))
abline(h = 0)
abline(v=df[which.max(loglik)],col="red")


#Now compare to the Normal distribution
#compute the variance-covariance matrix
ReturnCov<-as.matrix(cov(ReturnSet),header=FALSE)
#compute the means (if we don't take zero as a proxy)
MeansVector<-c(mean(TSLAReturn), mean(BNGOReturn))

y=rmnorm(n,MeansVector,ReturnCov)
plot(y[,1],y[,2],main="(b) Multivariate-Normal, Direct",
     xlab=expression(Y[1]),ylab=expression(Y[2]),xlim=c(-0.15,0.15),ylim=c(-0.15,0.15))
abline(h=0); abline(v=0)



