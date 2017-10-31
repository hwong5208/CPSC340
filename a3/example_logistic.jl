# Load X and y variable
using JLD
data = load("logisticData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Standardize columns and add bias
n = size(X,1)
include("misc.jl")
(X,mu,sigma) = standardizeCols(X)
X = [ones(n,1) X]

# Standardize columns of test data, using mean/std from train data
t = size(Xtest,1)
Xtest = standardizeCols(Xtest,mu=mu,sigma=sigma)
Xtest = [ones(t,1) Xtest]

# Fit logistic regression model
#include("logReg.jl")
#model = logReg(X,y)

# Fit logistic regression L2 model
#include("logReg.jl")
#model = logRegL2(X,y,1)

# Fit logistic regression L1 model
#include("logReg.jl")
#model = logRegL1(X,y,1)

# Fit logistic regression L0 model
include("logReg.jl")
model = logRegL0(X,y,1)

# Count number of non-zeroes in model
numberOfNonZero = sum(model.w .!= 0)
@show(numberOfNonZero)

# Compute training and validation error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@show(trainError)
yhat = model.predict(Xtest)
validError = mean(yhat .!= ytest)
@show(validError)
