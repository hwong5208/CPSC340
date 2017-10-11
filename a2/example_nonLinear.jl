# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a least squares model
include("leastSquares.jl")
#model = leastSquares(X,y)
model = leastTanBasis(X,y)
#model = leastSquaresBasis(X,y,10)
#model = leastSinBasis(X,y)
# Evaluate training error
yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("Squared train Error with least squares: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("Squared test Error with least squares: %.3f\n",testError)



println(size(X))

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
Xhat = minimum(X):.1:maximum(X)
yhat = model.predict(Xhat)
plot(Xhat,yhat,"g")


function run_leastSquaresBasis_n_time(n)

for i in 1:n
model = leastSquaresBasis(X,y,i)

yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("Squared train Error with p: %i and least squares: %.3f\n",i,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("Squared test Error with p: %i and least squares: %.3f\n",i,testError)
#figure()
#plot(X,y,"b.")
#Xhat = minimum(X):.1:maximum(X)
#yhat = model.predict(Xhat)
#plot(Xhat,yhat,"g")
end
end