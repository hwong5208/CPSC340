# Load X and y variable
using JLD
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
Xtest = load("citiesSmall.jld","Xtest")
ytest = load("citiesSmall.jld","ytest")

# Fit a KNN classifier
k = 10
include("knn.jl")
model = knn(X,y,k)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
trainError = mean(yhat .!= ytest)
@printf("Test Error with %d-nearest neighbours: %.3f\n",k,testError)

include("plot2Dclassifier.jl")
# Q5.1.3 Plot classifier
function plotKNN(k)
    model = knn(X, y, k)
    plot2Dclassifier(X, y, model)
end

# Q.5.2.5
function citiesBig2Errors(k)
    @printf("k: %d\n", k)

    X = load("citiesBig2.jld", "X")
    y = load("citiesBig2.jld", "y")
    Xtest = load("citiesBig2.jld", "Xtest")
    ytest = load("citiesBig2.jld", "ytest")

    model = cknn(X, y, k)

    yhat = model.predict(X)
    Etrain = sum(yhat .!= y) / size(yhat, 1)
    @printf("Etrain: %.3f\n", Etrain)
    plot2Dclassifier(X, y, model)

    yhat = model.predict(Xtest)
    Etest = sum(yhat .!= ytest) / size(yhat, 1)
    @printf("Etest: %.3f\n", Etest)
    plot2Dclassifier(Xtest, ytest, model)
end
