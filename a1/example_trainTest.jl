# Load X and y variable
using JLD
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)

# Train a depth-2 decision tree
depth = 2
include("decisionTree_infoGain.jl")
model = decisionTree_infoGain(X,y,depth)

# Evaluate the trianing error
yhat = model.predict(X)
trainError = sum(yhat .!= y)/n
@printf("Train error with depth-%d decision tree: %.3f\n",depth,trainError)

# Evaluate the test error
Xtest = load("citiesSmall.jld","Xtest")
ytest = load("citiesSmall.jld","ytest")
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum(yhat .!= ytest)/t
@printf("Test error with depth-%d decision tree: %.3f\n",depth,testError)


# Q3.2 Plot training and test error from depth 1 to 15

function calcError(X, y, model, size)
    yhat = model.predict(X)
    return sum(yhat .!= y)/size
end

function plotErrors(depth)
    E = zeros(depth*2, 2)
    d = zeros(depth*2)

    for i in 1:depth
        model = decisionTree_infoGain(X, y, i)

        E[i, 1] = i;
        E[i, 2] = calcError(X, y, model, n)
        d[i] = 1

        E[i+depth, 1] = i;
        E[i+depth, 2] = calcError(Xtest, ytest, model, t)
        d[i+depth] = 2
    end

    include("plot2Dclassifier.jl")
    plot2Dscatter(E,d)
end
