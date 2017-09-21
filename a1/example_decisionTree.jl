# Load X and y variable
using JLD
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)

# Fit a decision tree and compute error
include("decisionTree.jl")
depth = 2
model = decisionTree(X,y,depth)

# Evaluate training error
yhat = model.predict(X)
trainError = sum(yhat .!= y)/n
@printf("Error with depth-%d decision tree: %.3f\n",depth,trainError)

# Plot classifier
include("plot2Dclassifier.jl")
plot2Dclassifier(X,y,model)

# Q3.3 Validation set error

function calcError(X, y, model)
    n = size(y, 1)
    yhat = model.predict(X)
    return sum(yhat .!= y)/n
end

function halfSetValidation(depth)
    @printf("depth: %d\n", depth)

    (r, c) = size(X)

    @printf("\nUsing first half for training and second half for validation\n")
    Xtrain = X[1:end .<= r/2, :]
    ytrain = y[1:end .<= r/2]
    Xvalidate = X[1:end .> r/2, :]
    yvalidate = y[1:end .> r/2]

    model = decisionTree(Xtrain, ytrain, depth)

    Etrain = calcError(Xtrain, ytrain, model)
    @printf("Train error: %.3f\n", Etrain);

    Evalidate = calcError(Xvalidate, yvalidate, model)
    @printf("Validation error: %.3f\n", Evalidate);

    @printf("\nUsing second half for training and first half for validation\n")
    Xtrain = X[1:end .> r/2, :]
    ytrain = y[1:end .> r/2]
    Xvalidate = X[1:end .<= r/2, :]
    yvalidate = y[1:end .<= r/2]

    model = decisionTree(Xtrain, ytrain, depth)

    Etrain = calcError(Xtrain, ytrain, model)
    @printf("Train error: %.3f\n", Etrain);

    Evalidate = calcError(Xvalidate, yvalidate, model)
    @printf("Validation error: %.3f\n", Evalidate);
end
