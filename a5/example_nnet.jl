# Load X and y variable
using JLD
using PyPlot
data = load("basisData.jld")
(X,y) = (data["X"],data["y"])

#(n, d) = size(X)
#for j in 1:d
#    m = mean(X[:,j])
#    s = std(X[:,j])
#    X[:,j] = (X[:,j] .- m) ./ s
#end


n = size(X,1)
X = [ones(n,1) X]
d = 2

# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [150]
nParams = NeuralNet_nParams(d,nHidden)
w = randn(nParams,1)

# Train with stochastic gradient
maxIter = 50000
stepSize = 1e-4
w0 = w
for t in 1:maxIter
	
        if (t > 25000)
            stepSize = 1e-5
        end

        if (t > 35000)
            stepSize = 1e-6
        end

        if (t > 45000)
            stepSize = 1e-7
        end

	# The stochastic gradient update:
	i = rand(1:n)
	(f,g) = NeuralNet_backprop(w,X[i,:],y[i],nHidden)

        wtmp = w
        w = w - stepSize*g + stepSize*(w - w0);
        w0 = wtmp

        #w = w - stepSize*g
	# Every few iterations, plot the data/model:
	if (mod(t-1,round(maxIter/50)) == 0)
		@printf("Training iteration = %d\n",t-1)
		figure(1)
		clf()
		Xhat = -10:.05:10
		yhat = NeuralNet_predict(w,[ones(length(Xhat),1) Xhat],nHidden)
		plot(X[:,2],y,".")
		plot(Xhat,yhat,"g-")
		sleep(.1)
	end
end


