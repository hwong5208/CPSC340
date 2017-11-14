include("misc.jl")
include("findMin.jl")

function logReg(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,y)

	# Solve least squares problem
	w = findMin(funObj,w,derivativeCheck=true)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticObj(w,X,y)
	yXw = y.*(X*w)
	f = sum(log.(1 + exp.(-yXw)))
	g = -X'*(y./(1+exp.(yXw)))
	return (f,g)
end

# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function logRegOnevsAll(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] = -1 # Treat other classes as -1

		# Each binary objective has the same features but different lables
		funObj(w) = logisticObj(w,X,yc)

		W[:,c] = findMin(funObj,W[:,c],verbose=false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(indmax,Xhat*W,2)

	return LinearModel(predict,W)
end

function softmaxObj(w,X,y,k)
        (n,d) = size(X)
        W = reshape(w,d,k)
        f = 0
        g = zeros(d,k)

        partial = function(j,c)
                total = 0
                for i in 1:n
                        term1 = y[i] == c ? 1 : 0
                        term2 = exp(W[:,c]'*X[i,:]) / sum(exp.(X[i,:]'*W))
                        total += X[i,j]*(term2-term1)
                end
                return total
        end

        for i in 1:n
                f += -W[:,y[i]]'*X[i,:] + log(sum(exp.(X[i,:]'*W)))
        end

        for j in 1:d
                for c in 1:k
                        g[j,c] = partial(j,c)
                end
        end

        return (f,reshape(g,d*k,1))
end

function softmaxClassifier(X,y)
        (n,d) = size(X)
        k = maximum(y)

        W = zeros(d,k)
        Wp = reshape(W,d*k,1)

        funObj(w) = softmaxObj(w,X,y,k)

        Wp = findMin(funObj,Wp,derivativeCheck=true,verbose=false)
        W = reshape(Wp,d,k);
        @show(W)

        predict(Xhat) = mapslices(indmax,Xhat*W,2)

        return LinearModel(predict,W)
end
