include("misc.jl")
include("findMin.jl")

function leastSquaresGradient(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = leastSquaresObj(w,X,y)

	# This is how you compute the function and gradient:
	(f,g) = funObj(w)

	# Derivative check that the gradient code is correct:
	g2 = numGrad(funObj,w)

	if maximum(abs.(g-g2)) > 1e-4
		@printf("User and numerical derivatives differ:\n")
		@show([g g2])
	else
		@printf("User and numerical derivatives agree\n")
	end

	# Solve least squares problem
	w = findMin(funObj,w)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end

function leastSquaresObj(w,X,y)

	f = sum(log(exp.(X*w-y)+exp.(y-X*w)))
	g = X'*((exp.(X*w-y)-exp.(y-X*w))./(exp.(X*w-y)+exp.(y-X*w)))
	return (f,g)
end
