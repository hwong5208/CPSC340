include("misc.jl")
include("clustering2Dplot.jl")

type PartitionModel
	predict # Function for clustering new points
	y # Cluster assignments
	W # Prototype points
	error # Error (Q2.1.3)
end

function kMeansError(X, y, W)
	error = 0;
	(n,d) = size(X);

	for i in 1:n
		error += sum((X[i,:] .- W[Int(y[i]),:]).^2);
	end

	return error;
end

function kMediansError(X, y, W)
	error = 0;
	(n,d) = size(X);

	for i in 1:n
		error += sum(abs.(X[i,:] .- W[Int(y[i]),:]));
	end

	return error;
end

function kMeans(X,k;doPlot=false)
# K-means clustering

(n,d) = size(X)

# Choos random points to initialize means
W = zeros(k,d)
perm = randperm(n)
for c = 1:k
	W[c,:] = X[perm[c],:]
end

# Initialize cluster assignment vector
y = zeros(n)
changes = n
error = Inf;

while changes != 0

	# Compute (squared) Euclidean distance between each point and each mean
	D = distancesSquared(X,W)

	# Assign each data point to closest mean (track number of changes labels)
	changes = 0
	for i in 1:n
		(~,y_new) = findmin(D[i,:])
		changes += (y_new != y[i])
		y[i] = y_new
	end

	# Optionally visualize the algorithm steps
	if doPlot && d == 2
		clustering2Dplot(X,y,W)
		sleep(.1)
	end


	# Find mean of each cluster
	for c in 1:k
		W[c,:] = mean(X[y.==c,:],1)
	end

	# Optionally visualize the algorithm steps
	if doPlot && d == 2
		clustering2Dplot(X,y,W)
		sleep(.1)
	end

	error = kMeansError(X,y,W);
#	@printf("Running k-means, changes = %d\n",changes)
	@printf("Running k-means, error = %d\n", error)
end

function predict(Xhat)
	(t,d) = size(Xhat)

	D = distancesSquared(Xhat,W)

	yhat = zeros(Int64,t)
	for i in 1:t
		(~,yhat[i]) = findmin(D[i,:])
	end
	return yhat
end

return PartitionModel(predict,y,W,error)
end

# Q2.3.3
function kMedians(X,k;doPlot=false)
# K-medians clustering

(n,d) = size(X)

# Choos random points to initialize medians
W = zeros(k,d)
perm = randperm(n)
for c = 1:k
	W[c,:] = X[perm[c],:]
end

# Initialize cluster assignment vector
y = zeros(n)
changes = n
error = Inf;

while changes != 0

	# Compute L1-norm between each point and each mean
	D = LOneNorms(X,W)

	# Assign each data point to closest median (track number of changes labels)
	changes = 0
	for i in 1:n
		(~,y_new) = findmin(D[i,:])
		changes += (y_new != y[i])
		y[i] = y_new
	end

	# Optionally visualize the algorithm steps
	if doPlot && d == 2
		clustering2Dplot(X,y,W)
		sleep(.1)
	end


	# Find median of each cluster
	for c in 1:k
		W[c,:] = median(X[y.==c,:],1)
	end

	# Optionally visualize the algorithm steps
	if doPlot && d == 2
		clustering2Dplot(X,y,W)
		sleep(.1)
	end

	error = kMediansError(X,y,W);
	@printf("Running k-medians, error = %d\n",error)
end

function predict(Xhat)
	(t,d) = size(Xhat)

	D = LOneNorms(Xhat,W)

	yhat = zeros(Int64,t)
	for i in 1:t
		(~,yhat[i]) = findmin(D[i,:])
	end
	return yhat
end

return PartitionModel(predict,y,W,error)
end
