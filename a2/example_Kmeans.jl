# Load data
using JLD
X = load("clusterData.jld","X")

# K-means clustering
k = 4
include("kMeans.jl")
model = kMeans(X,k,doPlot=true)
y = model.predict(X)

include("clustering2Dplot.jl")
clustering2Dplot(X,y,model.W)

#= Q2.1.3, Q2.3.1, Q2.3.3
X = load("clusterData2.jld","X")

k = 4;
(n,d) = size(X);
minError = Inf;
minY = zeros(n);
minW = zeros(k,d);
for i in 1:50
	model = kMedians(X,k,doPlot=false);
	if (model.error < minError)
		minY = model.y;
		minW = model.W;
		minError = model.error;
	end
end
@show(minError)
clustering2Dplot(X,minY,minW);
=#

#= Q2.2.3, Q2.3.2
X = load("clusterData2.jld","X")

(n,d) = size(X);
minY = zeros(n);
minW = zeros(k,d);

kvals = zeros(10);
minErrors = zeros(10);

for k in 1:10
	minError = Inf;
	for i in 1:50
		model = kMedians(X,k,doPlot=false);
		if (model.error < minError)
			minError = model.error;
		end
	end
	kvals[k] = k;
	minErrors[k] = minError;
end
@show(kvals, minErrors)
plot(kvals, minErrors, "o");
=#
