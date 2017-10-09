# Load data
using JLD
X = load("clusterData2.jld","X")

# Density-based Clustering
radius = 15
minPts = 5
include("dbCluster.jl")
y = dbCluster(X,radius,minPts,doPlot=true)

include("clustering2Dplot.jl")
clustering2Dplot(X,y)
