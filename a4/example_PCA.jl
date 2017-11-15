# Load data
dataTable = readcsv("animals.csv")
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)



function FrobeniusNorm(X)
     X0 = X.^2
     X1 = sum(X0)
     return X1
end

function VarianceRemain(ZW,X)

    return 1- vecnorm(ZW-X).^2/vecnorm(X).^2
end

# Standardize columns
include("misc.jl")
(X,mu,sigma) = standardizeCols(X)

# # Plot matrix as image
# using PyPlot
# figure(1)
# clf()
# imshow(X)

# # Show scatterplot of 2 random features
# j1 = rand(1:d)
# j2 = rand(1:d)
# figure(2)
# clf()
# plot(X[:,j1],X[:,j2],".")
# for i in rand(1:n,10)
#     annotate(dataTable[i+1,1],
# 	xy=[X[i,j1],X[i,j2]],
# 	xycoords="data")
# end


include("PCA.jl")
k =13
model = PCA(X,k)

Z = model.compress(X)
ZW=model.expand(Z)

print(VarianceRemain(ZW,X))

# Plot matrix as image
using PyPlot
figure(1)
clf()
imshow(X)

# Show scatterplot of 2 random features

figure(2)
clf()
plot(Z[:,1],Z[:,2],".")
for i in (1:n)
    annotate(dataTable[i+1,1],
    xy=[Z[i,1],Z[i,2]],
    xycoords="data")
end


