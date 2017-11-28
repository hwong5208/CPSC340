include("misc.jl")
include("PCA.jl")
include("findMin.jl")

function MDS(X)
    (n,d) = size(X)

    # Compute all distances
    D = distancesSquared(X,X)
    D = sqrt.(abs.(D))

    # Initialize low-dimensional representation with PCA
    model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stress(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end

function stress(z,D)
    n = size(D,1)
    Z = reshape(z,n,2)

    f = 0
    G = zeros(n,2)
    for i in 1:n
        for j in i+1:n
            # Objective function
            Dz = norm(Z[i,:] - Z[j,:])
            s = D[i,j] - Dz
            f = f + (1/2)s^2

            # Gradient
            df = s
            dgi = (Z[i,:] - Z[j,:])/Dz
            dgj = (Z[j,:] - Z[i,:])/Dz
            G[i,:] -= df*dgi
            G[j,:] -= df*dgj
        end
    end
    return (f,G[:])
end


function ISOMAP(X)
    k =3
    (n,d) = size(X)

    # Compute all distances
    D = distancesSquared(X,X)
    D = sqrt.(abs.(D))

    E = ones(n,n).*Inf

    for i in 1:n
        A = sortperm(D[i,:])
        for j in 2:k+1
          
          E[i,A[j]] =D[i,A[j]]
          E[A[j],i] =D[A[j],i]     
        end 
    end

    D = zeros(n,n)
    for i in 1:n
        for j in 1:n
         D[i,j] = dijkstra(E,i,j)
        end
    end

    # Initialize low-dimensional representation with PCA
    model = PCA(X,2)
    Z = model.compress(X)
    

    funObj(z) = stress(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end



function ISOMAPdisconnectedGraph(X)
    k =2
    (n,d) = size(X)

    # Compute all distances
    D = distancesSquared(X,X)
    D = sqrt.(abs.(D))

    E = ones(n,n).*Inf

    for i in 1:n
        A = sortperm(D[i,:])
        for j in 2:k+1
          
          E[i,A[j]] =D[i,A[j]]
          E[A[j],i] =D[A[j],i]     
        end 
    end

    D = zeros(n,n)
    max = -Inf
    for i in 1:n
        for j in 1:n
         D[i,j] = dijkstra(E,i,j)
         if(D[i,j]!=Inf && D[i,j] > max)
            max =  D[i,j]
         end
        end
    end


    for i in 1:n
        for j in 1:n
             if(D[i,j]==Inf )
             D[i,j] = max
         end
        end
    end


    # Initialize low-dimensional representation with PCA
    model = PCA(X,2)
    Z = model.compress(X)
    

    funObj(z) = stress(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end



