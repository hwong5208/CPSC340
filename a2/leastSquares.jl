include("misc.jl")

function leastSquares(X,y)

	# Find regression weights minimizing squared error
	w = (X'*X)\(X'*y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end


function leastSquaresBias(X,y)

     (n,)= size(X)
     X0 =  addBasis(X)
    # Find regression weights minimizing squared error
    w = (X0'*X0)\(X0'*y)

    # Make linear prediction function
    predict(Xhat) =   addBasis(Xhat)*w

    # Return model
    return GenericModel(predict)
end

function leastSquaresBasis(X,y,p)

     (n,)= size(X)
      X0 = polyBasis(X,p)


    # Find regression weights minimizing squared error
    w = (X0'*X0)\(X0'*y)

    # Make linear prediction function
    predict(Xhat) = polyBasis(Xhat,p)*w

    # Return model
    return GenericModel(predict)
end

function weigthedleastSquares(X,y,v)
    (n,) = size(v)
    V =  zeros(n,n)
     
    for i in 1:n
        for j in 1:n
            if i==j
                V[i,j]=v[i]
              end 
        end
    end
    
    w = (X'*V*X)\(X'*V*y)
    # Make linear prediction function
    predict(Xhat) = Xhat*w

    # Return model
    return GenericModel(predict)
end


function leastSinBasis(X,y)

     (n,)= size(X)
      X0 = sinBasis(X)


    # Find regression weights minimizing squared error
    w = (X0'*X0)\(X0'*y)

    # Make linear prediction function
    predict(Xhat) = sinBasis(Xhat)*w

    # Return model
    return GenericModel(predict)
end

function addBasis(X)
    (n,)= size(X)
    X0 = ones(n,2)
    for i in 1:n
        X0[i,2]= X[i]
     end 
   return X0
end 


function polyBasis(X,p)
    (n,)= size(X)
    X0 = ones(n,p+1)
    for i in 1:n
        X0[i,2]= X[i]
     end 
   for i in 3:p+1
      X0[:,i]= X0[:,2].^(i-1)
   end
   return X0
end 

function sinBasis(X)
    (n,)= size(X)
    X0 = ones(n,5)

    for i in 1:n
        
        X0[i,2] = (X[i]-2).^3

         X0[i,3] = sin(5*X[i]) 
         X0[i,4] = (X[i]-2)^2
         X0[i,5] = (X[i]-2)

     end 

   return X0
end

