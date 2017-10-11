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
            #  V[i][j]= v[i]+0.0
              end 
        end
    end


  #  println(size(X'))
   #  println(V)
   #   println(size(y))
    w = (X'*V*X)\(X'*V*y)
    # Make linear prediction function
    predict(Xhat) = Xhat*w

    # Return model
    return GenericModel(predict)
end


function leastTanBasis(X,y)

     (n,)= size(X)
      X0 = tanBasis(X)


    # Find regression weights minimizing squared error
    w = (X0'*X0)\(X0'*y)

    # Make linear prediction function
    predict(Xhat) = tanBasis(Xhat)*w

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

function tanBasis(X)
    (n,)= size(X)
    X0 = ones(n,12)

    for i in 1:n
        X0[i,2] =X[i]
        X0[i,3]= tan(0.1357*X[i])
        X0[i,4]= tan(0.1357*X[i]).^2
         X0[i,5]= tan(0.1357*X[i]).^3
         X0[i,6]= tan(0.1357*X[i]).^4
         X0[i,7]= tan(0.1357*X[i]).^5
         X0[i,8]= tan(0.1357*X[i]).^6
         X0[i,9]= tan(0.1357*X[i]).^7
         X0[i,10]= tan(0.1357*X[i]).^8
         X0[i,11] = sin(X[i])  
         X0[i,12] = sin(X[i]).^2  
         
     end 

   return X0
end

