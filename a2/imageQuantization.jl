function quantizeImage(fn,b)
	Xo = imread(fn);
	(nRows,nCols) = size(Xo);
	X = reshape(Xo, (nRows*nCols,3));

	model = kMeans(X,2^b,doPlot=false)
	return model.y,model.W,nRows,nCols
end

function deQuantizeImage(y,W,nRows,nCols)
	X = zeros(nRows*nCols,3)
	for i in 1:nRows*nCols
		X[i,:] = W[Int(y[i]),:];
	end

	Xo = reshape(X, (nRows,nCols,3));
	imshow(Xo);
	return Xo
end
