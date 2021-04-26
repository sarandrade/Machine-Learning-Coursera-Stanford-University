function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
C = 0;
sigma = 0;
low_error = 0;
C_vetor = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vetor = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

for(i=1:8)
  for(j=1:8)
    model= svmTrain(X, y, C_vetor(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vetor(j)));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    total_error = sum(error);
    
    if(i==1 && j==1)
      low_error = total_error;
    end;
    
    if(total_error < low_error)
      low_error = total_error;
      C = C_vetor(i);
      sigma = sigma_vetor(j);
    end;
  end;
end;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
% =========================================================================

end