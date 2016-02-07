clear all
load('ASSIGNMENT1.mat')

%Separate DATA into training and test data
[trainInd, valInd, testInd] = dividerand(length(DATA), 0.25, 0, 0.75);
train_data = DATA(trainInd,:);
test_data = DATA(testInd,:);

%Get the X matrix and target vectors
X = [ones(length(train_data),1) train_data(:,[1 2])];
t1 = train_data(:, 3);
t2 = train_data(:, 4);

%Linear regression with regularization
lambda = 10; %Regularization parameter
PHI = [X(:,1) sin(2*pi*X(:,2)) cos(2*pi*X(:,2)) sin(5*pi*X(:,3)) cos(5*pi*X(:,3))]; %The PHI Matrix

W1 = inv(PHI'*PHI + lambda*identity(size(X,2))*PHI'*t1;
W2 = inv(PHI'*PHI + lambda*identity(size(X,2))*PHI'*t2;
