clear all
clc

%Solution for question 1a
%Create a meshgrid to evaluate the kernel function on a grid
X = [0:0.01:1];
Y = [0:0.01:1];
[Xv Yv] = meshgrid(X,Y);

%Plotting Gaussian kernel matrix
%Gaussian function => exp(-((Xv-Yv).^2)/a)
%choosing a=0.2
Z = exp(-((Xv-Yv).^2)/0.2);
surface(Xv,Yv,Z)

%Plotting Sigmoid function Kernel matrix
figure();
%tanh function => tanh(aXY + b)
%choosing a=1, b=1
Z = tanh((Xv.*Yv) + 1);
surface(Xv,Yv,Z)

%Solution for 1b

%Splitting the data into train and test 25 75 split
[trainInd, valInd, testInd] = dividerand(length(DATA), 0.25,0,0.75);
trainDATA = DATA(trainInd, :);

%Now: separating X, t1, t2
X = trainDATA(:,[1 2]);
t1 = trainDATA(:,3);
t2 = trainDATA(:,4);

%Using kernel ridge regression
%Calcualte the Gram matrix
lambda = 10;
s = 0.07
K = squareform(pdist(X, 'euclidean'));
K = exp(-(K.^2)/s^2);

%The model for kernel regression is y = sum ai*k(x,xi)
%Calculate a values

a1 = inv(K + lambda*eye(length(K)))*t1;
a2 = inv(K + lambda*eye(length(K)))*t2;

[Xv Yv] = meshgrid([0:0.02:1], [0:0.02:1]);
coords = [reshape(Xv, [length(Xv)*length(Xv) 1]) reshape(Yv, [length(Yv)*length(Yv) 1])];

kernel_matrix = exp(-(pdist2(coords, X, 'euclidean').^2)/s^2);

Z1_preds = reshape(kernel_matrix*a1, [length(Xv) length(Xv)]);
Z2_preds = reshape(kernel_matrix*a2, [length(Xv) length(Xv)]);

figure()
surface(Xv,Yv,Z1_preds);
figure()
surface(Xv,Yv,Z2_preds);


