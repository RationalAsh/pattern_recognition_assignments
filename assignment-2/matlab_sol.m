clear all;

%Load the data from the file
load('TRAINTEST2D.mat');

%Get the training data
CL1 = TRAIN{1}{1}';
CL2 = TRAIN{1}{2}';
CL3 = TRAIN{1}{3}';
CL4 = TRAIN{1}{4}';
%Combined dataset
X = [CL1;CL2;CL3;CL4];

%Plot the data
figure('Units', 'Inches', 'Position', [0 0 8.27 5.112], 'PaperPositionMode', 'auto')
scatter(CL1(:,1), CL1(:,2), 'x');
hold on;
scatter(CL2(:,1), CL2(:,2), 'o');
scatter(CL3(:,1), CL3(:,2), '^');
scatter(CL4(:,1), CL4(:,2), 's');
title('Scatter plot of raw data');

%Do PCA on the data
%First find the covariance of the dataset
C = cov(X);
%Find the eigenvalues of the covariance matrix
[V, D] = eig(C);
D = sum(D, 2);
E = V(:,find(D==max(D)))

%Project all the data onto the principal axis
CL1_proj = CL1*E;
CL2_proj = CL2*E;
CL3_proj = CL3*E;
CL4_proj = CL4*E;

%Plot the projected data
figure('Units', 'Inches', 'Position', [0 0 8.27 5.112], 'PaperPositionMode', 'auto')
scatter(CL1_proj, zeros(size(CL1_proj)), 'x')
hold on
scatter(CL2_proj, zeros(size(CL2_proj)), 'o')
scatter(CL3_proj, zeros(size(CL3_proj)), '^')
scatter(CL4_proj, zeros(size(CL4_proj)), 's')
title('Projected Data after PCA');

%Do normal LDA on the data
%Find within class covariance matrix
Sw = cov(CL1) + cov(CL2) + cov(CL3) + cov(CL4);
%Find Between class covariance matrix
means = [mean(CL1); mean(CL2); mean(CL3); mean(CL4)];
Sb = cov(means);

% Find eigenvalues of $S_w^{-1}S_b$
Me = inv(Sw)*Sb;
[V, D] = eig(Me);
D = sum(D, 2);
E = V(:,find(D==max(D)));

%Project all the data onto the principal axis
CL1_proj = CL1*E;
CL2_proj = CL2*E;
CL3_proj = CL3*E;
CL4_proj = CL4*E;

%Plot the projected data
figure('Units', 'Inches', 'Position', [0 0 8.27 5.112], 'PaperPositionMode', 'auto')
scatter(CL1_proj, zeros(size(CL1_proj)), 'x')
hold on
scatter(CL2_proj, zeros(size(CL2_proj)), 'o')
scatter(CL3_proj, zeros(size(CL3_proj)), '^')
scatter(CL4_proj, zeros(size(CL4_proj)), 's')
title('Projected Data after LDA');


%Do Kernel-LDA on the data using the gaussian kernel
%Compute the Gram matrix for each class
s = 1
K1 = exp(-(pdist2(X,CL1).^2)/(2*s^2));
K2 = exp(-(pdist2(X,CL2).^2)/(2*s^2));
K3 = exp(-(pdist2(X,CL3).^2)/(2*s^2));
K4 = exp(-(pdist2(X,CL4).^2)/(2*s^2));
% K1 = pdist2(X,CL1).^2;
% K2 = pdist2(X,CL2).^2;
% K3 = pdist2(X,CL3).^2;
% K4 = pdist2(X,CL4).^2;
%Compute the N matrix
N = K1*(eye(length(CL1)) - eye(length(CL1))./length(CL1))*K1'...
    + K2*(eye(length(CL2)) - eye(length(CL2))./length(CL2))*K2'...
    + K3*(eye(length(CL3)) - eye(length(CL3))./length(CL3))*K3'...
    + K4*(eye(length(CL4)) - eye(length(CL4))./length(CL4))*K4';

%Compute the M matrix
%Overall gram matrix
K = exp(-(squareform(pdist(X)).^2)/(2*s^2));
% K = squareform(pdist(X)).^2;
Ms = sum(K,2)/size(X,1);
M1 = sum(K1,2)/size(CL1,1);
M2 = sum(K2,2)/size(CL2,1);
M3 = sum(K3,2)/size(CL3,1);
M4 = sum(K4,2)/size(CL4,1);
Me = [M1';M2';M3';M4'];
A = Me - repmat(Ms', 4,1);
M = size(CL1,1)*A(1,:)'*A(1,:) + size(CL2,1)*A(2,:)'*A(2,:)...
    + size(CL3,1)*A(3,:)'*A(3,:) + size(CL4,1)*A(4,:)'*A(4,:);

%Need to find eigenvalues of N^-1*M
[V, D] = eig(pinv(N + eye(size(N,1)))*M);
E = real(V(:,1));
pts = K*E;


%Plot the projected data
figure('Units', 'Inches', 'Position', [0 0 8.27 5.112], 'PaperPositionMode', 'auto')
scatter(pts(1:13), zeros(size(CL1_proj)), 'x')
hold on
scatter(pts(14:26), zeros(size(CL2_proj)), 'o')
scatter(pts(27:39), zeros(size(CL3_proj)), '^')
scatter(pts(40:52), zeros(size(CL4_proj)), 's')
title('Projected Data after K-LDA');




