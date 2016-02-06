#!/usr/bin/python

import scipy.io
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv, norm
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


#Load the data from the file
DATA = scipy.io.loadmat('ASSIGNMENT1.mat')
data = DATA['DATA']

#Split the data into training and test sets
samples = len(data)
#Shuffle the data randomly
np.random.shuffle(data)
#Split into test and train
training, test = data[0:2500,:], data[2500:,:]

#Plot the training data
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.set_title('Scatter plot of Z1')
ax.scatter(training[:,0], training[:,1], training[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z1')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(training[:,0], training[:,1], training[:,3])
ax2.set_title('Scatter Plot of Z2')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z2')

#Using Regularized regression
#Get data and append column of ones to the left
X = np.concatenate((np.ones((training.shape[0],1)), training[:,0:2]), axis=1)
#Get the target vectors
t1 = training[:,2]
t2 = training[:,3]
#Regularization paramater lambda
l = 1.

#Use sine and cosine basis functions here
#5 Features in total
PHI = np.vstack((X[:,0], np.sin(2*np.pi*X[:,1]), np.cos(2*np.pi*X[:,1]), 
                 np.sin(2*np.pi*2.5*X[:,2]), np.cos(2*np.pi*2.5*X[:,2])))

#Use Formula for regularized regression to get optimal W
#W = (PHI'*PHI + lambda*I)^-1 *PHI' t
W1 = np.dot(t1, np.dot(inv(np.dot(PHI.T, PHI) + l*np.identity(PHI.shape[1])), PHI.T))
W2 = np.dot(t2, np.dot(inv(np.dot(PHI.T, PHI) + l*np.identity(PHI.shape[1])), PHI.T))

#Print the values of W
print("Using regularized regression")
print(W1, W2)

#Plot the best fit function to check
Xv, Yv = np.meshgrid(np.arange(0,1,0.01), np.arange(0,1,0.01))

Z1_cap = W1[0] + W1[1]*np.sin(2*np.pi*Xv) + W1[2]*np.cos(2*np.pi*2.5*Xv) +\
         W1[3]*np.sin(2*np.pi*2.5*Yv) + W1[4]*np.cos(2*np.pi*2.5*Yv)
#Z1_cap = W1[0] + W1[1]*Xv + W1[2]*Yv + W1[3]*Xv**2 + W1[4]*Yv**2 + W1[5]*Xv**3 + W1[6]*Yv**3
Z2_cap = W2[0] + W2[1]*np.sin(2*np.pi*Xv) + W2[2]*np.cos(2*np.pi*Xv) +\
         W2[3]*np.sin(2*np.pi*2.5*Yv) + W2[4]*np.cos(2*np.pi*2.5*Yv)

fig2 = plt.figure()
ax3 = fig2.gca(projection='3d')
surf = ax3.plot_surface(Xv, Yv, Z1_cap, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig2.colorbar(surf, shrink=0.5, aspect=5)
ax3.set_title('Regularized Regression')
ax3.set_xlabel('X1')
ax3.set_ylabel('Y1')
ax3.set_zlabel('Z1 prediction')

fig3 = plt.figure()
ax4 = fig3.gca(projection='3d')
surf = ax4.plot_surface(Xv, Yv, Z2_cap, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True)
fig3.colorbar(surf, shrink=0.5, aspect=5)
ax4.set_title('Regularized Regression')
ax4.set_xlabel('X1')
ax4.set_ylabel('Y1')
ax4.set_zlabel('Z2 prediction')

#Using kernel methods
print("Using kernel Methods")
l2 = 10
s = 0.07
#Calculate the Gram matrix
pairwise_dists = squareform(pdist(X[:,1:], 'euclidean'))
K = scipy.exp(-(pairwise_dists)** 2 / s ** 2)
print(K.shape)

#Calculate the vector a
a1 = np.dot(inv(K + l2*np.identity(X.shape[0])), t1)
a2 = np.dot(inv(K + l2*np.identity(X.shape[0])), t2)

#Calculate the predictions for Z
#Get the coordinate grid
Xv = np.linspace(0,1,50)
Yv = np.linspace(0,1,50)
Xv, Yv = np.meshgrid(Xv,Yv)
#Unroll the grid of coordinates
coords = np.vstack((Xv.flatten(), Yv.flatten())).T
print("coords shape: "+str(coords.shape))
#Calculate the kernel matrix by calculating pairwise distances
kernel_matrix = scipy.exp(-(cdist(coords, X[:,1:], 'euclidean'))**2/s**2)

print("kernel shape: "+str(kernel_matrix.shape))

#Calculate the predicted Z values and reshape into a grid for plotting.
Z1_preds = np.reshape(np.dot(kernel_matrix, a1), (50,50))
Z2_preds = np.reshape(np.dot(kernel_matrix, a2), (50,50))

#print(Z1_preds.shape, Z2_preds.shape)
#print(Xv.shape, Yv.shape)

# for i in xrange(50):
#     for j in xrange(50):
#         Z1_preds[i,49-j] = np.dot(kx_x(np.array([Xv[i], Xv[j]])), a1)

#Plot the predictions for kernel regression
fig4 = plt.figure()
ax5 = fig4.gca(projection='3d')
surf = ax5.plot_surface(Xv, Yv, Z1_preds, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=True)
fig4.colorbar(surf, shrink=0.5, aspect=5)
ax5.set_title('Kernel Regression Using Gaussian Kernel')
ax5.set_xlabel('X1')
ax5.set_ylabel('Y1')
ax5.set_zlabel('Z1 prediction')

fig5 = plt.figure()
ax6 = fig5.gca(projection='3d')
surf2 = ax6.plot_surface(Xv, Yv, Z2_preds, rstride=1, cstride=1, cmap=cm.coolwarm,
                         linewidth=0, antialiased=True)
fig5.colorbar(surf, shrink=0.5, aspect=5)
ax6.set_title('Kernel Regression Using Gaussian Kernel')
ax6.set_xlabel('X1')
ax6.set_ylabel('Y1')
ax6.set_zlabel('Z2 prediction')

plt.show()
