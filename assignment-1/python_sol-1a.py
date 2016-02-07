import scipy.io
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv, norm
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


#Assignment 1 (a): Plotting the kernel function
#Gaussian kernel
s = 0.1
Xv = np.linspace(0,1,100)
Yv = np.linspace(0,1,100)
Xv, Yv = np.meshgrid(Xv,Yv)

#Calculate the kernel matrix by calculating pairwise distances
gaussian_kernel = scipy.exp(-(Yv-Xv)** 2 / s ** 2)
sigmoid_kernel = np.tanh(Xv*Yv)
print gaussian_kernel.shape

plt.figure()
plt.pcolor(Xv, Yv, gaussian_kernel, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.title('Gaussian Kernel Matrix')

plt.figure()
plt.pcolor(Xv, Yv, sigmoid_kernel, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.title('Sigmoid Kernel Matrix')

plt.show()
