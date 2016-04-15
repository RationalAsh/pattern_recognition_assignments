#!/usr/bin/python

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import multiclass_logistic_reg as mclf



def PCA(train_data, NDIM):
    '''Performs PCA on given data.
    train_data: An NxD matrix. Each row is a D dimensional
                input vector.
    NDIM      : The number of dimensions to which I want to 
                reduce the data to. NDIM must be less than D.
    Returns:
    T    : A NDIMxD transformation matrix that projects input
           vectors into a lower dimensional subspace.
    EIGS : A list of the most significant eigen values
    '''
    C = np.cov(train_data, rowvar=0)
    W, V = np.linalg.eig(C)
    T = np.abs(V[:,:NDIM])
    EIGS = W[:NDIM]

    return T, EIGS

def dim_reduce(train_data, NDIM):
    '''Helper function that reduces the dimension of the data
    using PCA'''
    T, EIGS = PCA(train_data, NDIM)
    Xr = np.dot(train_data, T)
    return Xr

def onehot(DIM, K):
    a = np.zeros(DIM)
    a[K] = 1.
    return a



SIZE = 48
CLASSES = 40
NDIM = 200
SAMPLES = 400

#First read the data from the matlab file
print("Loading dataset...")
MATFILE = loadmat('ORLFACEDATABASE.mat')
DATA = MATFILE['C'].T
print("Done")

#Generate the labels
T = np.vstack([np.tile(onehot(CLASSES, i), (10,1)) for i in range(CLASSES)])

 #Partition the dataset into test and train.
inds = [i for i in range(DATA.shape[0])]
np.random.shuffle(inds)
trainInds = inds[:int(0.7*SAMPLES)]
testInds = inds[int(0.7*SAMPLES):]
#Training set
Xtr = DATA[trainInds,:]
Ttr = T[trainInds,:]
#Test set
Xte = DATA[testInds,:]
Tte = T[testInds,:]

#Perform PCA on the data
#print("Performing PCA on the data...")
#Xtr = dim_reduce(Xtr, NDIM)
#Xte = dim_reduce(Xte, NDIM)
#print("Done")

#Try out logistic regression on it

#First normalize data to prevent overflows
MAX = np.max(Xtr)
Xtr = Xtr/MAX
Xte = Xte/MAX

print("Attempting Logistic Regression on the training data and checking error rate...")
clf = mclf.classifier(Xtr.shape[1], 40)

EPOCHS = 1500

#Arrays to store the data
costs = np.zeros(EPOCHS)
train_acc = np.zeros(EPOCHS)
test_acc = np.zeros(EPOCHS)

for i in range(EPOCHS):
    clf.SGD(Xtr, Ttr, batch_size=50, epochs=1, eta=0.003)
    teacc = clf.evalData(Xte, Tte)
    tracc = clf.evalData(Xtr, Ttr)
    cost = clf.costf(Xte, Tte)

    costs[i] = cost
    train_acc[i] = tracc
    test_acc[i] = teacc
    
    if teacc >= 0.9:
        break

#Plot everything
plt.plot(costs)
plt.title('Cost over time')
plt.xlabel('Epochs')
plt.ylabel('Cost')

plt.figure()
plt.plot(train_acc, label='Training Set Classification Accuracy')
plt.hold(True)
plt.plot(test_acc, label='Test Set Classification Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Percentage Correctly Classified')


#Plot a visualization of the classification accuracy
YL = np.argmax(clf.Y(Xte), axis=1)
TL = np.argmax(Tte, axis=1)

inds = np.argsort(TL)
Y = YL[inds]
T = TL[inds]

plt.figure()
pred = plt.scatter(np.arange(len(Y)), Y, marker='x')
plt.hold(True)
actual = plt.scatter(np.arange(len(T)), T, marker='o', facecolors='none')
plt.title('Visualization of classifier Accuracy')
plt.xlabel('Sample')
plt.ylabel('Class Index')
plt.legend((pred, actual), ('Predicted Class', 'Actual Class'), loc='upper left')

plt.show()
