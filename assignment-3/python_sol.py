#!/usr/bin/python

import numpy as np
from numpy import dot, random
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat

def softmax(x):
    exps = np.nan_to_num(np.exp(x))
    return exps/np.nan_to_num(np.sum(exps))

class classifier(object):
    def __init__(self, input_size, classes, debug=True):
        '''Initialize the classifier with the input vector size
        and the number of classes required'''
        self.input_size = input_size
        self.classes = classes
        self.W = random.randn(input_size, classes)
        self.b = random.randn(classes, 1)
        self.DEBUG = debug
        self.cost_over_time = np.zeros(100)

    def setDebug(lev=True):
        self.DEBUG = lev

    def getCostOverTime():
        return self.cost_over_time

    def Y(self, train_data):
        '''The model that predicts the class of the input vectors using
        the current parmeters.'''
        a = dot(train_data, self.W) + np.tile(self.b.flatten(), (len(train_data), 1))
        return np.array([softmax(x) for x in a])

    def costf(self, train_data, train_targets):
        '''The traindata should contain the training inputs and
        train_targets the target vectors. Evaluates the cross entropy cost
        with the current set of data and parameters'''
        Y = self.Y(train_data)
        J = -sum([dot(t, ly) for t,ly in zip(train_targets, np.nan_to_num(np.log(np.nan_to_num(Y))))])
        return J

    def grad_costf(self, train_data, train_targets):
        '''Computes the gradient of the cost function for a batch. This one was hell
        to calculate by hand but I did it.'''
        Y = self.Y(train_data)
        gradW = dot(train_data.T, (Y - train_targets))
        gradb = np.reshape(np.sum(Y - train_targets, axis=0), (self.classes, 1)) 
        return gradW, gradb

    def GD(self, train_data, train_targets, epochs=30, eta=0.01):
        '''Trains the classifier using gradient descent. Uses the entire
        dataset for a single epoch. Maybe I\'ll implement the stochastic
        version soon.'''
        #Reserve the array 
        self.cost_over_time = np.zeros(epochs)
        #Start the training
        for i in range(epochs):
            print("Training Epoch %d..."%(i))
            gradW, gradb = self.grad_costf(train_data, train_targets)
            self.W = self.W - eta*gradW
            self.b = self.b - eta*gradb
            if self.DEBUG:
                cost = self.costf(train_data, train_targets)
                self.cost_over_time[i] = cost
                print("Cost: "+str(cost))
            print("Done")

    def SGD(self, train_data, train_targets, batch_size=10, epochs=30, eta=0.01):
        '''Trains the data using stochastic gradient descent.'''
        self.cost_over_time = np.zeros(epochs)

        for i in range(epochs):
            print("Training Epoch %d..."%(i))
            #Split the data into mini batches
            NROWS = train_data.shape[0]
            ROWS = [n for n in range(NROWS)]
            random.shuffle(ROWS)
            batches = [ROWS[n:n+batch_size] for n in range(0,NROWS,batch_size)] 
            
            for batch in batches:
                #Compute the gradient for the mini batches
                gradW, gradb = self.grad_costf(train_data[batch,:], train_targets[batch,:])
                #Do gradient descent for each of the mini batches
                self.W = self.W - eta*gradW
                self.b = self.b - eta*gradb
            
            if self.DEBUG:
                cost = self.costf(train_data, train_targets)
                self.cost_over_time[i] = cost
                print("Cost: "+str(cost))
            print("Done")

    def evalData(self, test_data, test_targets):
        '''Takes the testing data and calculates the number of
        incorrectly classified inputs'''
        Y = self.Y(test_data)
        TOTAL = test_data.shape[0]
        corrects = np.array(np.argmax(Y, axis=1) == np.argmax(test_targets, axis=1), dtype=float)
        pcor = 100*sum(corrects)/TOTAL
        print("Percentage correctly Classified: "+str(pcor))

if __name__ == '__main__':
    #Extract the data from the file and prep it
    DATA = loadmat('TRAINTEST2D.mat')
    CL1 = DATA['TRAIN'][0][0][0][0].T
    CL2 = DATA['TRAIN'][0][0][0][1].T
    CL3 = DATA['TRAIN'][0][0][0][2].T
    CL4 = DATA['TRAIN'][0][0][0][3].T
    t1  = np.tile([1,0,0,0], (CL1.shape[0],1))
    t2  = np.tile([0,1,0,0], (CL2.shape[0],1))
    t3  = np.tile([0,0,1,0], (CL3.shape[0],1))
    t4  = np.tile([0,0,0,1], (CL4.shape[0],1))

    X = np.vstack((CL1,CL2,CL3,CL4))
    T = np.vstack((t1,t2,t3,t4))

    #Plot the unclassified data
    plt.scatter(CL1[:,0], CL1[:,1], marker='x', c='r')
    plt.scatter(CL2[:,0], CL2[:,1], marker='o', c='g')
    plt.scatter(CL3[:,0], CL3[:,1], marker='s', c='b')
    plt.scatter(CL4[:,0], CL4[:,1], marker='^', c='y')
    plt.title('Scatter plot of the raw data')

    #Initialize the classifier
    clf = classifier(2, 4)
    #Train the classifier
    clf.SGD(X, T, epochs=30, eta=0.1)


    #Get the testing data
    # CL1 = DATA['TEST'][0][0][0][0].T
    # CL2 = DATA['TEST'][0][0][0][1].T
    # CL3 = DATA['TEST'][0][0][0][2].T
    # CL4 = DATA['TEST'][0][0][0][3].T

    #Trying to visualize the decision boundary
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.argmax(clf.Y(np.c_[xx.ravel(), yy.ravel()]), axis=1)

    # Put the result into a color plot
    plt.figure()
    plt.title('Plot of the decision boundaries')
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(T,axis=1), cmap=plt.cm.Paired)

    plt.show()

