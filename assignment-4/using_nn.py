#!/usr/bin/python

import numpy as np
from numpy import random, dot
from scipy.io import loadmat

#Sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def softmax(x):
    exps = np.exp(x)
    return exps/np.sum(exps)

def colwise_softmax(A):
    exps = np.exp(A)
    return exps/np.sum(exps,axis=0)

def onehot(DIM, K):
    a = np.zeros(DIM)
    a[K] = 1.
    return a

class ANN(object):
    def __init__(self, sizes, debug=True):
        '''Initialize the neural network'''
        self.input_size = sizes[0]
        self.output_size = sizes[-1]
        self.hiddens = sizes[1:-1]
        self.wsizes = sizes[1:]
        self.weights = [random.randn(sz[0], sz[1]) for sz in zip(sizes[1:], sizes[:-1])]
        self.biases = [random.randn(sz,1) for sz in sizes[1:]]
        self.activations = []
        self.DEBUG = debug

    def feedforward(self, train_data):
        '''Perform the feedforward operation on the neural network'''
        self.activations = []
        A = train_data.T
        for i in range(len(self.weights)):
            W = self.weights[i]
            b = self.biases[i]
            #If not final layer, apply logistic sigmoid
            if i < len(self.weights)-1:
                A = sigmoid(np.dot(W,A) + b)
            else:
                A = colwise_softmax(np.dot(W,A) + b)
            self.activations.append(A)
        return A

    def costf(self, train_targets):
        Y = self.activations[-1].T
        J = -np.sum([dot(a,b) for a,b in zip(train_targets,np.nan_to_num(np.log(Y)))])
        return J

    def backprop(self, train_data, train_targets):
        '''Uses the backpropogation algorithm to calculate the 
        gradients for the weights of each layer of the neural network.
        '''
        L = len(self.weights)
        #First get the output of the final layer
        Y = self.feedforward(train_data).T
        #Calculate delta
        DL = Y - train_targets
        deltas = [DL]
        gradWs = [np.dot(self.activations[0], DL).T]
        gradbs = [np.sum(DL, axis=0).reshape(-1,1)]
        D = DL
        #Calculate the deltas for each layer
        for i in range(L-1):
            D = np.dot(D, self.weights[-(i+1)])
            if i == L-2:
                gradW = np.dot(train_data.T, D).T
                gradb = np.sum(D,axis=0).reshape(-1,1)
            else:
                gradW = np.dot(self.activations[-(i+2)], D).T
                gradb = np.sum(D, axis=0).reshape(-1,1)
            gradWs.append(gradW)
            gradbs.append(gradb)
        return gradWs, gradbs

    def GD(self, train_data, train_targets, epochs=30, eta=0.01, debug=True):
        '''Trains the neural network using gradient descent. Uses backpropogation
        to compute the gradients of the weights and biases of the network. The 
        train_data should be in a matrix where each row is a data sample and the
        train_targets should be in the form of a matrix where each row is a one-of-K
        encoded binary vector that represents the class of the corresponding sample in
        train_data'''
        for j in range(epochs):
            alpha = 0.0001
            eta = eta
            print("Training Epoch %d: "%(j))
            #First calculate the gradients
            gradWs, gradbs = self.backprop(train_data, train_targets)
            #Update the weights
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - eta*gradWs[-(i+1)]
                self.biases[i] = self.biases[i] - eta*gradbs[-(i+1)]

            #eta = eta*(1-alpha)

            if self.DEBUG:
                cost = self.costf(train_targets)
                print("Cost: %f"%(cost))
            print("Done")

    def SGD(self, traindata, testdata, epochs=30, batch_size=10, eta=3.0, debug=True):
        '''Training data and testing data is given as a tuple of inputs and targets
        of the form (X, T).'''
        #Learning rate
        eta = 0.1
        alpha = 0.0001
        #Loop for number of epochs
        for i in range(epochs):
            if debug: print("Starting epoch 1...")
            #Generate mini batches
            random.shuffle(traindata)
            mini_batches = [traindata[k:k+batch_size] for k in range(0,len(traindata),batch_size)]
            for mini_batch in mini_batches:
                self.batch_update(mini_batch, eta)
            if debug:
                print("Epoch 1 complete")

    def evalData(self, test_data, test_targets):
        Y = self.feedforward(test_data).T
        TOTAL = test_data.shape[0]
        corrects = np.array(np.argmax(Y, axis=1) == np.argmax(test_targets, axis=1), dtype=float)
        pcor = 100*sum(corrects)/TOTAL
        print("Percentage correctly Classified: "+str(pcor))
    

if __name__ == '__main__':
    SIZE = 48
    CLASSES = 40
    NDIM = 200
    SAMPLES = 400
    HIDDEN0 = 340
    HIDDEN1 = 100

    #First read the data from the matlab file
    print("Loading dataset...")
    MATFILE = loadmat('ORLFACEDATABASE.mat')
    DATA = MATFILE['C'].T
    print("Done")

    #Generate the labels
    T = np.vstack([np.tile(onehot(CLASSES, i), (10,1)) for i in range(CLASSES)])

    #Partition data into train and test sets
    print("Partitioning data into train and test sets...")
    trinds = np.hstack([np.arange(i,i+5) for i in np.arange(0,SAMPLES,10)])
    testinds = np.hstack([np.arange(i+5,i+10) for i in np.arange(0,SAMPLES,10)])
    Xtr = DATA[trinds,:]
    Ttr = T[trinds,:]
    Xte = DATA[testinds,:]
    Tte = T[testinds,:]

    Xtr = Xtr/np.max(Xtr)
    Xte = Xte/np.max(Xtr)

    EPOCHS = 100
    eta = 0.00001

    #Initialize the neural network
    nn = ANN([SIZE*SIZE, HIDDEN0, CLASSES])
    #Manual implementation of training to test
    for i in range(EPOCHS):
        print("Training epoch %d: "%(i))
        #Do feedforward on the network to get the activations
        Y = nn.feedforward(Xtr).T

        #Get the delta for the last layer.
        D2 = (Y-Ttr)
        gradW2 = np.dot(nn.activations[0], D2).T
        gradb2 = np.sum(D2, axis=0).reshape(-1,1)

        #Backpropogate the errors
        D1 = np.dot(D2, nn.weights[1])
        gradW1 = np.dot(Xtr.T, D1).T
        gradb1 = np.sum(D1, axis=0).reshape(-1,1)

        #Do the gradient descent step
        nn.weights[1] = nn.weights[1] - eta*gradW2
        nn.biases[1] = nn.biases[1] - eta*gradb2

        nn.weights[0] = nn.weights[0] - eta*gradW1
        nn.biases[0] = nn.biases[0] - eta*gradb1

        #Evaluate and print the cost
        cost = nn.costf(Ttr)
        print("Cost: %f" %(cost))
        print("Done")

        #Evaluate the model on the test set
        nn.evalData(Xte, Tte)
        

    
    
