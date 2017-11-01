# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:44:32 2017

@author: rportelas
"""
from tme6 import MNISTData
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

def init_model(nx, nh, ny, eta):
    
    model = torch.nn.Sequential(
    nn.Linear(nx, nh),
    nn.ReLU(),
    nn.Linear(nh, nh),
    nn.ReLU(),
    nn.Linear(nh, ny),
    nn.ReLU(),
    nn.Linear(ny, ny))

    optim = torch.optim.SGD(model.parameters(), lr=eta)
    loss = nn.CrossEntropyLoss()
    return model, loss, optim

def loss_accuracy(Ytil, Y, Y_onehot):
    
    L = loss(Ytil, Y)
    
    Yhat = nn.functional.log_softmax(Ytil)
    _, indYhat = torch.max(Yhat, 1)
    _, indY = torch.max(Y_onehot, 1)
    acc = torch.sum(indY == indYhat).float() * 100. / float(indY.size(0));

    return L, acc

def evaluation(model,N,X,Y):
    loss = 0
    acc = 0
    for k in range(N // len(X)):
        indsBatch = range(k * Nbatch, (k+1) * Nbatch)
        X = Variable(X[indsBatch, :], requires_grad=False)
        Y = Variable(Y[indsBatch, :], requires_grad=False)
        
        Ytil = model(X)
        _, Y_not_onehot = Y.max(1)
        l, a = loss_accuracy(Ytil, Y_not_onehot, Y)
        loss += l
        acc += a   
    return loss.data.numpy() / float(k+1), acc.data.numpy() / float(k+1)

if __name__ == '__main__':

    data = MNISTData()

    # init
    Xtrain = data.Xtrain
    Ytrain = data.Ytrain
    Xtest = data.Xtest
    Ytest = data.Ytest
    N = data.Xtrain.shape[0]
    Ntest = data.Xtest.shape[0]
    Nbatch = 32
    nx = data.Xtrain.shape[1]
    nh = 500
    ny = data.Ytrain.shape[1]
    
    model, loss, optim = init_model(nx, nh, ny, 0.001)
    
    curves = [[],[], [], []]

    # epoch
    Ltrains = []
    Ltests = []
    acctrains = []
    acctests =[]
    
    graphic_step = 1024
    iteration = 0
    
    #shuffle training data
    perm = torch.randperm(N)
    Xtrain = Xtrain[perm]
    Ytrain = Ytrain[perm]
    
    for epoch in range(1):

        # batches
        model.train()
        for j in range(N // Nbatch):
            indsBatch = range(j * Nbatch, (j+1) * Nbatch)
            X = Variable(Xtrain[indsBatch, :], requires_grad=False)
            Y = Variable(Ytrain[indsBatch, :], requires_grad=False)
            
            Ytil = model(X)
            _, Y_not_onehot = Y.max(1)
            loss_train, acc_train = loss_accuracy(Ytil, Y_not_onehot, Y)
            
            optim.zero_grad()
            loss_train.backward()
            optim.step()
            
            iteration += Nbatch
             
            if iteration % graphic_step == 0:
                model.eval()

                loss_train, acc_train = evaluation(model, N, Xtrain, Ytrain)
                acctrains.append(acc_train)
                Ltrains.append(loss_train)
        
                loss_test, acc_test = evaluation(model, Ntest, Xtest, Ytest)
                Ltests.append(loss_test)
                acctests.append(acc_test)
                
                model.train()
                
                
    
    plt.plot(Ltrains)
    plt.plot(Ltests)
    plt.ylabel('evolution of Loss during training')
    plt.xlabel('training iteration number')
    plt.savefig('loss.png')
    plt.show()
    plt.plot(acctrains)
    plt.plot(acctests)
    plt.xlabel('training iteration number')
    plt.legend(['train accuracy', 'test accuracy'], loc='lower right')
    plt.savefig('accuracy.png')
    plt.ylabel('evolution of accuracy during training')
    plt.show()
    print 'max accuracy on test set: ' + str(max(acctests))
