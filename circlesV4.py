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

def init_model(nx, nh, ny, eta):
    
    model = torch.nn.Sequential(
    nn.Linear(nx, nh),
    nn.Tanh(),
    nn.Linear(nh, ny))

    optim = torch.optim.SGD(model.parameters(), lr=eta)
    loss = nn.CrossEntropyLoss()
    return model, loss, optim

def loss_accuracy(Ytil, Y, Y_onehot):
    L = loss(Ytil, Y)
    
    _, indYhat = torch.max( nn.functional.log_softmax(Ytil), 1)
    _, indY = torch.max(Y_onehot, 1)
    acc = torch.sum(indY == indYhat).float() * 100. / float(indY.size(0));

    return L, acc

if __name__ == '__main__':

    data = MNISTData()

    # init
    Xtrain = data.Xtrain
    Ytrain = data.Ytrain
    N = data.Xtrain.shape[0]
    Nbatch = 32
    nx = data.Xtrain.shape[1]
    nh = 100
    ny = data.Ytrain.shape[1]
    
    model, loss, optim = init_model(nx, nh, ny, 0.02)

    curves = [[],[], [], []]

    # epoch
    Ltrains = []
    Ltests = []
    acctrains = []
    acctests =[]
    
    graphic_step = 1024
    iteration = 0
    
    for epoch in range(1):

        # batches
        model.train()
        for j in range(N // Nbatch):
            indsBatch = range(j * Nbatch, (j+1) * Nbatch)
            X = Variable(Xtrain[indsBatch, :], requires_grad=False)
            Y = Variable(Ytrain[indsBatch, :], requires_grad=False)
            
            Ytil = model(X)
            _, Y_not_onehot = Y.max(1)
            L, _ = loss_accuracy(Ytil, Y_not_onehot, Y)
            
            optim.zero_grad()
            L.backward()
            optim.step()
            
            iteration += Nbatch
            
            if iteration % graphic_step == 0:
        
                model.eval()
                Ytil_train = model(Variable(data.Xtrain, requires_grad=False))
                Ytil_test = model(Variable(data.Xtest, requires_grad=False))
                _, Ytest_not_onehot = torch.max(data.Ytest,1)
                _, Ytrain_not_onehot = torch.max(data.Ytrain,1)
                Ltrain, acctrain = loss_accuracy(Ytil_train, Variable(Ytrain_not_onehot, requires_grad=False), Variable(data.Ytrain, requires_grad=False))
                Ltest, acctest = loss_accuracy(Ytil_test, Variable(Ytest_not_onehot, requires_grad=False), Variable(data.Ytest, requires_grad=False))
                
                Ltrains.append(Ltrain.data.numpy())
                Ltests.append(Ltest.data.numpy())
                acctrains.append(acctrain.data.numpy())
                acctests.append(acctest.data.numpy())
        
                model.zero_grad()
        
    
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
    

    print "done"
