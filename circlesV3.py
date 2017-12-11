# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:44:32 2017

@author: rportelas
"""
from tme6 import CirclesData
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.functional as F

def init_model(nx, nh, ny):
    
    model = torch.nn.Sequential(
    nn.Linear(nx, nh),
    nn.Tanh(),
    nn.Linear(nh, ny))

    loss = nn.CrossEntropyLoss()
    return model, loss

def loss_accuracy(Ytil, Y, Y_onehot):
    L = loss(Ytil, Y)
    
    _, indYhat = torch.max( nn.functional.log_softmax(Ytil), 1)
    _, indY = torch.max(Y_onehot, 1)
    acc = torch.sum(indY == indYhat).float() * 100. / float(indY.size(0));

    return L, acc


def sgd(model, eta):
    for param in model.parameters():
        param.data -= eta * param.grad.data
    model.zero_grad()



if __name__ == '__main__':

    data = CirclesData()

    data.plot_data()

    # init
    Xtrain = data.Xtrain
    Ytrain = data.Ytrain
    N = data.Xtrain.shape[0]
    Nbatch = 16
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    
    model, loss = init_model(nx, nh, ny)

    curves = [[],[], [], []]

    # epoch
    Ltrains = []
    Ltests = []
    acctrains = []
    acctests =[]
    
    for iteration in range(900):

        # batches
        model.train()
        for j in range(N // Nbatch):
            indsBatch = range(j * Nbatch, (j+1) * Nbatch)
            X = Variable(Xtrain[indsBatch, :], requires_grad=False)
            Y = Variable(Ytrain[indsBatch, :], requires_grad=False)
            Ytil = model(X)
            _, Y_not_onehot = Y.max(1)
            L, _ = loss_accuracy(Ytil, Y_not_onehot, Y)
            L.backward()
            sgd(model, 0.03)
       

        model.eval()
        Ytil_train = model(Variable(data.Xtrain, requires_grad=False))
        Ytil_test = model(Variable(data.Xtest, requires_grad=False))
        _, Ytest_not_onehot = torch.max(data.Ytest,1)
        _, Ytrain_not_onehot = torch.max(data.Ytrain,1)
        Ltrain, acctrain = loss_accuracy(Ytil_train, Variable(Ytrain_not_onehot, requires_grad=False), Variable(data.Ytrain, requires_grad=False))
        Ltest, acctest = loss_accuracy(Ytil_test, Variable(Ytest_not_onehot, requires_grad=False), Variable(data.Ytest, requires_grad=False))
        Ygrid = model(Variable(data.Xgrid, requires_grad=False))
        
        Ltrains.append(Ltrain.data.numpy())
        Ltests.append(Ltest.data.numpy())
        acctrains.append(acctrain.data.numpy())
        acctests.append(acctest.data.numpy())
        
        model.zero_grad()
        

        #Use This for online plotting        
        
        #title = 'Iter {}: Acc train {:.1f}% ({:.2f}), acc test {:.1f}% ({:.2f})'.format(iteration, acctrain, Ltrain, acctest, Ltest)
        #print(title)
        #data.plot_data_with_grid(Ygrid, title)
        #data.plot_loss(Ltrain, Ltest, acctrain, acctest)
    
    
    #data.plot_data_with_grid(Ygrid.data)
    plt.plot(Ltrains)
    plt.plot(Ltests)
    plt.ylabel('evolution of Loss during training')
    plt.xlabel('training iteration number')
    plt.savefig('lossC3.png')
    plt.show()
    plt.plot(acctrains)
    plt.plot(acctests)
    plt.xlabel('training iteration number')
    plt.legend(['train accuracy', 'test accuracy'], loc='lower right')
    plt.savefig('accuracyC3.png')
    plt.ylabel('evolution of accuracy during training')
    plt.show()
    print 'max accuracy on test set: ' + str(max(acctests))
    

    print "done"