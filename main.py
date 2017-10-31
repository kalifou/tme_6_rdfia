# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:08:16 2017

@author: kalifou
"""

# Chargement de la classe
from tme6 import CirclesData
from layer import *
import torch
import matplotlib.pyplot as plt


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
    params = init_params(nx, nh, ny)

    curves = [[],[], [], []]

    # epoch
    Ltrains = []
    Ltests = []
    acctrains = []
    acctests =[]
    for iteration in range(900):

        # batches
        for j in range(N // Nbatch):
            indsBatch = range(j * Nbatch, (j+1) * Nbatch)
            X = Xtrain[indsBatch, :]
            Y = Ytrain[indsBatch, :]
            Yhat, outputs = forward(params, X)
            L, _ = loss_accuracy(Yhat, Y)
            grads = backward(params, outputs, Y)
            params = sgd(params, grads, 0.02)

        Yhat_train, _ = forward(params, data.Xtrain)
        Yhat_test, _ = forward(params, data.Xtest)
        Ltrain, acctrain = loss_accuracy(Yhat_train, data.Ytrain)
        Ltest, acctest = loss_accuracy(Yhat_test, data.Ytest)
        Ygrid, _ = forward(params, data.Xgrid)
        
        Ltrains.append(Ltrain)
        Ltests.append(Ltest)
        acctrains.append(acctrain)
        acctests.append(acctest)
        

        #Use This for online plotting        
        
        #title = 'Iter {}: Acc train {:.1f}% ({:.2f}), acc test {:.1f}% ({:.2f})'.format(iteration, acctrain, Ltrain, acctest, Ltest)
        #print(title)
        #data.plot_data_with_grid(Ygrid, title)
        #data.plot_loss(Ltrain, Ltest, acctrain, acctest)
    
    data.plot_data_with_grid(Ygrid)
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