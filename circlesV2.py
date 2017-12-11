# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:44:32 2017

@author: rportelas
"""
from tme6 import CirclesData
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

def init_params(nx, nh, ny):
    params = {}
    params['Wh'] = Variable(torch.randn(nh, nx) * 0.3, requires_grad=True)
    params['bh'] = Variable(torch.zeros(nh, 1), requires_grad=True)
    params['Wy'] = Variable(torch.randn(ny, nh) * 0.3, requires_grad=True)
    params['by'] = Variable(torch.zeros(ny, 1), requires_grad=True)
    return params

def forward(params, X):
    bsize = X.size(0)
    nh = params['Wh'].size(0)
    ny = params['Wy'].size(0)
    outputs = {}
    outputs['X'] = X
    outputs['htilde'] = torch.mm(X, params['Wh'].t()) + params['bh'].t().expand(bsize, nh)
    outputs['h'] = torch.tanh(outputs['htilde'])
    outputs['ytilde'] = torch.mm(outputs['h'], params['Wy'].t()) + params['by'].t().expand(bsize, ny)
    outputs['yhat'] = torch.exp(outputs['ytilde'])
    outputs['yhat'] = outputs['yhat'] / (outputs['yhat'].sum(1, keepdim=True)).expand_as(outputs['yhat'])
    return outputs['yhat'], outputs

def loss_accuracy(Yhat, Y):
    L = - torch.mean(Y * torch.log(Yhat))

    _, indYhat = torch.max(Yhat, 1)
    _, indY = torch.max(Y, 1)
    acc = torch.sum(indY == indYhat).float() * 100. / float(indY.size(0));

    return L, acc


def sgd(params, eta):
    #print 1000. * params['Wy'].grad.data
    params['Wy'].data -= eta * params['Wy'].grad.data
    params['Wh'].data -= eta * params['Wh'].grad.data
    params['by'].data -= eta * params['by'].grad.data
    params['bh'].data -= eta * params['bh'].grad.data
    
    params['Wy'].grad.data.zero_()
    params['Wh'].grad.data.zero_()
    params['by'].grad.data.zero_()
    params['bh'].grad.data.zero_()

    return params


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
            Yhat, outs = forward(params, Variable(X, requires_grad=False))
            L, _ = loss_accuracy(Yhat, Variable(Y, requires_grad=False))
            L.backward()
            params = sgd(params, 0.03)
       

        
        Yhat_train, _ = forward(params, Variable(data.Xtrain, requires_grad=False))
        Yhat_test, _ = forward(params, Variable(data.Xtest, requires_grad=False))
        Ltrain, acctrain = loss_accuracy(Yhat_train, Variable(data.Ytrain, requires_grad=False))
        Ltest, acctest = loss_accuracy(Yhat_test, Variable(data.Ytest, requires_grad=False))
        Ygrid, _ = forward(params, Variable(data.Xgrid, requires_grad=False))
        
        Ltrains.append(Ltrain.data.numpy())
        Ltests.append(Ltest.data.numpy())
        acctrains.append(acctrain.data.numpy())
        acctests.append(acctest.data.numpy())
        

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
    plt.savefig('lossC2.png')
    plt.show()
    plt.plot(acctrains)
    plt.plot(acctests)
    plt.xlabel('training iteration number')
    plt.legend(['train accuracy', 'test accuracy'], loc='lower right')
    plt.savefig('accuracyC2.png')
    plt.ylabel('evolution of accuracy during training')
    plt.show()
    print 'max accuracy on test set: ' + str(max(acctests))
    

    print "done"