# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:10:55 2017

@author: kalifou
"""
import math as ma
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from tme6 import CirclesData

def init_params(nx,nh,ny):
    mean = 0.0
    std = 0.3 
    theta_1 =  t.Tensor(nx,nh).normal_(mean,std)
    theta_2 =  t.Tensor(nh,ny).normal_(mean,std)
    
    bias_1 =  t.Tensor(1,nh).normal_(mean,std) #?!
    bias_2 =  t.Tensor(1,ny).normal_(mean,std) #?!
    
    return {'Wh':theta_1,'Wy':theta_2, 'bh':bias_1,'by':bias_2}

def softmax(x):
    return t.exp(x)/t.sum(t.exp(x),0)
    
def forward(params, X):
    """Inference on X"""
    batch_size = X.shape[0]
    
    b_h =  params['bh']
    n_h = b_h.shape[1]
    
    b_y =  params['by']
    n_y = b_y.shape[1]
    
    H_tild = t.mm(X, params['Wh'])  + b_h.expand(batch_size,n_h) 
    H =  t.tanh(H_tild)
    Y_tild = t.mm(H, params['Wy']) + b_y.expand(batch_size,n_y)            
    Y = softmax(Y_tild)
    
    outputs = {'X':X,'H_tild':H_tild, 'H':H, 'Y_tild':Y_tild, 'yhat':Y}   
    return outputs['yhat'], outputs

def loss_accuracy(Yhat, Y):
    L = -t.sum(Y * t.log(Yhat))
    
    acc = 0
    _, predInd = t.max(Yhat,1)
    _, trueInd = t.max(Y,1)
    nb_pred = t.sum(predInd == trueInd)
    acc = float(nb_pred)*100 / trueInd.size(0)

    return L, acc

def backward(params, outputs, Y):
    grads = {}

    # TODO remplir"Wy" avec les paramètres Wy, Wh, by, bh
    # grads["Wy"] = ...
    grads = dict()
    grads['Y_tild'] = outputs['yhat'] - Y
    grads['Wy'] = grads['Y_tild'].t().mm(outputs['H'])
    grads['by'] = t.sum(grads['Y_tild'],0)
    print grads['Y_tild'].shape, params['Wy'].shape, outputs['H'].shape
    
    
    inter_grad = grads['Y_tild'].mm(params['Wy'].t() )
    grads['H_tild'] = inter_grad.mul(1-outputs['H'].pow(2) )

    grads['Wh'] = grads['H_tild'].t().mm(outputs['H']) 
    grads['bh'] = t.sum(grads['H_tild'],0) #No transpose ?!
    return grads

def sgd(params, grads, eta):
    # TODO mettre à jour le contenu de params
    
    return params



if __name__ == '__main__':

    # init
    data = CirclesData()
    data.plot_data()
    N = data.Xtrain.shape[0]
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.03

    # Premiers tests, code à modifier
    params = init_params(nx, nh, ny)
    Yhat, outputs = forward(params, data.Xtrain)
    L, _ = loss_accuracy(Yhat, data.Ytrain)
    grads = backward(params, outputs, data.Ytrain)
    params = sgd(params, grads, eta)

    # TODO apprentissage

    # attendre un appui sur une touche pour garder les figures
    input("done")
