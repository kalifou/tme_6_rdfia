# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:10:55 2017

@author: kalifou
"""

import torch as t
import numpy as np

def init_params(nx,nh,ny):
    mean = 0.0
    std = 0.3 
    theta_1 =  t.Tensor(nx,nh).normal_(mean,std)
    theta_2 =  t.Tensor(nh,ny).normal_(mean,std)
    
    bias_1 =  t.Tensor(1,nh).normal_(mean,std) #?!
    bias_2 =  t.Tensor(1,ny).normal_(mean,std) #?!
    
    return {'W_h':theta_1,'W_y':theta_2, 'b_h':bias_1,'b_y':bias_2}

def softmax(x):
    return t.exp(x)/t.sum(t.exp(x),0)
    
def forward(params, X):
    """Inference on X"""
    batch_size = X.shape[0]
    
    b_h =  params['b_h']
    n_h = b_h.shape[1]
    
    b_y =  params['b_y']
    n_y = b_y.shape[1]
    
    H_tild = t.mm(X, params['W_h'])  + b_h.expand(batch_size,n_h) 
    H =  t.tanh(H_tild)
    Y_tild = t.mm(H, params['W_y']) + b_y.expand(batch_size,n_y)            
    Y = softmax(Y_tild)
    
    return {'H_tild':H_tild, 'H':H, 'Y_tild':Y_tild, 'Y':Y}

