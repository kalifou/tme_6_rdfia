# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:10:55 2017

@author: kalifou
"""

import torch
from tme6 import CirclesData

def init_params(nx, nh, ny):
    params = {}
    params['Wh'] = torch.randn(nh, nx) * 0.3
    params['bh'] = torch.zeros(nh, 1)
    params['Wy'] = torch.randn(ny, nh) * 0.3
    params['by'] = torch.zeros(ny, 1)
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

    acc = torch.sum(indY == indYhat) * 100. / indY.size(0);

    return L, acc

def backward(params, outputs, Y):
    bsize = Y.shape[0]
    grads = {}
    deltay = outputs['yhat'] - Y
    grads['Wy'] = torch.mm(deltay.t(), outputs['h'])
    grads['by'] = deltay.sum(0, keepdim=True).t()
    deltah = torch.mm(deltay, params['Wy']) * (1 - torch.pow(outputs['h'], 2))
    grads['Wh'] = torch.mm(deltah.t(), outputs['X'])
    grads['bh'] = deltah.sum(0, keepdim=True).t()

    grads['Wy'] /= bsize
    grads['by'] /= bsize
    grads['Wh'] /= bsize
    grads['bh'] /= bsize

    return grads

def sgd(params, grads, eta):
    params['Wy'] -= eta * grads['Wy']
    params['Wh'] -= eta * grads['Wh']
    params['by'] -= eta * grads['by']
    params['bh'] -= eta * grads['bh']

    return params
