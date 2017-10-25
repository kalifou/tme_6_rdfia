# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:08:16 2017

@author: kalifou
"""

# Chargement de la classe
from tme6 import CirclesData
from layer import *
# import de la classe
data = CirclesData()

# instancie la classe fournie
# Acces aux donn ́ees
Xtrain = data.Xtrain
Ytrain = data.Ytrain
# torch.Tensor contenant les entr ́ees du r ́eseau pour
#l’apprentissage

# affiche la taille des donn ́ees : torch.Size([200, 2])
N = Xtrain.shape[0]
# nombre d’exemples
nx = Xtrain.shape[1]
# dimensionalit ́e d’entr ́ee
# donn ́ees disponibles : data.Xtrain, data.Ytrain, data.Xtest, data.Ytest,data.Xgrid
ny = Ytrain.shape[1]
# Fonctions d’affichage
data.plot_data()

print "Nx : ",nx,"\nNy : ",ny
params = init_params(nx,2,ny)
# affiche les points de train et test
Ygrid = forward(params, data.Xgrid)
# calcul des predictions Y pour tous les points de la grille (forward et params non fournis, `a coder)

y = Ygrid['Y']
print data.Xgrid.shape,y.shape,y
data.plot_data_with_grid(y)
# affichage des points et de la fronti`ere de d ́ecision gr^ace `a la grille

#data.plot_loss(loss_train, loss_train, acc_train, acc_test)
# affiche les courbes
#de loss et accuracy en train et test. Les valeurs `a fournir sont des scalaires,
#elles sont stock ́ees pour vous, il suffit de passer les nouvelles valeurs `a
#chaque it ́eratio