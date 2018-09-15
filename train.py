# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:07:32 2018

@author: ashli
"""


import chainer.functions as F
import numpy as np
from chainer import cuda, optimizers
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, ResNet):
        self.ResNet=ResNet
    def fit(self, X, Y, epochs=30):
        self.X=X
        self.Y=Y
        self.epochs=epochs
        
        o_resnet=optimizers.Adam(alpha=1e-5, beta1=0.1)
        o_resnet.setup(self.tgs)
        self.loss=[]
        for epoch in range(1,epochs+1):
            
            sum_loss=np.float32(0)
            
            for i in range(10):
                l=len(X)
                image=X[i:i*l:]
                mask=Y[i:i*l:]
                
                output=self.ResNet(image,mask)
                
                loss = F.mean_absolute_error(mask, output)
                
                self.ResNet.cleargrads()
                loss.backward()
                o_resnet.update()
                
                sum_loss += loss.data
                
            print('epoch:', epoch, 'loss:', loss,'sum_loss_2:', sum_loss)
                
            self.loss.append(sum_loss)

        print(self.loss)