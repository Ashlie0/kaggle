# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 23:11:35 2018

@author: ashli
"""

from load import loadim,loadcsv,showgray,ImageToMask
#from load import loadb
#import chainer
#import chainer.functions as F
#import chainer.links as L
from chainer import Variable,cuda
from net import ResNet
import numpy as np
from train import Trainer
import pandas as pd

model=ResNet()
#gpu_device=0
#cuda.get_device(gpu_device).use()
#model.to_gpu(gpu_device)

tr_im=loadim(a=-1,j="trainimages")
tr_ma=loadim(a=-1,j="trainmasks")
#teimage=loadim(a=-1,j="test")

#trdepth,tedepth=loadcsv()
#trdepth=sorted(trdepth,key=lambda t:t[0])
#tedepth=sorted(tedepth,key=lambda t:t[0])
#tr_dep=[int(t[1]) for t in trdepth]
#te_dep=[int(t[1]) for t in tedepth]
#tr_im_ma_dep=[[tr_im[i],tr_ma[i],tr_dep[i]] for i in range(4000)]
#tr_im_ma_dep=sorted(tr_im_ma_dep,key=lambda t:t[2])

gpu_device=0
X=cuda.to_gpu(tr_im,device=0)
Y=cuda.to_gpu(tr_ma,device=0)

model=ResNet()
cuda.get_device(gpu_device).use()
model.to_gpu(gpu_device)

trainer = Trainer(model)

trainer.fit(X)

df_loss=pd.DataFrame(trainer.loss)
df_loss.to_csv('loss_csv')














