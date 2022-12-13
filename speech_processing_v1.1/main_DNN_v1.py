# -*- coding: utf-8 -*-
"""
DNN 基础架构
"""

import os
import librosa  #这个版本时0.9.1 要求numpy 版本小于等于 1.20
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
import keras

import PaintTestUseEtc as myPaint  #我自己补充的，用于直观展示和比较等
import basicUse #我自己补充的一些小点的代码
import generateNoisy as myNoisy #我自己补充的用于混合语音用的库
import myVocalCordModel #自己些的声带模型

import measure #用于测量声音

import basicClass
import applyDNN

import utils 
    
noisyMatchListTest=myNoisy.readNoisyList("output/noisyMatchListTest.csv") #测试组合列表
noisyMatchListTrain=myNoisy.readNoisyList("output/noisyMatchListTrain.csv") #训练组合列表

data1=basicClass.test_train_data() #数据
dirs1=basicClass.test_train_dir()  #路径
args1=basicClass.test_train_args() #参数

dirs1.check_files_dir() #检查文件夹
###*这里修改路径等
#dirs1.datasets_dir="?"
#dirs1.output_dir="?"
dirs1.renew_output_path() #output 相关输出文件路径更新

###*调整参数
args1.winWide=512
args1.fftSize=1024
args1.renew_all_args()

args1.epochN = 10 #*训练DNN的迭代次数
args1.batchN = 256 



"""   
### 单个训练
applyDNN.loadWav(data1,args1,dirs1,noisyMatchListTrain,listi=0,snr=0) #*读取一个训练数据
#model=keras.models.load_model(data1.PATH_ROOT+"/myDNNModel.h5") #*读取旧的继续训练
model=applyDNN.getDnnModel(args1)                    #************使用新的来训练(2选1)
model,info=applyDNN.dnnTrain(data1,args1,model,save=True) #训练DNN
### 单个训练 end
"""



"""   """
### 单个测试
model=keras.models.load_model(data1.PATH_ROOT+"/myDNNModel.h5")  
applyDNN.loadWav(data1,args1,dirs1,noisyMatchListTest,listi=0,snr=0) #读取一个测试数据

applyDNN.dnnTest(data1,args1,model)
applyDNN.synthesizeSpeech(data1,args1,dirs1)
### 单个测试 end

myPaint.paintFT(data1.magnitude_clean,title="Spectrogram",MoP="M",dbYoN="Y")


"""   
### 批量化训练1by1
#model=keras.models.load_model(data1.PATH_ROOT+"/myDNNModel.h5") #*读取旧的继续训练
model=applyDNN.getDnnModel(args1)                    #************使用新的来训练(2选1)

for i in range(2):
    applyDNN.loadWav(data1,args1,dirs1,noisyMatchListTrain,listi=i,snr=0) #*读取一个训练数据
    if i%10==0:
        model,history=applyDNN.dnnTrain(data1,args1,model,save=True) #训练DNN 训练几个后再存储
    else:
        model,history=applyDNN.dnnTrain(data1,args1,model,save=False) #训练DNN
### 批量化训练1by1 end
#accuracy=history.history['sparse_categorical_accuracy']
#b=history.history['val_sparse_categorical_accuracy']
#c=history.history['val_loss']
a=history.history
### 批量化训练1by1 end
"""


"""  
###批量化测试
model=keras.models.load_model(data1.PATH_ROOT+"/myDNNModel.h5")  
for snr in [0,5,10]:
    for i in [2,3,6,7]:
        E_D_E=[(i&4)>>2,(i&2)>>1,i&1]
        tableName=str(E_D_E[0])+str(E_D_E[1])+str(E_D_E[2])+"_snr"+str(snr)
        applyDNN.measureWav(data1,args1,dirs1,model,noisyMatchListTest,listRange=[0,100],snr=snr,E_D_E=E_D_E,verbose=True,tableName=tableName)
###批量化测试 end
"""











