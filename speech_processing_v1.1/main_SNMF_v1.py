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
import applySNMF

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

#********SNMF 的参数设置
args1.winWide=512
args1.fftSize=1024
args1.renew_all_args()

args1.lambda1=2
args1.d=800
args1.maxiter=101
args1.displayIterN=50
args1.costfcn="kl"


#%%

"""   """
###单个训练或  多段合并后的训练尝试 多个语音段合并放入
applySNMF.loadWav(data1,args1,dirs1,noisyMatchListTrain,listi=0,snr=0) #*读取一个训练数据 单个,2选1
#applySNMF.loadWavRange(data1,args1,dirs1,noisyMatchListTrain,listRange=(0,3),snr=0) #*读取多个合并为一段 2选1 listRange控制读取范围
W_clean,W_noise=applySNMF.snmfTrain(data1,args1,oldW_clean=np.ones([1,1]),oldW_noise=np.ones([1,1]))
np.savetxt(fname="W_clean.csv", X=W_clean, fmt="%f",delimiter=",")
np.savetxt(fname="W_noise.csv", X=W_noise, fmt="%f",delimiter=",")
###单个训练或  多段合并后的训练尝试 多个语音段合并放入 end


#%%
"""   """
### 单个测试
W_clean = np.loadtxt(fname="W_clean.csv", dtype=np.float32, delimiter=",")
W_noise = np.loadtxt(fname="W_noise.csv", dtype=np.float32, delimiter=",")

applySNMF.loadWav(data1,args1,dirs1,noisyMatchListTrain,listi=0,snr=0) #读取一个测试数据
data1.magnitude_noisy=myVocalCordModel.preprocessVocalCord\
    (data1.magnitude_noisy,sr=data1.sr,fftSize=args1.fftSize,plotMask=False,A=1,C=1)  #EFF pre-process
applySNMF.snmfTest(data1,args1,W_clean,W_noise)
#data1.magnitude_est=myVocalCordModel.preprocessVocalCord\
#    (data1.magnitude_est,sr=data1.sr,fftSize=args1.fftSize,plotMask=False,A=0.3,C=1) #EFF post-process

applySNMF.synthesizeSpeech(data1,args1,dirs1)
### 单个测试 end
myPaint.paintFT(data1.magnitude_est,title="Spectrogram",MoP="M",dbYoN="Y")

#%%

"""   
### 批量化训练1by1 效果差
listRange=[0,2]
for i in range(listRange[0],listRange[1]):
    applySNMF.loadWav(data1,args1,dirs1,noisyMatchListTrain,listi=i,snr=0)
    if i==listRange[0]:
        W_clean=np.ones([1,1])
        W_noise=np.ones([1,1])
    else:
        W_clean = np.loadtxt(fname="W_clean.csv", dtype=np.float32, delimiter=",")
        W_noise = np.loadtxt(fname="W_noise.csv", dtype=np.float32, delimiter=",")
    
    W_clean,W_noise=applySNMF.snmfTrain(data1,args1,oldW_clean=W_clean,oldW_noise=W_noise)
    if i%1==0:
        np.savetxt(fname="W_clean.csv", X=W_clean, fmt="%f",delimiter=",")
        np.savetxt(fname="W_noise.csv", X=W_noise, fmt="%f",delimiter=",")

### 批量化训练1by1 end
"""


#%%


"""  
###批量化测试
W_clean = np.loadtxt(fname="W_clean.csv", dtype=np.float32, delimiter=",")
W_noise = np.loadtxt(fname="W_noise.csv", dtype=np.float32, delimiter=",")
for snr in [0,5,10]:
    for i in [2,3,6,7]:
        E_D_E=[(i&4)>>2,(i&2)>>1,i&1]
        tableName=str(E_D_E[0])+str(E_D_E[1])+str(E_D_E[2])+"_snr"+str(snr)
        
        applySNMF.measureWav(data1,args1,W_clean,W_noise,dirs1,noisyMatchListTest,\
                       listRange=[0,100],snr=snr,E_D_E=E_D_E,verbose=True,tableName=tableName)
###批量化测试 end
"""











