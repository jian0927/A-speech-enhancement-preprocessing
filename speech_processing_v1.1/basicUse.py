# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 19:11:04 2022

@author: Lenovo
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import shutil


def movWinAvg(y,aveWin=21):
    #aveWin=21 #平均窗口大小为 20 个间隔，对于 1024 的汉宁窗大小 和 16000 采样率，21左右对称
    #输出滑动窗平均 可以处理一维和二维数组
    theSize=np.array(y).shape
    if len(theSize)==1:#如果被滑动平均的是一维数组
        l=theSize[0]
    elif len(theSize)==2: #如果被滑动平均的是二维数组 是计算某行的所有列元
        l=theSize[1]
        
    aveWin=min(l-1,aveWin)
    aveWin=int(aveWin/2)*2+1
    effWins=np.ones(l)*aveWin#求平均时有效平均窗口大小的记录集
    
    effWins[0:int(aveWin/2+1)]=np.linspace(int(aveWin/2+1),aveWin,int(aveWin/2+1))
    effWins[-int(aveWin/2+1):]=np.linspace(aveWin,int(aveWin/2+1),int(aveWin/2+1))
    
    if len(theSize)==1:  #如果被滑动平均的是一维数组
        output=np.convolve(y,[1]*aveWin,'same')
        output= output/effWins#对平均值做纠正
    elif len(theSize)==2: #如果被滑动平均的是二维数组
        output=np.zeros([theSize[0],theSize[1]])
        for n in range(theSize[0]):
            output[n]=np.convolve(y[n],[1]*aveWin,'same')
            output[n]= output[n]/effWins#对平均值做纠正
    
    return output



# srcfile 需要复制、移动的文件   
# dstpath 目的地址
def mycopyfile(srcfile,dstpath,printMessage=True):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        if printMessage==True: print ("copy %s -> %s"%(srcfile, dstpath + fname))
 
 

# srcfile 需要复制、移动的文件   
# dstpath 目的地址
def mymovefile(srcfile,dstpath,printMessage=True):                       # 移动函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.move(srcfile, dstpath + fname)          # 移动文件
        if printMessage==True: print ("move %s -> %s"%(srcfile, dstpath + fname))
 



def sign(y):
    #很简单的函数 提取 y 的符号
    y[y>0]=1
    y[y<0]=-1
    
    return y


def dataTranslation(y,distance=0):
    #对一维或二维 y 数组进行平移操作距离 distance，移出的数据会消失，另一端留出的空位用0填补 二维在列上上下平移
    distance=int(distance)  #为写代码图轻松，就直接让其为整型，则在频率移动上固定为 15.625Hz 的整倍数
    y=np.array(y)
    
    theSize=np.array(y).shape
    l=theSize[0]

    if len(theSize)==1:                         #对一维数组的处理
        outy=np.zeros(l)
        if distance>=0:
            outy[distance:l]=y[0:l-distance]
        else:
            outy[0:l+distance]=y[-distance:l]
    elif len(theSize)==2:                       #对二维维数组一列上的平移处理
        outy=np.zeros([theSize[0],theSize[1]])
        if distance>=0:
            for i in range(theSize[1]):
                c=y[0:l-distance,i]
                outy[distance:l,i]=y[0:l-distance,i]
                
        else:
            for i in range(theSize[1]):
                outy[0:l+distance,i]=y[-distance:l,i]
        
    return outy


def dataScale(y,scaleFactor):
    #先对一维数据自原点开始缩放， 缩小时空出来的部分用0补充， 扩大时会移除远端数据
    #这里采用简单的一阶近似的办法缩放 ， 所以缩放的数据最好是比较平缓的 
    theSize=np.array(y).shape
    
    if len(theSize)==1:          #*********************************************#一维情况
        y=np.array(y)
        
        l=theSize[0]
        
        outy=np.zeros(l)
        
        n_max=int(l*scaleFactor)-2 #防止放大时出范围
        n_max=min(n_max,l-1)
        
        for n in range(1,n_max):
            c=int(n/scaleFactor) # c=int(kn)
            z=n/scaleFactor-c    # z=kn-c
            outy[n]=(1-z)*y[c]+z*y[c+1]
            outy[n]+=(y[c+2]+y[c+1]-y[c]-y[c-1])/4*(z**2)
    
    elif len(theSize)==2:        #*********************************************#二维情况
        outy=np.zeros([theSize[0],theSize[1]])    
        for i in range(theSize[1]):
            outy[:,i]=dataScale(y[:,i],scaleFactor)    #***********************#迭代
    
    
    return outy



def listToLin(listx,listy,limx):
    #将列出的一维的按顺序的数据点相连扩展成为连续的线
    x=np.linspace(limx[0],limx[1]-1,limx[1])
    y=np.zeros(limx[1])


    y[0:listx[0]]=[listy[0]]*listx[0]

    for i in range(len(listx)-1):
        y[listx[i]:listx[i+1]+1]=np.linspace(listy[i],listy[i+1],listx[i+1]-listx[i]+1)

    y[listx[-1]:]=[listy[-1]]*(limx[1]-listx[-1])
    
    return x,y


def enlargelists(positions,values,rangex=[0,513]):
    #将位置列表和数值列表组合扩张成线性连接的声谱图
    #rangex 是构成新的声谱图的纵向分量范围
    l=len(positions)
    enlargeV=np.zeros([rangex[1],l])
    
    for i in range(l):
        enlargeP,enlargeV[:,i]=listToLin(positions[i],values[i],[0,rangex[1]])

    return enlargeV


def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi) * sigma)






