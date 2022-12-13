import scipy.io.wavfile as wav
import numpy as np
import librosa
import os
import csv
from tqdm import tqdm

import tensorflow.compat.v1 as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, AveragePooling2D
import keras
from keras import metrics
#from keras.layers.normalization import BatchNormalization
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras import initializers
from keras.layers import LeakyReLU, PReLU, ELU
import ipykernel

import utils
import PaintTestUseEtc as myPaint  #用于绘制声谱图等
import generateNoisy as myNoisy #我自己补充的用于混合语音用的库
import myVocalCordModel
import measure



def getDnnModel(args):
    #输入DNN相关的参数，获得初始化的DNN模型
    keras.backend.clear_session()
    # LeakyReLU, PReLU, ELU, ThresholdedReLU, SReLU
    model = Sequential()
    model.add(Dense(args.hiddenUnitN, input_dim=args.inputDim, kernel_initializer='glorot_normal'))  
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(args.dropOut))

    model.add(Dense(args.hiddenUnitN, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(args.dropOut))
    
    model.add(Dense(args.hiddenUnitN, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(args.dropOut))
    
    model.add(Dense(units=args.outputDim, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # model.summary()

    
    return model



def dnnTrain(data,args,model,save=False):
    #这是采用 dB 来拟合计算
    X_train = np.log((abs(data.magnitude_noisy)+1)**2)
    Y_train = np.log(abs(data.magnitude_clean+1)**2)-X_train #这是依据 ny*exp(M(ln(ny)))=c 来训练
    #model.fit(..., verbose=0, callbacks=[TqdmCallback(verbose=2)])
    ######################### DNN training stage ##############################
    with tf.device('/gpu:0'):
        history = model.fit(X_train.T, Y_train.T, batch_size=args.batchN, \
                               epochs=args.epochN,verbose=0)
        
    if save==True:
        model.save(data.PATH_ROOT+"/myDNNModel.h5")
   
    return model,history

def dnnTest(data,args,model):
    #这是采用 dB 来拟合计算
    X_test  = np.log((abs(data.magnitude_noisy)+1)**2)
    theSize = data.magnitude_noisy.shape

    ######################## DNN predict ##############################
    dnnOutPut=model.predict(X_test.T).T
    
    from numpy.linalg import norm
    ########################## Enhancement stage ##############################
    speech_masks = np.sqrt(np.exp(dnnOutPut))
    speech_masks = (speech_masks)/(norm(speech_masks,2,axis=0)/np.sqrt(theSize[0]))
    
    data.magnitude_est = data.magnitude_noisy * speech_masks   
    #\原带噪谱与掩膜结合
    
    data.magnitude_est *= norm(data.magnitude_noisy)/norm(data.magnitude_est)  
    #\强度统一
    
    data.phase_est=data.phase_clean
    
    #myPaint.paintFT(speech_masks,title="mask",MoP="M",dbYoN="N") #绘制
    #myPaint.paintFT(data.magnitude_est,title="est",MoP="M",dbYoN="Y") #绘制

def loadWav(data,args,dirs,noisyMatchList,listi=0,snr=0):
    #读取数据库中的一对speech和noise （从noisyMatchList[1,2][listi]）获知，
    #以 snr 组装（按照speech.wav 的尺寸）
    
    path_clean = dirs.datasets_dir + "/speech/" + noisyMatchList[0][listi] 
    path_noise = dirs.datasets_dir + "/noise/" + noisyMatchList[1][listi]
    #\这里默认的是datasets下级有文件夹speech和noise
    
    data.sr,data.wav_clean          = wav.read(path_clean)
    _,data.wav_noise                = wav.read(path_noise)
    #data.wav_noisy = myNoisy.signal_by_db(data.wav_clean, data.wav_noise, snr, handle_method="noise_append")
    data.wav_noisy,_,data.wav_noise = myNoisy.signal_by_db\
        (data.wav_clean, data.wav_noise, snr, handle_method="noise_append",allOut=True)
    #\组合的尺寸按照clean 的大小
    
    data.magnitude_noisy,data.phase_noisy    = utils.wav_stft(data.wav_noisy, args)
    data.magnitude_clean,data.phase_clean    = utils.wav_stft(data.wav_clean, args)
    data.magnitude_noise,data.phase_noise    = utils.wav_stft(data.wav_noise, args)
    #\在仅仅训练DNN到干净语音的映射，不需要添加噪声的声谱图
    
    wav.write(filename=dirs.write_noisy_path,rate=data.sr,data=data.wav_noisy)  #测试着写来用
    wav.write(filename=dirs.write_clean_path,rate=data.sr,data=data.wav_clean)  #测试PESQ着写来用 干净
    wav.write(filename=dirs.write_noise_path,rate=data.sr,data=data.wav_noise)  #测试PESQ着写来用 干净
    
    #myPaint.paintFT(data.magnitude_noisy,title="noisy",MoP="M",dbYoN="Y") #绘制声谱图 带噪声语音
    
    return 0



def synthesizeSpeech(data,args,dirs):
    # Reconstruction 输出est
    stft_reconstructed_clean = utils.merge_magphase(data.magnitude_est, data.phase_est)
    x=librosa.istft(stft_reconstructed_clean, hop_length=args.hopSize, window=args.window)
    
    signal_reconstructed_clean=np.zeros(len(data.wav_noisy))
    signal_reconstructed_clean[:len(x)]=x  #对尾端一帧不到的内容取静音，以保证长度与原文件相同
    
    data.wav_est = signal_reconstructed_clean.astype('int16')
    wav.write(dirs.write_est_path,data.sr,data.wav_est)
    
    return 0



def measureWav(data,args,dirs,model,noisyMatchList,listRange=[0,100],snr=0,E_D_E=[0,1,0],verbose=False,tableName=""):
   #用来测试的，特别需要注意的是 E_D_E=[0,1,0]，表示基频强调和DNN开关
   #tableName="" 时是不记录表格的
   ###用测试集来测试###########################
   listRange[1]=min(listRange[1],len(noisyMatchList[0]))
   measureTest=np.zeros([listRange[1],4])
   for i in range(listRange[1]):
       
       loadWav(data,args,dirs,noisyMatchList,listi=i,snr=snr) 
       #\加载语音#这仅仅只是加载了手动输入部分的声音，若要批量化加载，需要嵌套在DNN中才比较方便
       
       if E_D_E[0]==1:
           data.magnitude_noisy=myVocalCordModel.preprocessVocalCord\
               (data.magnitude_noisy,sr=data.sr,fftSize=args.fftSize,plotMask=False) 
               #\基于声带谐波模型做一次预处理
       if E_D_E[1]==1:    
           dnnTest(data,args,model) #DNN掩膜语音增强 （在声谱图层面）
       if E_D_E[2]==1:
           data.magnitude_est=myVocalCordModel.preprocessVocalCord\
               (data.magnitude_est,sr=data.sr,fftSize=args.fftSize,plotMask=False)   #-EFF 
               #\基于声带谐波模型做一次后处理
       
       synthesizeSpeech(data,args,dirs) #对增强语音合成 （声谱图处理为波形并存储）
       
       ref = data.wav_clean
       deg = data.wav_noisy
       est = data.wav_est
       rate = data.sr
   
       measureTest[i][0]=measure.pesq(rate, ref, deg, 'wb')
       measureTest[i][1]=measure.pesq(rate, ref, est, 'wb')
       measureTest[i][2]=measure.stoi(ref, deg, rate, extended=False)
       measureTest[i][3]=measure.stoi(ref, est, rate, extended=False)
       
       #basicUse.mymovefile("dataSets2/noise.wav","dataSets2/test/"+str(i)+"/")
       #basicUse.mymovefile("dataSets2/clean.wav","dataSets2/test/"+str(i)+"/")
       #basicUse.mymovefile("dataSets2/noisy.wav","dataSets2/test/"+str(i)+"/")
       #basicUse.mymovefile("dataSets2/outPut.wav","dataSets2/test/"+str(i)+"/")
       
       if verbose==True:
           print("test: "+str(i))
   if tableName!="":
        headers=["noisy PESQ","est PESQ","noisy STOI","est STOI"]
        measureAvg=sum(measureTest)/(listRange[1]-listRange[0])
        #np.savetxt(fname=dirs.output_dir+"/"+tableName+".csv", X=measureTest, fmt="%f",delimiter=",")
        with open(dirs.output_dir+"/"+tableName+".csv","w",encoding="utf-8",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(measureTest)
            writer.writerow([])
            writer.writerow(measureAvg)
   return measureTest


















