import scipy.io.wavfile as wav
import numpy as np
import librosa
import os
import csv
from tqdm import tqdm


import utils
import PaintTestUseEtc as myPaint  #用于绘制声谱图等
import generateNoisy as myNoisy #我自己补充的用于混合语音用的库
import myVocalCordModel
import measure


def normalizeW(W):
    eps = 1e-15
    Q = np.sqrt(sum(W**2,0));
    W = W/np.tile(Q+eps,[W.shape[0],1]);
    return W

from numpy.linalg import norm
def ls_update(V,W,H,d,nuH,nuW,cost_old,accel,lambda1,Rec,updateW):
    eps = 1e-15
    
    # Update H
    if accel>1:
        H_old=H;
        grad = (np.matmul(np.transpose(W)*V))/(np.matmul(np.transpose(W),Rec)+eps+lambda1+lambda1);
        while True:
            H = H_old*(grad**nuH);
            Rec = np.matmul(W,H); 
            sse = norm(V-Rec)**2;
            cost = 0.5*sse+lambda1*sum(sum(H));
            if cost>cost_old: nuH = max(nuH/2,1);
            else: nuH = nuH*accel; break;
            
        cost_old = cost;
    else:
        H = H*(np.matmul(np.transpose(W),V))/(np.matmul(np.transpose(W),Rec)+eps+lambda1);
        Rec = np.matmul(W,H); 
        sse = norm(V-Rec)**2;
        cost = 0.5*sse+lambda1*sum(sum(H));
        cost_old = cost;
        
     
    # Update W
    if updateW != 'off':
        if accel>1:
            W_old=W;
            Wxa = np.matmul(V,np.transpose(H));
            Wya = np.matmul(Rec,np.transpose(H));
            Wx = Wxa + np.tile(sum(Wya*W),[W.shape[0],1])*W;
            Wy = Wya + np.tile(sum(Wxa*W),[W.shape[0],1])*W;
            grad = Wx/(Wy+eps);
            while True:
                W = normalizeW(W_old*(grad**nuW));
                Rec = np.matmul(W,H); 
                sse = norm(V-Rec)**2;
                cost = 0.5*sse+lambda1*sum(sum(H));
                if cost>cost_old: nuW = max(nuW/2,1);
                else: nuW = nuW*accel; break;
            
            cost_old = cost;
        else:
            Wxa = np.matmul(V,np.transpose(H));
            Wya = np.matmul(Rec,np.transpose(H));
            Wx = Wxa + np.tile(sum(Wya*W),[W.shape[0],1])*W;
            Wy = Wya + np.tile(sum(Wxa*W),[W.shape[0],1])*W;
            W = normalizeW(W*Wx/(Wy+eps));
            Rec = np.matmul(W,H); 
            sse = norm(V-Rec)**2;
            cost = 0.5*sse+lambda1*sum(sum(H));
            cost_old = cost;
        
    return W,H,nuH,nuW,cost,sse,Rec

#% Kullback Leibler update function
def kl_update(V,W,H,d,nuH,nuW,cost_old,accel,lambda1,Rec,updateW):
    eps=1e-15
    #% Update H
    if accel>1:
        H_old=H;
        VR = V/(Rec+eps);
        O = np.ones(V.shape);
        grad = (np.matmul(np.transpose(W),VR))/(np.matmul(np.transpose(W),O)+eps+lambda1+lambda1);
        while True:
            H = H_old*(grad**nuH);
            Rec = np.matmul(W,H); 
            ckl = sum(sum(V*np.log((V+eps)/(Rec+eps))-V+Rec));
            cost = ckl + lambda1*(sum(sum(abs(H))));
            if cost>cost_old: nuH = max(nuH/2,1);
            else: nuH = nuH*accel; break;
        cost_old = cost;
    else:
        H = H*(np.matmul(np.transpose(W),(V/(Rec+eps))))/(np.matmul(np.transpose(W),np.ones(V.shape))+eps+lambda1);
        Rec = np.matmul(W,H); 
        ckl = sum(sum(V*np.log((V+eps)/(Rec+eps))-V+Rec));
        cost = ckl + lambda1*(sum(sum(abs(H))));
        cost_old = cost;
    
     
    #% Update W
    if updateW!="off":
        if accel>1:
            W_old=W;
            Wxa = np.matmul((V/(Rec+eps)),np.transpose(H));
            Wya = np.matmul(np.ones(V.shape),np.transpose(H));
            Wx = Wxa + np.tile(sum(Wya*W),[W.shape[0],1])*W;
            Wy = Wya + np.tile(sum(Wxa*W),[W.shape[0],1])*W;
            grad = Wx/(Wy+eps);
            while True:
                W = normalizeW(W_old*(grad**nuW));
                Rec = np.matmul(W,H); 
                ckl = sum(sum(V*np.log((V+eps)/(Rec+eps))-V+Rec));
                cost = ckl + lambda1*(sum(sum(abs(H))));
                if cost>cost_old: nuW = max(nuW/2,1);
                else: nuW = nuW*accel; break; 
            cost_old = cost;
        else:
            Wxa = np.matmul((V/(Rec+eps)),np.transpose(H));
            Wya = np.matmul(np.ones(V.shape),np.transpose(H));
            Wx = Wxa + np.tile(sum(Wya*W),[W.shape[0],1])*W;
            Wy = Wya + np.tile(sum(Wxa*W),[W.shape[0],1])*W;
            W = normalizeW(W*Wx/(Wy+eps));
            Rec = np.matmul(W,H); 
            ckl = sum(sum(V*np.log((V+eps)/(Rec+eps))-V+Rec));
            cost = ckl + lambda1*(sum(sum(abs(H))));
            cost_old = cost;
    sse = norm(V-Rec)**2;
    return W,H,nuH,nuW,cost,sse,Rec


from numpy.linalg import norm
def snmf(V, args):
    """
    % SNMF Sparse non-negative matrix factorization with adaptive mult. updates
    % 
    % Usage:
    %   [W,H] = snmf(V,d,[options])
    %
    % Input:
    %   V                 M x N data matrix
    %   d                 Number of factors
    %   args -- options
    %     .costfcn        Cost function to optimize
    %                       'ls': Least squares (default)
    %                       'kl': Kullback Leibler
    %     .W              Initial W, array of size M x d
    %     .H              Initial H, array of size d x N 
    %     .lambda         Sparsity weight on H
    %     .updateW        Update W [<on> | off]
    %     .maxiter        Maximum number of iterations (default 100)
    %     .conv_criteria  Function exits when cost/delta_cost exceeds this
    %     .plotfcn        Function handle for plot function
    %     .plotiter       Plot only every i'th iteration
    %     .accel          Wild driver accelleration parameter (default 1)
    %     .displaylevel   Level of display: [off | final | <iter>]
    % 
    % Output:
    %   W                 M x d
    %   H                 d x N
    %
    % Example I, Standard NMF:
    %   d = 4;                                % Four components
    %   [W,H] = snmf(V,d);
    %
    % Example I, Sparse NMF:
    %   d = 2;                                % Two components
    %   args.costfcn = 'kl';                  % Kullback Leibler cost function
    %   args.lambda = 0.1;                   % Sparsity
    %   [W,H] = snmf(V,d,args);
    % 
    % Authors:
    %   Mikkel N. Schmidt and Morten M鴕up, 
    %   Technical University of Denmark, 
    %   Institute for Matematical Modelling
    %
    % References:
    %   [1] M. M鴕up and M. N. Schmidt. Sparse non-negative matrix factor 2-D 
    %       deconvolution. Technical University of Denmark, 2006.
    %   [2] M. N. Schmidt and M. M鴕up. Nonnegative matrix factor 2-D 
    %       deconvolution for blind single channel source separation. 
    %       ICA, 2006.
    %   [3] M. N. Schmidt and M. M鴕up. Sparse non-negative matrix factor 2-d 
    %       deconvolution for automatic transcription of polyphonic music. 
    %       Submitted to EURASIP Journal on Applied Signal Processing, 2006.
    %   [4] P. Smaragdis. Non-negative matrix factor deconvolution; 
    %       extraction of multiple sound sourses from monophonic inputs. 
    %       ICA 2004.
    %   [5] J. Eggert and E. Korner. Sparse coding and nmf. In Neural Networks,
    %       volume 4, 2004.
    %   [6] J. Eggert, H. Wersing, and E. Korner. Transformation-invariant 
    %       representation and nmf. In Neural Networks, volume 4, 2004.
     
    % -------------------------------------------------------------------------
    % Parse input arguments
    """
    
    
    costfcn = args.costfcn# mgetopt(opts, 'costfcn', 'ls', 'instrset', {'ls','kl'});
    if args.W.shape != (V.shape[0],args.d): args.W=np.random.rand(V.shape[0],args.d)
    W = args.W#  mgetopt(opts, 'W', rand(size(V,1),d));
    args.H = np.random.rand(args.d,V.shape[1])
    H = args.H # mgetopt(opts, 'H', rand(d,size(V,2)));
    W = normalizeW(W);
    updateW = args.updateW # mgetopt(opts, 'updateW', 'on', 'instrset', {'on','off'});
    nuH = args.nuH # mgetopt(opts, 'nuH', 1);
    nuW = args.nuW # mgetopt(opts, 'nuW', 1);
    lambda1 = args.lambda1 # mgetopt(opts, 'lambda', 0);
    maxiter = args.maxiter # mgetopt(opts, 'maxiter', 400);
    conv_criteria = args.conv_criteria # mgetopt(opts, 'conv_criteria', 1e-4);
    accel = args.accel # mgetopt(opts, 'accel', 1);
    plotfcn = args.plotfcn # mgetopt(opts, 'plotfcn', []);
    plotiter = args.plotiter # mgetopt(opts, 'plotiter', 1);
    displaylevel = args.displaylevel # mgetopt(opts, 'displaylevel', 'iter', 'instrset', {'off','iter','final'});
    
    
    updateWRows = args.updateWRows # mgetopt(opts, 'updateWRows', []);
    eps=1e-15
    
     
    # -------------------------------------------------------------------------
    # Initialization
    sst = sum(sum((V-np.mean(np.mean(V)))**2));
    Rec = np.matmul(W,H); 
    if costfcn=='ls':
        sse = norm(V-Rec)**2;
        cost_old = 0.5*sse + lambda1*(sum(sum(abs(H))));
    elif costfcn=='kl':
        ckl = sum(sum(V*np.log((V+eps)/(Rec+eps))-V+Rec));
        cost_old = ckl + lambda1*(sum(sum(abs(H))));
    
    delta_cost = 1;
    iterN = 0;
    keepgoing = True;
    

    while keepgoing:
        if iterN % args.displayIterN==0:
            print("iter:",str(iterN))
        # Update H and W
        if costfcn=='ls':
            W,H,nuH,nuW,cost,sse,Rec = ls_update(V,W,H,args.d,nuH,nuW,cost_old,accel,lambda1,Rec,updateW);
        elif costfcn== 'kl':
            W,H,nuH,nuW,cost,sse,Rec = kl_update(V,W,H,args.d,nuH,nuW,cost_old,accel,lambda1,Rec,updateW);
     
        delta_cost = cost_old - cost;
        cost_old=cost;
        iterN=iterN+1;
     
        # Check if we should stop
        if delta_cost<cost*conv_criteria :
            # Small improvement with small step-size
            if nuH<=accel & nuW<=accel:
                #if displaylevel == 'iter':
                #    print("C2NMF has converged, iter:",str(iterN));
                
                keepgoing = False;
            # Small improvement - maybe because of too large step-size?
            else:
                nuH = 1; nuW = 1;
            
        
        # Reached maximum number of iterations
        if iterN>=maxiter:
            #if displaylevel == 'iter':
            #    print("Maximum number of iterations reached")
            
            keepgoing=False; 
    return W,H,cost



def snmfTrain(data,args,oldW_clean=np.ones([1,1]),oldW_noise=np.ones([1,1])):
    #用于基础字典的继续训练
    args.updateW="on"
    args.W=np.copy(oldW_clean)
    W_clean_new, H_clean,_=snmf(data.magnitude_clean+1, args)
    args.W=np.copy(oldW_noise)
    W_noise_new, H_noise,_=snmf(data.magnitude_noise+1, args)
    
    data.magnitude_estimated_clean=np.matmul(W_clean_new,H_clean)

    
    return W_clean_new,W_noise_new

def snmfTest(data,args,W_clean,W_noise):
    
    W_noisy = np.concatenate([W_clean,W_noise], axis=1)
    args.W=W_noisy
    args.updateW="off"
    
    args.d=int(args.d*2)
    W_test,H_reconstructed_noisy,cost=snmf(data.magnitude_noisy+1, args)
    args.d=int(args.d/2)
    
    H_reconstructed_clean = H_reconstructed_noisy[:args.d,:]
    H_reconstructed_noise = H_reconstructed_noisy[args.d:,:]
    
    magnitude_reconstructed_clean =np.matmul(W_clean,H_reconstructed_clean)
    magnitude_reconstructed_noise = np.matmul(W_noise,H_reconstructed_noise)
     
    #Gain function similar to wiener filter to enhance the speech signal
    wiener_gain = np.sqrt((np.power(magnitude_reconstructed_clean,2)+1) / 
                          (1+np.power(magnitude_reconstructed_clean,2) + np.power(magnitude_reconstructed_noise, 2)))
    data.magnitude_est = wiener_gain * data.magnitude_noisy   #masky
    
    data.phase_est     = data.phase_noisy     
    
    #myPaint.paintFT(magnitude_reconstructed_clean,title="reconstructed_clean",MoP="P",dbYoN="Y") #绘制声谱图 带噪声语音
    #myPaint.paintFT(magnitude_reconstructed_noise,title="reconstructed_noise",MoP="P",dbYoN="Y") #绘制声谱图 带噪声语音
    #myPaint.paintFT(wiener_gain,title="mask",MoP="M",dbYoN="N") #绘制声谱图 带噪声语音


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
    
    wav.write(filename=dirs.write_noisy_path,rate=data.sr,data=data.wav_noisy)  #测试着写来用
    wav.write(filename=dirs.write_clean_path,rate=data.sr,data=data.wav_clean)  #测试PESQ着写来用 干净
    wav.write(filename=dirs.write_noise_path,rate=data.sr,data=data.wav_noise)  #测试PESQ着写来用 干净
    
    #myPaint.paintFT(data.magnitude_noisy,title="noisy",MoP="M",dbYoN="Y") #绘制声谱图 带噪声语音
    
    return 0


def loadWavRange(data,args,dirs,noisyMatchList,listRange=(0,100),snr=0):  
    #这有些修改，采用整体导入合成一个大的声谱图后进行训练
    #listRange 左闭右开
    l=int(listRange[1]-listRange[0])
    
    data.wav_clean=[]
    data.wav_noise=[]
    data.wav_noisy=[]
    
    for listi in range(l):        
        path_clean = dirs.datasets_dir + "/speech/" + noisyMatchList[0][listi] 
        path_noise = dirs.datasets_dir + "/noise/" + noisyMatchList[1][listi]
        #\这里默认的是datasets下级有文件夹speech和noise
        
        data.sr,cleanWav          = wav.read(path_clean)
        _,noiseWav                = wav.read(path_noise)
        noisyWav,_,noiseWav       = myNoisy.signal_by_db\
            (cleanWav, noiseWav, snr, handle_method="noise_append",allOut=True)
        
        data.wav_clean.extend(cleanWav)
        data.wav_noise.extend(noiseWav)
        data.wav_noisy.extend(noisyWav)
        
        print("现在装载完成了:"+str(listi))
    data.wav_clean=np.array(data.wav_clean)
    data.wav_noise=np.array(data.wav_noise)
    data.wav_noisy=np.array(data.wav_noisy)  
    
    data.magnitude_noisy,data.phase_noisy    = utils.wav_stft(data.wav_noisy, args)
    data.magnitude_clean,data.phase_clean    = utils.wav_stft(data.wav_clean, args)
    data.magnitude_noise,data.phase_noise    = utils.wav_stft(data.wav_noise, args)

    wav.write(filename=dirs.write_noisy_path,rate=data.sr,data=data.wav_noisy)  #测试着写来用
    wav.write(filename=dirs.write_clean_path,rate=data.sr,data=data.wav_clean)  #测试PESQ着写来用 干净
    wav.write(filename=dirs.write_noise_path,rate=data.sr,data=data.wav_noise)  #测试PESQ着写来用 干净
    
    #myPaint.paintFT(data.magnitude_noisy_train,title="clean",MoP="M",dbYoN="Y") #绘制声谱图 带噪声语音

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



def measureWav(data,args,W_clean,W_noise,dirs,noisyMatchList,\
               listRange=[0,100],snr=0,E_D_E=[0,1,0],verbose=False,tableName=""):
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
           snmfTest(data,args,W_clean,W_noise) #SNMF掩膜语音增强 （在声谱图层面）
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


















