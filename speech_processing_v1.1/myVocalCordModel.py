import numpy as np
from matplotlib import pyplot as plt
import scipy
import PaintTestUseEtc as myPaint  #我自己补充的，用于直观展示和比较等
import basicUse

def preprocessVocalCord(spec,sr=16000,fftSize=1024,plotMask=False,A=0.3,C=1):
    """
    This is a function of the prominent voiceprint designed with the characteristic 
    of evenly spaced voiceprint by vocal cord vibration
    
    require:
        the sampling rate of sound should be 16k Hz
        and the width of Hanning window should be 1024, and the hop should be 256 
    
    当前功能：
    0.抓取样本的峰
    1.基频拟合
    2.根据基频判断是否有语音
    3.较高频段频率线性滑动
    
    根据模型函数构成软掩膜
    
    不太可靠的：
    根据声音强度判断有没有语音
    根据估计参数的方差判断噪声的干扰情况

    Parameters
    ----------
    spec : 2D npArray
        Spectrograms to be processed

    Returns
    -------
      reSpec:    2D npArray
        return spectrograms
        

    """
    #这是用声带振动有等间隔声纹的特点设计的突出声纹的函数
    # spec 是我们要强调用的声谱图， coef 是为了避免过曝等原因，在最后掩膜处乘的系数
    
    theSize=spec.shape # 需要处理的声谱图 的尺寸
    noisydB=np.log((spec+1)**2)  #化为分贝谱
    
    ######为获得样本的峰位置

    noisydB_ave_move=basicUse.movWinAvg(noisydB.T,int(350/sr*fftSize)).T  #计算噪声的滑动平均

    stNoisydB=(noisydB+1)/(noisydB_ave_move+1)  #现在已经对分贝谱标准化
    
    peakPositions=[0]*theSize[1]
    for j in range(theSize[1]):
        h_sigma=np.std(stNoisydB[:,j])
        peaks,_ = scipy.signal.find_peaks(stNoisydB[:,j], height=1+h_sigma, distance=3)
        peakPositions[j]=peaks       #抓取到样本的峰的位置
    
    #####下面是做模型与样本的匹配
    from scipy.optimize import curve_fit
    
    p_fit=np.zeros([theSize[1],2]) #记录原声谱图各帧拟合函数的参数
    p_cov=[0]*theSize[1]
    initT=np.zeros(theSize[1])  #记录拟合函数用到的初始参数，由傅里叶变换取最大得到的
    
    
    
    def myModelFun(x,T,fai):
        #这个模型是初步判断其基频用的
        return np.cos(2*np.pi*(1/T)*x+fai)

    def searchBestInitT(seg,maxRangeAdd=(5,-1)):
        #查找最优的初始间隔设定，用来作为我设计的函数的初始参数去拟合某帧峰位置，这里的拟合其实起到的是将峰位置统合降参
        # seg 是某帧的数据片段， maxSeatStarti 是查最优间隔时圈定的范围增量
        from scipy.fftpack import fft
        N=np.array(seg).shape[0]
        FFT_y=np.abs(fft(seg))
        
        maxSeat=np.argmax(FFT_y[maxRangeAdd[0]:int(N/2+maxRangeAdd[1])])+maxRangeAdd[0]
        bestInitT=N/maxSeat
        
        return bestInitT

    noSpeechPcov=np.array([[int(400/sr*fftSize),0],[0,6.28]])
    
    x=np.linspace(0,theSize[0],theSize[0])
    modelM=np.zeros((theSize[0],theSize[1]))-1 #记录模型在某帧的曲线
    
    #开始对每一帧做模型拟合
    allfev=0
    for i in range(theSize[1]):
        peakPositionsPart=peakPositions[i][peakPositions[i]<int(3000/sr*fftSize)]  #大约到 3000 Hz
        if len(peakPositionsPart)<3: #如果3k Hz内 抓取的峰的数量太少，不做模型拟合，认为没有语音
            p_cov[i]=noSpeechPcov
            continue
        
        y0=np.ones(peakPositionsPart.shape[0]) #拟合的峰高度统一为1，拟合的是位置
        
        initT[i]=searchBestInitT(stNoisydB[:int(3000/sr*fftSize),i]) #获取较好的初始间隔

        try:    fitOutPut=curve_fit(myModelFun,peakPositionsPart,y0,[initT[i],0],maxfev=5000,xtol=0.01,ftol=0.01,full_output=True)#保证运行的稳定
        except: fitOutPut[0][0]=0; 
        if fitOutPut[0][0]<int(70/sr*fftSize) or fitOutPut[0][0]>int(400/sr*fftSize): #如果拟合出来的结果超出范围约 70~400 Hz 被认为没有语音
            p_cov[i]=noSpeechPcov
            continue
        p_fit[i][0:2],p_cov[i],nfev=fitOutPut[0],fitOutPut[1],fitOutPut[2]['nfev'] #在合理范围内被视为有语音，赋予拟合结果
        
        #对有语音的片段放入增益曲线  注意：这里就判断了哪里没有语音，若有语音就添加增益曲线
        #如果预先设定没有稳态噪声问题 即 steady==0 同样要添加增益曲线
        
        modelM[:,i]=myModelFun(x,p_fit[i][0],p_fit[i][1])  #我的模型函数掩膜
        #modelM[:,i]=basicUse.sign(modelM[:,i])*(modelM[:,i]**2)
            
        
    ####### 下面构建 mask
    singleFev=allfev/theSize[1] #对单个声谱图做模型的拟合处理花费的拟合迭代次数
    preMask=(modelM*A+C)/(A+C)
    preMask[preMask<0]=0
    #preMask[preMask>2]=2
    

    
    if plotMask==True:
        myPaint.paintFT(preMask,MoP="M",dbYoN="N",title="my model mask")
    

    newSpec=spec*preMask
    
    return newSpec


from scipy.fftpack import ifft
def getPitch(magSpec, Fs, p_max=400, p_min=100):
    """
    另一套采用倒谱法查找基频的算法
    通过幅度谱 magSpec, 得到各个帧的基频, 采取倒谱法
    获取一段音频的基音频率 fp(Hz) = fs/Np 
    步骤：分帧、hamming窗、倒谱c(n)、求Np
    Np是倒谱上最大峰值和次峰值之间的采样点数
    
    Args:
        magSpec (ndarray): 音频信号的幅度谱，shape为(N, frames)，N为FFT点数，frames为帧数
        Fs (int): 采样率
        p_max : 基音范围内最高 音调
        p_min : 基音范围内最低 音调
        
    Returns:
        ndarray: 各帧的基频，shape为(frames-1,)
    """
    frames = magSpec.shape[1]  # 总帧数
    LNp = int(np.floor(Fs/p_max))  # 基音周期的范围
    HNp = int(np.floor(Fs/p_min))   # 基音周期最多可能有多少个采样点数
    pitchf = np.zeros(frames)

    for m in range(frames):
        lgS1 = np.log(np.abs(magSpec[:,m]) + np.finfo(float).eps)  # 傅里叶变换后取模,再取对数
        lgS2 = lgS1[1:]
        lgS = np.concatenate((lgS1, np.flipud(lgS2)))

        cn = ifft(lgS)  # 得到x(n)的倒谱c(n) 
        lenc = int(np.ceil(len(cn)/2))  # 圆周共轭,为减少运算取一半
        if HNp > lenc:
            HNp = lenc
        c = cn[LNp:HNp]  # 在合适范围内搜索Np
        idx = np.argmax(c)  # 搜索出max
        maxcn = c[idx]
        if maxcn > 0.08:  # 门限设置为0.08
            pitchN = LNp + idx
            pitchf[m] = Fs/pitchN

    return pitchf


def remove_silent_frames(x, y, dyn_range, framelen, hop):
    """ Remove silent frames of x and y based on x
    A frame is excluded if its energy is lower than max(energy) - dyn_range
    The frame exclusion is based solely on x, the clean speech signal
    # Arguments :
        x : array, original speech wav file
        y : array, denoised speech wav file
        dyn_range : Energy range to determine which frame is silent
        framelen : Window size for energy evaluation
        hop : Hop size for energy evaluation
    # Returns :
        x without the silent frames
        y without the silent frames (aligned to x)
    """
    # Compute Mask
    w = np.hanning(framelen + 2)[1:-1]

    x_frames = np.array(
        [w * x[i:i + framelen] for i in range(0, len(x) - framelen, hop)])
    y_frames = np.array(
        [w * y[i:i + framelen] for i in range(0, len(x) - framelen, hop)])

    # Compute energies in dB
    x_energies = 20 * np.log10(np.linalg.norm(x_frames, axis=1) + 1)

    # Find boolean mask of energies lower than dynamic_range dB
    # with respect to maximum clean speech energy frame
    mask = (np.max(x_energies) - dyn_range - x_energies) < 0

    # Remove silent frames by masking
    x_frames = x_frames[mask]
    y_frames = y_frames[mask]



    return mask

def peaksPositionGet(spec,fftSize=1024,sr=16000):
    #从声谱图中提取基频和谐波峰位置，返回峰位置列表
    theSize=spec.shape
    peaksPositions=[0]*theSize[1]
    peaksValues=[0]*theSize[1]
    fStep=sr/fftSize
    
    specdB=np.log((spec+1)**2)  #化为分贝谱
    ######为获得样本的峰位置

    specdBAvgW=basicUse.movWinAvg(specdB.T,int(500/fStep)).T  #计算带噪声的滑动平均
    specdBAvgN=basicUse.movWinAvg(specdB.T,int(50/fStep)).T  #计算带噪声的滑动平均
    stSpecdB=(specdBAvgN+1e-8)-(specdBAvgW+1e-8)  #现在已经对分贝谱标准化
    

    for j in range(theSize[1]):
        h_sigma=np.std(stSpecdB[:,j])
        peaks,_ = scipy.signal.find_peaks(stSpecdB[:,j], height=h_sigma, distance=1)
        peaksPositions[j]=peaks       #抓取到样本的峰的位置
        
    return peaksPositions


def peaksPositionToIBM(peaksPositions,fftSize=1024,coverR=1):
    #根据峰位置peaksPositions和每个峰在频率上掩盖范围coverR，构建IBM掩膜
    theSize=[int(fftSize/2+1),len(peaksPositions)]
    IBM=np.zeros(theSize)
    IBM2=np.zeros([theSize[0]+coverR,theSize[1]])#多出来几行是为了赋值时不做边界判断
    
    coverV=np.ones(coverR*2+1)
    for i in range(theSize[1]):
        for j in range(len(peaksPositions[i])):
            cj=peaksPositions[i][j]
            IBM2[cj-coverR:cj+coverR+1,i]=coverV
        IBM[:,i]=IBM2[:theSize[0],i]
    
    return IBM




















