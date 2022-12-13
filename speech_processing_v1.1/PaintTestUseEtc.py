# coding = utf-8
import numpy as np
from matplotlib import pyplot as plt
import librosa
#import scipy
import scipy.io.wavfile as wavfile



def paintFT(FT,sr=16000,MoP="P",dbYoN="Y",hop_length=256,title="Spectrogram",figureSize=(16,10),printType="hz",outFigure=False,figPlace=111,ref=np.max):
    #读取声谱图直接就绘制 声谱图 绘制的是能量的dB模式， 如果 MoP!="power" 那么就只是幅度的dB模式
    #outFigure=False,figPlace=111 其中outFigure用于外部还有其他声谱图一起叠加时使用，figPlace是此时的位置
    if outFigure==False: plt.figure(figsize=figureSize)

    plt.subplot(figPlace)
    spec = FT
    
    if MoP=="P" : spec=spec**2
    if dbYoN=="Y":
        spec = np.abs(FT)
        minV = np.min(FT)
        if minV==0:
            FT = FT+1  #用来确保在转换为分贝谱显示的时候不会出现小于0的数
        spec = librosa.amplitude_to_db(spec, ref=ref)
    
    try:
        librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis=printType,hop_length=hop_length);
        
        plt.title(title)
        plt.colorbar()
        
        if outFigure==False: plt.show()
    except:
        print("没有成功绘图")
    
    return 0


def paintWav(signal,sr,title="wav"):
    plt.figure(figsize=(10,10))
    
    plt.subplot(3, 1, 1)
    signal = signal.astype('float')   #astype：强制类型转换
    librosa.display.waveshow(signal, sr=sr)
    plt.title(title)

    


#主要绘制 声谱图 的幅度数值分布
def paintDistributionFT(FT,bins=200,disPlayRange=(0,10000)):
    A=np.array(FT).flatten()
    plt.hist(A, bins, density=0, facecolor='g', alpha=0.75, range=disPlayRange)#准备绘画直方图
    plt.grid(True)
    plt.title("clean")
    plt.show()#把图显示出来

def amountCount(data,bins=10,theRange="all"):
    """
    Parameters
    ----------
    data : 一维列表
        被统计的数据
    bins : 正整数 ,可选
        The default is 10. 划分统计数量的区间数
    theRange :  optional "all" 或者 (a,b)
        The default is "all". 统计范围

    Returns
    -------
    n :  一维列表
        统计后数量分布
    binDis :  一维列表
         各统计区间中点
    与hist相同的功能，但这个是画点

    """
    import numpy as np
    if theRange=="all":
        theRange=(np.min(data),np.max(data)+1e-8)
    
    dn=(theRange[1]-theRange[0])/bins
    
    binDis=np.linspace(theRange[0]+dn/2,theRange[1]-dn/2,bins)
    n=[0 for i in range(bins)]
    
    for i in range(len(data)):
        if data[i]>theRange[0] and data[i]<theRange[1]:
            n[int((data[i]-theRange[0])/(theRange[1]-theRange[0])*bins)]+=1
        
        
    return n,binDis,theRange

