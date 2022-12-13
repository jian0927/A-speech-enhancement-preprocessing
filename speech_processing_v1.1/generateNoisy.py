import numpy as np
from matplotlib import pyplot as plt
import librosa
#import scipy
import scipy.io.wavfile as wavfile



def SNR(x1, x2):
    from numpy.linalg import norm
    return 20 * np.log10(norm(x1) / norm(x2))


def signal_by_db(x1, x2, snr,snrType="wav",snrAdj=1, handle_method="noise_append",allOut=False):
    #将两段声音匹配剪切
    #mini_cut 指长度按照最小的那个剪
    #noise_append 指长度按照x1剪切，如果x2长度不够，将x2重复
    #allOut是指输出参数数量设置，为了适应不同的版本才这么做的
    
    x1 = x1.astype(np.int32)
    x2 = x2.astype(np.int32)
    l1 = x1.shape[0]
    l2 = x2.shape[0]

    if l1 != l2:
        if handle_method == 'mini_cut':
            ll = min(l1, l2)
            x1 = x1[:ll]
            x2 = x2[:ll]
        elif handle_method == 'noise_append':
            ll = max(l1, l2)
            if l1 < ll: #如果语音太短，就把噪声剪短到与语音相同长度
                x2 = x2[:l1]
            if l2 < ll: #如果语音较长，就把噪声重复叠加到与语音相同长度
                for i in range(int(ll/l2)):
                    x2 = np.append(x2, x2[:ll-x2.shape[0]])


    from numpy.linalg import norm
    if snrType=="wav":
        x2 = x2 / norm(x2) * norm(x1) / (10.0 ** (0.05 * snr)) 
    elif snrType=="spec":
        x2 = x2  / (10.0 ** (0.05 * snr))*snrAdj                               #snrAdj是外部得到的声谱图强度比例
    x2=x2.astype(np.int32)
    mix=(x1 + x2)
    mix=mix / norm(mix) *norm(x1)
    mix=mix.astype(np.int16)

    #print(SNR(x1,x2 ))
    
    if allOut==False:
        return mix
    else:
        x1 = x1.astype(np.int16)
        x2 = x2.astype(np.int16)
        return mix,x1,x2

def wavSizeCut(x1, length):
    # x1 是待裁剪或重复补充的wav
    #length 是裁剪到目标的设定长度
    #返回裁剪的结果
    x1 = x1.astype(np.int16)
    l1 = x1.shape[0]
    
    if length <= l1: #如果语音太短，就把噪声剪短到与语音相同长度
        x1 = x1[:length]
    if l1 < length: #如果语音较长，就把噪声重复叠加到与语音相同长度
        for i in range(int(length/l1)):
            x1 = np.append(x1, x1[:length-x1.shape[0]])

    
    return x1

def generateNoisyList(speechDir,noiseDir,listName="noisyMatchList.csv",matchN=100,listAddress="",matchType="random"):
    #生成语音和噪声构成带噪语音匹配列表
    #matchType="all", "random"
    #matchN 在 random 中是表示随机匹配出多少个， 在all 中表示语音和噪声中各自最多取出多少个来匹配（小于其文件总数下）
    import os #检索文件目录等
    import numpy as np
    import csv
    headers=["speech","noise"]
    value=[]
    
    PATH_ROOT = os.getcwd()
    os.chdir(PATH_ROOT) #根目录确定
     
    speechFileNames = os.listdir(speechDir)
    noiseFileNames = os.listdir(noiseDir) 
    
    l1=len(speechFileNames)
    l2=len(noiseFileNames)
    
    if matchType=="random":
        for i in range(matchN):
            value.append([speechFileNames[int(np.random.rand()*l1)],noiseFileNames[int(np.random.rand()*l2)]])
    elif matchType=="all":
        match1=min(matchN,l1)
        match2=min(matchN,l2)
        
        for i in range(match1):
            for j in range(match2):
                value.append([speechFileNames[i],noiseFileNames[j]])
        
        
    with open(listAddress+listName,"w",encoding="utf-8",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(value)

    
def readNoisyList(noisyListPath):
    #读取带噪语音匹配列表
    import csv
    import os #检索文件目录等
    
    PATH_ROOT = os.getcwd()
    os.chdir(PATH_ROOT) #根目录确定
    
    speechList=[]
    noiseList=[]
    with open(noisyListPath,"r") as rf:
        reader = csv.reader(rf)
        next(reader)
        
        for read in reader:
            speechList.append(read[0])
            noiseList.append(read[1])
    
    return [speechList,noiseList]
        


def connectWav(wavListPath):
    #将多个wav文件连接为一个大的wav文件,但是因为不检查采样率是否一致，所以列表最好使用统一的采样率
    #wavListPath 是装载了要被连接成为较大wav的wav的路径列表
    import scipy.io.wavfile as wav
    
    l=len(wavListPath)
    largeWav=[]
    
    
    for i in range(l):
        sr,wavx=wav.read(wavListPath[i])
        largeWav.extend(wavx)
    largeWav=np.array(largeWav)
    
    
    return sr,largeWav






















