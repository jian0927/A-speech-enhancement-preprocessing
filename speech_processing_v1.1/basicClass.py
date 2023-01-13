import os
import numpy as np

class test_train_data(object):
    #测试或训练的数据类
    magnitude_clean=[]
    magnitude_noise=[]
    magnitude_noisy=[]
    magnitude_est=[]
    phase_clean=[]
    phase_noise=[]
    phase_noisy=[]
    phase_est=[]
    wav_clean=[]
    wav_noise=[]
    wav_noisy=[]
    wav_est=[]
    sr=16000
    
    PATH_ROOT = os.getcwd()
    os.chdir(PATH_ROOT) #根目录确定
    
class test_train_dir(object):
    #用于测试或训练数据及处理后的输出地址，路径
    datasets_dir=r"datasets_speech" #****默认数据集所在的地址，下级文件夹是speech和noise
    output_dir=r"output"  #*****************默认输出的地址
    
    write_clean_path=output_dir+r"\clean.wav"
    write_noise_path=output_dir+r"\noise.wav"
    write_noisy_path=output_dir+r"\noisy.wav"
    write_est_path=output_dir+r"\est.wav"
    
    def check_files_dir(self):
        #检查文件是否存在
        if not os.path.exists(self.datasets_dir):
            os.makedirs(self.datasets_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        return 0
    
    def renew_output_path(self):
        #依据变动后的 output_dir 更新输出路径
        self.write_clean_path=self.output_dir+r"\clean.wav"
        self.write_noise_path=self.output_dir+r"\noise.wav"
        self.write_est_path=self.output_dir+r"\est.wav"
        return 0
        

class test_train_args(object):
    #stft相关参数
    window="hamming"
    winWide=512
    fftSize=1024 #如果想要stft分辨率更高一点，可以调大一点，但是需要防着畸变
    hopSize=256
    
    #dnn相关参数
    inputDim=int(fftSize/2+1)
    outputDim=int(fftSize/2+1)
    batchN=512
    epochN=30 #nn的迭代次数
    hiddenUnitN=1024
    dropOut=0.3
    
    #SNMF 相关参数
    costfcn = 'ls'
    W = np.ones([1,1])#  mgetopt(opts, 'W', rand(size(V,1),d));
    H = np.ones([1,1]) # mgetopt(opts, 'H', rand(d,size(V,2)));
    updateW = 'on'
    nuH = 1 #数值为 1 表示在进行SNMF时随机创建
    nuW = 1
    lambda1 = 0.2
    maxiter = 400 #最大迭代次数（到目标精度前）
    conv_criteria = 1e-4
    accel = 1
    plotfcn = []
    plotiter = 1
    displaylevel = 'iter'
    updateWRows = []
    displayIterN = 20
    d=800 #参数维度
    
    def renew_all_args(self):
        #依赖于其它参数的参数获得调整
        self.inputDim=int(self.fftSize/2+1)
        self.outputDim=int(self.fftSize/2+1)
