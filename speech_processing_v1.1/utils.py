# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:43:28 2018
@author: eesungkim
"""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io.wavfile as wav

def cost_snmf(V, W, H, beta=2,mu=0.1):
    A=tf.matmul(W,H)
    tmp=W*tf.matmul(A**(beta-1), H.T)
    numerator  = tf.matmul(A**(beta-2)*V,H.T) + W*(tf.matmul(tf.ones((tf.shape(tmp)[0],tf.shape(tmp)[0])),tmp )) 
    tmp2=W*tf.matmul(A**(beta-2)*V,H.T)
    denumerator= tf.matmul(A**(beta-1),H.T) + W*(tf.matmul(tf.ones((tf.shape(tmp2)[0],tf.shape(tmp2)[0])),tmp2))
    W_new = numerator/denumerator
    
    H_new= tf.matmul(W.T,V*A**(beta-2))/(tf.matmul(W.T,A**(beta-1))+mu)
    
    return W_new, H_new

def optimize(mode, V, W, H, beta, mu, lr, const_W):
    #cost function
    cost = tf.reduce_mean(tf.square(V - tf.matmul(W, H)))

    #update operation for H
    if mode=='snmf':
        #Sparse NMF MuR
        Wt=tf.transpose(W)
        Ht=tf.transpose(H)
        A=tf.matmul(W,H)
        H_new= tf.matmul(tf.transpose(W),V*A**(beta-2))/(tf.matmul(tf.transpose(W),A**(beta-1))+mu) #1
        #H_new=H*tf.matmul(tf.transpose(W),V)/(tf.matmul(tf.transpose(W),A)+mu) #2
        """  
        oneM=tf.ones_like(V) #3
        oneV=tf.constant(np.ones(V.shape[0]), tf.float32)
        H_new=H*(tf.matmul(Wt,V/A))/(tf.matmul(Wt,oneM)+mu)
        """
        H_update = H.assign(H_new)
    elif mode=='nmf':
        #Basic NMF MuR
        Wt = tf.transpose(W)
        H_new = H * tf.matmul(Wt, V) / tf.matmul(tf.matmul(Wt, W), H)
        H_update = H.assign(H_new)
    elif mode=='pg':
        """optimization; Projected Gradient method """
        dW, dH = tf.gradients(xs=[W, H], ys=cost)
        H_update_ = H.assign(H - lr * dH)
        H_update = tf.where(tf.less(H_update_, 0), tf.zeros_like(H_update_), H_update_)

    #update operation for W 
    if const_W == False:
        if mode=='snmf':
            #Sparse NMF MuR

            #vec = tf.reduce_sum(W,0)
            #multiply = tf.constant([tf.shape(W)[0]])

            #de=tf.reshape(tf.tile(tf.reduce_sum(W,0), multiply), [ multiply[0], tf.shape(tf.reduce_sum(W,0))[0]])
            #W=W/de
            #Ht=tf.transpose(H)
            #tmp=W*tf.matmul(A**(beta-1), Ht)
            #n=tf.shape(tmp)[0]
            #numerator  = tf.matmul(A**(beta-2)*V,Ht) + W*(tf.matmul(tf.ones((n,n)),tmp )) 
            #tmp2=W*tf.matmul(A**(beta-2)*V,Ht)
            #denumerator= tf.matmul(A**(beta-1),Ht) + W*(tf.matmul(tf.ones((n,n)),tmp2))
            #W_new = W*numerator/denumerator
            #W_update = W.assign(W_new)
            ###############################################################################
            
            """ """
            tmp=W*tf.matmul(A**(beta-1), Ht) #1
            n=tf.shape(tmp)[0]
            numerator  = tf.matmul(A**(beta-2)*V,Ht) 
            tmp2=W*tf.matmul(A**(beta-2)*V,Ht)
            denumerator= tf.matmul(A**(beta-1),Ht)
            W_new = W*numerator/denumerator
            W_update = W.assign(W_new)
            
            """  
            W_new=W*(tf.matmul(V,tf.transpose(H)))/(tf.matmul(A,tf.transpose(H))-W)#2
            W_new2=W_new/(tf.reduce_sum(W_new))
            W_update=W.assign(W_new2)
            """
            
            """  
            theSum1=tf.reduce_sum(W*tf.matmul(oneM,Ht),0) #3
            numerator  = tf.matmul(V/A,Ht)+W*(tf.multiply(oneV,theSum1))
            theSum2=tf.reduce_sum(W*tf.matmul(V/A,Ht),0)
            denumerator= tf.matmul(oneM,Ht)+W*(tf.multiply(oneV,theSum2))
            W_new = W*numerator/denumerator
            W_update = W.assign(W_new)
            """
            
        elif mode=='nmf':
            #Basic NMF MuR
            Ht = tf.transpose(H)
            W_new = W * tf.matmul(V, Ht)/ tf.matmul(W, tf.matmul(H, Ht))
            W_update = W.assign(W_new)
        elif mode=='pg':
            W_update_ = W.assign(W - lr * dW)
            W_update = tf.where(tf.less(W_update_, 0), tf.zeros_like(W_update_), W_update_)



        return W_update,H_update, cost

    return 0, H_update, cost







# code from https://github.com/eesungkim/NMF-Tensorflow
def NMF_MuR(V_input,r,max_iter,display_step,const_W,init_W,mode='nmf'):
    m,n=np.shape(V_input)
    
    #tf.reset_default_graph()
    tf.compat.v1.reset_default_graph() #这个函数名更新了
    
    
    #V = tf.placeholder(tf.float32)  #也更新了
    tf.compat.v1.disable_eager_execution() #需要添加这句来保证下一句的正常运行
    V = tf.compat.v1.placeholder(tf.float32) 
    
    initializer = tf.random_uniform_initializer(0,1)


    if const_W==False:
        W =  tf.compat.v1.get_variable(name="W", shape=[m, r], initializer=initializer) #W 是字典
        H =  tf.compat.v1.get_variable("H", [r, n], initializer=initializer)   #H 是权重矩阵
    else:
        W =  tf.compat.v1.constant(init_W, shape=[m, r], name="W")
        H =  tf.compat.v1.get_variable("H", [r, n], initializer=initializer)
    
    W_update, H_update, cost=optimize(mode, V, W, H, beta=2, mu=0.0001, lr=0.1, const_W=const_W)


    with tf.compat.v1.Session() as sess:
        
        sess.run(tf.compat.v1.global_variables_initializer())
        for idx in range(max_iter):
            if const_W == False:
                W=sess.run(W_update, feed_dict={V:V_input})
                H=sess.run(H_update, feed_dict={V:V_input})
            else:
                H=sess.run(H_update, feed_dict={V:V_input})
                
            if (idx % display_step) == 0:
                costValue = sess.run(cost,feed_dict={V:V_input})
                print("|Epoch:","{:4d}".format(idx), " Cost=","{:.3f}".format(costValue/1))

    print("================= [Completed Training NMF] ===================")
    return W, H


#我自己写的，用于可以让字典矩阵重复
def NMF_MuR2(V_input,r,max_iter,display_step,const_W,init_W,continue_W=False,mode='nmf'):
    m,n=np.shape(V_input)
    
    #tf.reset_default_graph()
    tf.compat.v1.reset_default_graph() #这个函数名更新了
    
    
    #V = tf.placeholder(tf.float32)  #也更新了
    tf.compat.v1.disable_eager_execution() #需要添加这句来保证下一句的正常运行
    V = tf.compat.v1.placeholder(tf.float32) 
    
    initializer = tf.random_uniform_initializer(0,1)


    if const_W==False:
        if continue_W==False:
            W =  tf.compat.v1.get_variable(name="W", shape=[m, r], initializer=initializer) #W 是字典
        else:
            W = tf.compat.v1.Variable(init_W)
            
            
        H =  tf.compat.v1.get_variable("H", [r, n], initializer=initializer)   #H 是权重矩阵
    else:
        W =  tf.compat.v1.constant(init_W, shape=[m, r], name="W")
        H =  tf.compat.v1.get_variable("H", [r, n], initializer=initializer)
    
    W_update, H_update, cost=optimize(mode, V, W, H, beta=2, mu=0.00001, lr=0.1, const_W=const_W)
    

    with tf.compat.v1.Session() as sess:
        
        sess.run(tf.compat.v1.global_variables_initializer())
        for idx in range(max_iter):
            if const_W == False:
                W=sess.run(W_update, feed_dict={V:V_input})
                H=sess.run(H_update, feed_dict={V:V_input})
            else:
                H=sess.run(H_update, feed_dict={V:V_input})
                
            if (idx % display_step) == 0:
                costValue = sess.run(cost,feed_dict={V:V_input})
                print("|Epoch:","{:4d}".format(idx), " Cost=","{:.3f}".format(costValue/1))

    print("================= [Completed Training NMF] ===================")
    return W, H


def NMF_MuR_H(V_input,r,max_iter,display_step,const_H,init_H):
    m,n=np.shape(V_input)
    
    tf.reset_default_graph()
    
    V = tf.placeholder(tf.float32) 
    
    initializer = tf.random_uniform_initializer(0,1)

    if const_H==False:
        W =  tf.get_variable(name="W", shape=[m, r], initializer=initializer)
        H =  tf.get_variable("H", [r, n], initializer=initializer)
    else:
        W =  tf.get_variable(name="W", shape=[m, r], initializer=initializer)
        H =  tf.constant(init_H, shape=[r, n], name="H")
        
    WH =tf.matmul(W, H)

    #cost function
    cost = tf.reduce_mean(tf.square(V - WH))

    #optimization; Multiplicative update Rule (MuR)
    #update operation for H
    if const_H == False:
        Wt = tf.transpose(W)
        H_new = H * tf.matmul(Wt, V) / tf.matmul(tf.matmul(Wt, W), H)
        H_update = H.assign(H_new)
    
    #update operation for H 
    Ht = tf.transpose(H)
    W_new = W * tf.matmul(V, Ht)/ tf.matmul(W, tf.matmul(H, Ht))
    W_update = W.assign(W_new)

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        for idx in range(max_iter):
            costValue = sess.run(cost,feed_dict={V:V_input})
            if const_H == False:
                W=sess.run(W_update, feed_dict={V:V_input})
                H=sess.run(H_update, feed_dict={V:V_input})
            else:
                W=sess.run(W_update, feed_dict={V:V_input})
                
            if (idx % display_step) == 0:
                
                print("|Epoch:","{:4d}".format(idx), " Cost=","{:.3f}".format(costValue/1000000))
    print("================= [Completed Training NMF] ===================")
    return W, H


    
def divide_magphase(D, power=1):
    """Separate a complex-valued stft D into its magnitude (S)
    and phase (P) components, so that `D = S * P`."""

    mag = np.abs(D)**power
    phase = np.exp(1.j * np.angle(D))

    return mag, phase

def merge_magphase(magnitude, phase):
    """merge magnitude (S) and phase (P) to a complex-valued stft D so that `D = S * P`."""
    # magnitude * np.exp(np.angle(D)*1j)
    # magnitude =  np.exp(magnitude)
    # magnitude = np.sqrt(magnitude)
    # magnitude * numpy.cos(np.angle(D))+ magnitude * numpy.sin(np.angle(D))*1j
    return magnitude * phase


def nfft(frame_length):
    fft_size = 2
    while fft_size < frame_length:
        fft_size = fft_size * 2
    return fft_size


def wav_stft(wav1,args,toRI=False):
    #toRI=False 将波形数据的声音转成幅度和相位的声谱图
    #toRI=True 将波形数据的声音转成实部和虚部的声谱图
    time_signal = wav1
    time_signal=time_signal.astype('float')
    fft_size = nfft(args.fftSize)
    freq_signal = librosa.stft(time_signal, n_fft=fft_size, win_length=args.winWide, hop_length=args.hopSize, window=args.window)
    if toRI == False:
        magnitude, phase = divide_magphase(freq_signal, power=1)
        return magnitude, phase
    else:
        realPart=freq_signal.real  #这里我简单用一下对实部和虚部建模，分开训练，然后再合在一起，测一下效果
        imaginaryPart=freq_signal.imag
        return realPart, imaginaryPart


from math import acos,asin
def degPhase(expPhase):
    #将角度格式从高斯转到弧度制
    
    if isinstance(expPhase, (int,float)): #用于单个转换处理
        a=expPhase.real  #相位角的实部
        b=expPhase.imag  #相位角的虚部
    
        if a>0 and b>=0:
            c=acos(a)
        elif a<=0 and b>0:
            c=acos(a)
        elif a<0 and b<=0:
            c=-asin(b)+np.pi
        elif a>=0 and b<0:
            c=asin(b)+np.pi*2   
            
        return c
    elif isinstance(expPhase, (np.ndarray)):  #用于数组批处理
        theSize=expPhase.shape
        c=np.zeros([theSize[0],theSize[1]])
        for i in range(theSize[0]):
            for j in range(theSize[1]):
                a=expPhase[i,j].real  #相位角的实部
                b=expPhase[i,j].imag  #相位角的虚部
            
                if a>0 and b>=0:
                    c[i,j]=acos(a)
                elif a<=0 and b>0:
                    c[i,j]=acos(a)
                elif a<0 and b<=0:
                    c[i,j]=-asin(b)+np.pi
                elif a>=0 and b<0:
                    c[i,j]=asin(b)+np.pi*2   
        return c
    
    
    
def expPhase(degPhase):
    #将角度格式从弧度制转到高斯
    return np.exp(degPhase*1j)



def normalizeW(W):
    #对数据统合归一化（L2）
    eps = 1e-15
    Q = np.sqrt(sum(W**2,0));
    W = W/np.tile(Q+eps,[Q.shape[0],1]);
    return W




















