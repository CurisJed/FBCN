#从wav片段中提取特征参数
import numpy as np

from scipy.fftpack import dct
dess = r'E:\black_check\data\wav_point\keliban.txt'


def abs_fec_mfcc(des) :
    keliban = np.loadtxt(des, delimiter=',')
    framerate = 48000

    fec = []
    count = 0
    for i in keliban:
        count += 1
        print("加入特征",count,"/n")
        signal= i
        signal_len=len(signal)
    #预加重
        signal_add=np.append(signal[0],signal[1:]-0.97*signal[:-1])   #预加重

    #plt.figure(figsize=(20,10))
    #plt.subplot(2,1,1)
    #plt.plot(time,signal)
    #plt.subplot(2,1,2)
    #plt.plot(time,signal_add)
    #plt.show()
    #分帧
        wlen=512
        inc=256
        N=512
        if signal_len<wlen:
            nf=1
        else:
            nf = int(np.ceil((1.0 * signal_len - wlen + inc) / inc))
        pad_len=int((nf-1)*inc+wlen)
        zeros=np.zeros(pad_len-signal_len)
        pad_signal=np.concatenate((signal,zeros))
        indices=np.tile(np.arange(0,wlen),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(wlen,1)).T
        indices=np.array(indices,dtype=np.int32)
        frames=pad_signal[indices]
        win=np.hanning(wlen)
        m=24
        s=np.zeros((nf,m))
        for i in range(nf):
            x=frames[i:i+1]
            y=win*x[0]
            a=np.fft.fft(y)
            b=np.square(abs(a))
            mel_high=1125*np.log(1+(framerate/2)/700)
            mel_point=np.linspace(0,mel_high,m+2)
            Fp=700 * (np.exp(mel_point / 1125) - 1)
            w=int(N/2+1)
            df=framerate/N
            fr=[]
            for n in range(w):
                frs=int(n*df)
                fr.append(frs)
            melbank=np.zeros((m,w))
            for k in range(m+1):
                f1=Fp[k-1]
                f2=Fp[k+1]
                f0=Fp[k]
                n1=np.floor(f1/df)
                n2=np.floor(f2/df)
                n0=np.floor(f0/df)
                for j in range(w):
                    if j>= n1 and j<= n0:
                        melbank[k-1,j]=(j-n1)/(n0-n1)
                    if j>= n0 and j<= n2:
                        melbank[k-1,j]=(n2-j)/(n2-n0)
                for c in range(w):
                    s[i,k-1]=s[i,k-1]+b[c:c+1]*melbank[k-1,c]

        logs=np.log(s)
        num_ceps=12
        D = dct(logs,type = 2,axis = 0,norm = 'ortho')[:,1 : (num_ceps + 1)]
        D = D [12: ]
        if len(fec) == 0 :
            fec = D
        else: fec = np.vstack((fec,D))
    return  fec

#输入模型进行预测
fec_klb = abs_fec_mfcc(dess)
fec_mdb = abs_fec_mfcc(r'E:\black_check\data\wav_point\miduban.txt')
print("特征生成完毕")
np.savetxt('mdb_mfcc.txt', fec_mdb, fmt='%f', delimiter=',')
np.savetxt("klb_mfcc.txt", fec_klb, fmt='%f', delimiter=',')
print("特征保存完毕")


