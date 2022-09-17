#从wav片段中提取特征参数



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
        s=[]
        for i in range(nf):
            x=frames[i:i+1]
            y=win*x[0]
            a=np.fft.fft(y)
            a = abs(a)
            s.append(a)
        if len(fec) == 0 :
            fec = s
        else: fec = np.vstack((fec,s))
    return  fec



if __name__ == '__main__':


    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa.display

    hanning = np.hanning(20)
    print(hanning)
    hanning = (np.ones((20, 20)) * hanning).T
    print(hanning)
