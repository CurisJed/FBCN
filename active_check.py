import os.path
import wave

import librosa
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import librosa.display


#plt.figure(dpi=600) # 将显示的所有图分辨率调高
#matplotlib.rc("font",family='SimHei') # 显示中文
#matplotlib.rcParams['axes.unicode_minus']=False # 显示符号




def displayWaveform(orign_wav_path): # 显示语音时域波形
    """
    display waveform of a given speech sample
    :param sample_name: speech sample name
    :param fs: sample frequency
    :return:
    """
    samples, sr = librosa.load(orign_wav_path,sr= None)

    samples = samples[sr*0:sr*10] # 切割文件前五秒

    print(len(samples), sr)
    time = np.arange(0, len(samples)) * (1.0 / sr)
    print(time,"秒")
    plt.plot(time, samples)
    plt.title("语音信号时域波形")
    plt.xlabel("时长（秒）")
    plt.ylabel("振幅")
    # plt.savefig("your dir\语音信号时域波形图", dpi=600)
    plt.show()

def displaySpectrum(orign_wav_path): # 显示语音频域谱线
    x, sr = librosa.load(orign_wav_path, sr=48000)
    print(len(x))
    # ft = librosa.stft(x)
    # magnitude = np.abs(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    # frequency = np.angle(ft)  # (0, 16000, 121632)

    ft = fft(x)
    print(len(ft), type(ft), np.max(ft), np.min(ft))
    magnitude = np.absolute(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    frequency = np.linspace(0, sr, len(magnitude))  # (0, 16000, 121632)

    print(len(magnitude), type(magnitude), np.max(magnitude), np.min(magnitude))
    print(len(frequency), type(frequency), np.max(frequency), np.min(frequency))

    # plot spectrum，限定[:40000]
    # plt.figure(figsize=(18, 8))
    plt.plot(frequency[:40000], magnitude[:40000])  # magnitude spectrum
    plt.title("语音信号频域谱线")
    plt.xlabel("频率（赫兹）")
    plt.ylabel("幅度")
    plt.savefig("your dir\语音信号频谱图", dpi=600)
    plt.show()

    # # plot spectrum，不限定 [对称]
    # plt.figure(figsize=(18, 8))
    # plt.plot(frequency, magnitude)  # magnitude spectrum
    # plt.title("语音信号频域谱线")
    # plt.xlabel("频率（赫兹）")
    # plt.ylabel("幅度")
    # plt.show()

from pydub import AudioSegment
def X_to_wav(f_path,file_class):
    if file_class == 'mp3':
        song = AudioSegment.from_mp3(f_path)
        song.export( f_path.split(".")[0]+ '.wav', format=str('wav'))
    if file_class == 'flac':
        song = AudioSegment.from_flac(f_path)
        song.export( f_path.split(".")[0]+ '.wav', format=str('wav'))
    if file_class == 'ogg':
        song = AudioSegment.from_ogg(f_path)
        song.export(f_path.split(".")[0] + '.wav', format=str('wav'))


def displaySpectrogram(orign_wav_path):
    x, sr = librosa.load(orign_wav_path, sr=48000)
    x = x[sr*5:sr*10]
    # compute power spectrogram with stft(short-time fourier transform):
    # 基于stft，计算power spectrogram
    spectrogram = librosa.amplitude_to_db(librosa.stft(x))

    # show
    librosa.display.specshow(spectrogram, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('语音信号对数谱图')
    plt.xlabel('时长（秒）')
    plt.ylabel('频率（赫兹）')
    plt.show()

import ffmpeg
def point_spilt(wav_path, out_path):

    if wav_path.split(".")[-1] != 'wav':    # 数据类型变换
        X_to_wav(wav_path,wav_path.split(".")[-1])
        wav_path = wav_path.split(".")[0]+'.wav'
    ffmpeg.input(wav_path).output(wav_path, ar=48000).run()

    f = wave.open(wav_path,"rb")

    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]  # 声道数、量化位数、采样频率、采样点数
    str_data = f.readframes(nframes)  # 读取音频，字符串格式
    f.close()
    wavedata = np.frombuffer(str_data, dtype=np.short)  # 将字符串转化为浮点型数据

    max_wave = (max(abs(wavedata)))
    points = 0
    i = 0.05*framerate
    while i < len(wavedata):
        if wavedata[int(i)] >0.15*max_wave:
            n_wave = wavedata[int(i-0.05*framerate):int(i+0.15*framerate)]
            f = wave.open( os.path.join(out_path, str(points)+".wav") , "wb")
            f.setnchannels(nchannels)
            f.setsampwidth(sampwidth)
            f.setframerate(framerate)
            f.writeframes(n_wave.tostring())
            f.close()
            points += 1
            i += 0.15*framerate
        else: i += 1

# 生成数据列表
def get_data_list(audio_path, list_path,testnum):
    sound_sum = 0
    audios = os.listdir(audio_path)
    for kind in audios:

        f_train = open(os.path.join(list_path, testnum+kind+ ".txt"), 'w')

        sounds = os.listdir(os.path.join(audio_path, kind))
        for sound in sounds:
            if '.wav' not in sound:continue
            sound_path = os.path.join(audio_path, kind, sound)
            f_train.write('%s\n' % (sound_path))
            sound_sum += 1

        f_train.close()

if __name__ == '__main__':
    point_spilt(r"E:\black_check\data\orign_knock_wav\榆木长測1.wav",r"E:\black_check\data\wav_test\test1\yumu")
    #displayWaveform()
    #displaySpectrum()
    #displaySpectrogram()
    #b = np.loadtxt('keliban.txt', delimiter=',')
    #get_data_list(r'E:\black_check\data\wav_test\test1',r'E:\black_check\data\class_list','test1')
    '''
    audio_path = r"E:\black_check\data\wav_add\klbedge"
    audios = os.listdir(audio_path)

    f_test = open(os.path.join(r'E:\black_check\tdnn\dataset', 'test_list.txt'), 'w')

    for i in range(len(audios)):
        sound_path = os.path.join(audio_path, audios[i])

        f_test.write('%s\t%d\n' % (sound_path, 2))
    f_test.close()
    '''
