import random
import sys

import warnings
from datetime import datetime

import torch

warnings.filterwarnings("ignore")
dic = {'duocenban': 0, 'keliban': 1, 'miduban': 2, 'shanmu': 3, 'songmu': 4, 'yumu': 5, 'daxinban': 6}


import librosa
import numpy as np
from torch.utils.data import Dataset

def combine_mask(feature_com,num):
    feature =[]
    featrue = []
    for i in range(num):
        if i == 0:
            feature_com[i] = feature_com[i][int(480/3):,:]
            feature = feature_com[i]
        if i == 3:
            feature_com[i] = feature_com[i][:int(60/3*2),:]
            feature = np.stack((feature,feature_com[i]))
        else:
            feature_com[i] = feature_com[i][int(480*0.5^(-i)/3):int(480*0.5^(-i)/3*2),:]
            feature = np.stack((feature, feature_com[i]))


    return featrue

# 加载并预处理音频
def load_audio(audio_path, feature_method='spectrogram', mode='train', sr=16000, chunk_duration=3, augmentors=None):
    # 读取音频数据
    wav, sr_ret = librosa.load(audio_path, sr=sr)

    # 获取音频特征
    if feature_method == 'melspectrogram':
        # 计算梅尔频谱
        features = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=400, n_mels=80, hop_length=160, win_length=400)
        features = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)
    elif feature_method == 'spectrogram':
        # 计算声谱图
        feature_stft = []   #宽带与窄带声谱图
        for i in range(4):
            linear = librosa.stft(wav, n_fft=960*0.5^(-i), win_length=960*0.5^(-i), hop_length=480)
            features, _ = librosa.magphase(linear)
            features = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)
            feature_stft.append(features)
        features = combine_mask(feature_stft,4)   #掩码滤波生成组合声谱图
        fec_c = np.array([])
        for i in range(47):
            if i == 0:
                fec_c = np.array(features[10 * i:10 * i + 20, :-1])[np.newaxis, :]
                continue
            fec_c = np.vstack((fec_c, np.array(features[10 * i:10 * i + 20, :-1])[np.newaxis, :]))
        features = np.array(fec_c)
        hanning = np.hanning(20)
        hanning = (np.ones((20, 20)) * hanning).T
        hanning = np.expand_dims(hanning,0).repeat(47,0)
        features = features*hanning

    else:
        raise Exception(f'预处理方法 {feature_method} 不存在！')
    # 归一化
    mean = np.mean(features, 0, keepdims=True)
    std = np.std(features, 0, keepdims=True)
    features = (features - mean) / (std + 1e-5)
    features = np.array(features,dtype="float32")
    return features


# 数据加载器
class CustomDataset(Dataset):
    def __init__(self, data_list_path, feature_method='melspectrogram', mode='train', sr=16000, chunk_duration=3, augmentors=None):
        super(CustomDataset, self).__init__()
        # 当预测时不需要获取数据
        if data_list_path is not None:
            with open(data_list_path, 'r') as f:
                self.lines = f.readlines()
        self.feature_method = feature_method
        self.mode = mode
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.augmentors = augmentors

    def __getitem__(self, idx):
        try:
            audio_path= self.lines[idx].replace('\n', '')
            label = audio_path .split('\\')[-2]
            label = dic[label]
            # 加载并预处理音频
            features = load_audio(audio_path, feature_method=self.feature_method, mode=self.mode, sr=self.sr,
                                  chunk_duration=self.chunk_duration, augmentors=self.augmentors)
            return features, np.array(int(label), dtype=np.int64)
        except Exception as ex:
            print(f"[{datetime.now()}] 数据: {self.lines[idx]} 出错，错误信息: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.lines)

    @property
    def input_size(self):
        if self.feature_method == 'melspectrogram':
            return 80
        elif self.feature_method == 'spectrogram':
            return 201
        else:
            raise Exception(f'预处理方法 {self.feature_method} 不存在！')


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch = sorted(batch, key=lambda sample: sample[0].shape[1], reverse=True)
    freq_size = batch[0][0].shape[0]
    max_audio_length = batch[0][0].shape[1]
    batch_size = len(batch)
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, freq_size, max_audio_length), dtype='float32')
    labels = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        labels.append(sample[1])
        seq_length = tensor.shape[1]
        # 将数据插入都0张量中，实现了padding
        inputs[x, :, :seq_length] = tensor[:, :]
    labels = np.array(labels, dtype='int64')
    # 打乱数据
    return torch.tensor(inputs), torch.tensor(labels)
