import torch
import torchaudio
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
import nnmnkwii.datasets.jvs
from nnmnkwii.io import hts
import pyworld as pw
import pysptk as ps
import os
import numpy as np


np.random.seed(0)


class TransSqueeze(nn.Module):
    def __init__(self, dim=0):
        super(TransSqueeze, self).__init__()
        
        self.dim = dim
    
    def __call__(self, x):
        return x.squeeze(self.dim)


class TransUnSqueeze(nn.Module):
    def __init__(self, dim=0):
        super(TransUnSqueeze, self).__init__()
        
        self.dim = dim
    
    def __call__(self, x):
        return x.unsqueeze(self.dim)


class TransChunked(nn.Module):
    def __init__(self, chunk=32, width=16):
        super(TransChunked, self).__init__()
        self.chunk = chunk
        self.width = width
    
    def __call__(self, x):
        num = x.shape[1]//self.width - 1
        l = []
        for i in range(num):
            start = i*self.width
            end = start+self.chunk
            l.append(x[:, start:end])
            
        l.append(x[:, (x.shape[1]-self.chunk):x.shape[1]])
        return l
    

class JVSDataset(Dataset):
    def __init__(self, root, speakers, data_type='wave', n_mels=32, chunk=32, width=16,):
        super().__init__()
        
        self.data_type = data_type
        self.speakers = speakers # list(int)
        self.fs = 24000
        self.n_mels = n_mels
        self.chunk = chunk
        self.width = width
        
        self.data = list(self.extract_data(root))
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
        
    def extract_data(self, root):
        for speaker in self.speakers:
            transcriptions = nnmnkwii.datasets.jvs.TranscriptionDataSource(root, categories=["parallel"], speakers=[speaker], max_files=100)
            wav_paths = nnmnkwii.datasets.jvs.WavFileDataSource(root, categories=["parallel"], speakers=[speaker], max_files=100)
            wave = self.extract_wave(wav_paths, transcriptions)
            transforms = torch.nn.Sequential(
                    torchaudio.transforms.Spectrogram(
                        n_fft=1024,
                        win_length=1024,
                        hop_length=256
                        ),
                    torchaudio.transforms.MelScale(sample_rate=24000, n_mels=self.n_mels, n_stft=513),
                    TransSqueeze(),   
                )
            
            if self.data_type == 'wave':
                yield speaker, wave
            
            elif self.data_type == 'mel':
                
                spec = transforms(wave)
                spec_max = spec.max()
                spec_min = spec.min()
                
                chunked = TransChunked(self.chunk, self.width)
                for chunk in chunked(spec):
                    chunk = chunk.unsqueeze(0)
                    
                    speaker_index = self.speakers.index(speaker)
                    speaker_one_hot = torch.nn.functional.one_hot(torch.tensor(speaker_index), num_classes=len(self.speakers))
                    
                    yield chunk, speaker_one_hot, spec_max, spec_min
            
            elif self.data_type == 'mc':
                f0, sp, ap = pw.wav2world(wave.squeeze(0).to(torch.double).numpy(), self.fs)
                mc = ps.sp2mc(sp, order=self.n_mels-1, alpha=ps.util.mcepalpha(24000))
                mc = torch.from_numpy(mc.T)
                chunked = TransChunked(self.chunk, self.width)

                for chunk in chunked(mc):
                    chunk = chunk.unsqueeze(0).to(torch.float)
                    
                    speaker_index = self.speakers.index(speaker)
                    speaker_one_hot = torch.nn.functional.one_hot(torch.tensor(speaker_index), num_classes=len(self.speakers))
                    
                    yield chunk, speaker_one_hot
                    
            elif self.data_type == 'sp':
                f0, sp, ap = pw.wav2world(wave.squeeze(0).to(torch.double).numpy(), self.fs)

                    
                speaker_index = self.speakers.index(speaker)
                speaker_one_hot = torch.nn.functional.one_hot(torch.tensor(speaker_index), num_classes=len(self.speakers))
                    
                yield f0, sp, ap, speaker_one_hot
    
    def extract_wave(self, wav_paths, transcriptions):
        xx = torch.tensor([[]])
        for idx, (text, wav_path) in enumerate(zip(transcriptions.collect_files(), wav_paths.collect_files())):
            x, sr = torchaudio.load(wav_path.replace("wav24kHz16bit/", "trimed_pau/"))
            xx = torch.cat((xx, x), dim=1)        
        return xx
    

    
class McJVSDataset(Dataset):
    def __init__(self, root="36_40_melceps"):
        super().__init__()
        
        self.fs = 24000
        self.data = list(self.extract_data(root))
        
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def extract_data(self, root):
        for i in os.listdir(root):
            yield torch.load(os.path.join(root, i))


class ImageDataset(Dataset):
    def __init__(self, root="images"):
        super().__init__()
        self.root = root
        self.data = self._extract_data()
        
    def __getitem__(self, index):
        return self._get_random(index, 1)
    
    def _get_random(self, index, label):
 
        return np.random.choice(self.data[label][index])
        
    def __len__(self):
        return len(self.data)
    
    def _extract_data(self):
        data = [[[] for i in range(51)] for j in range(3)]
        with open(self.root+"/"+"crossdata.txt") as f:
            for i in f:
                index, image, label = i.lstrip().split()
                image = torchvision.transforms.Resize((32,32))(torchvision.io.read_image(self.root+"/"+image))
                image = image.to(torch.float)
                data[int(label)][int(index)].append(image)
        return data


class CrossDataset(Dataset):
    def __init__(self, audio_data_dir="36_40_melceps", image_data_dir="images"):
        super().__init__()
        self.image = ImageDataset(root=image_data_dir)
        self.audio = McJVSDataset(root=audio_data_dir)
        
    def __getitem__(self, index):
        audio, label = self.audio[index]
        image = self.image[torch.argmax(label)]
        
        
        return image, audio, label
    
    def __len__(self):
        return self.audio.__len__()