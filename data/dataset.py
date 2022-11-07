import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
import nnmnkwii.datasets.jvs
from nnmnkwii.io import hts

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
    def __init__(self):
        super(TransChunked, self).__init__()
        self.chunk = 32
        self.width = 16
    
    def __call__(self, x):
        num = x.shape[1]//self.width - 1
        l = []
        for i in range(num):
            start = i*self.width
            end = start+self.chunk
            l.append(x[:, start:end])
            
        l.append(x[:, x.shape[1]-32:x.shape[1]])
        return l
    

class JVS_Dataset(Dataset):
    def __init__(self, root, speakers, data_type='wave'):
        super().__init__()
        
        self.data_type = data_type
        self.speakers = speakers # list(int)
        self.data = list(self.extract_data(root))
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
        
    def extract_data(self, root):
        for speaker in self.speakers:
            transcriptions = nnmnkwii.datasets.jvs.TranscriptionDataSource(root, categories=["parallel"], speakers=[speaker])
            wav_paths = nnmnkwii.datasets.jvs.WavFileDataSource(root, categories=["parallel"], speakers=[speaker])
            wave = self.extract_wave(wav_paths, transcriptions)
            transforms = torch.nn.Sequential(
                    torchaudio.transforms.Spectrogram(
                        n_fft=1024,
                        win_length=1024,
                        hop_length=256
                        ),
                    torchaudio.transforms.MelScale(sample_rate=24000, n_mels=32, n_stft=513),
                    TransSqueeze(),   
                )
            
            if self.data_type == 'wave':
                yield speaker, wave
            
            elif self.data_type == 'mel':
                
                spec = transforms(wave)
                spec_max = spec.max()
                spec_min = spec.min()
                
                chunked = TransChunked()
                
                for chunk in chunked(spec):
                    chunk = chunk.unsqueeze(0)
                    
                    speaker_index = self.speakers.index(speaker)
                    speaker_one_hot = torch.nn.functional.one_hot(torch.tensor(speaker_index), num_classes=len(self.speakers))
                    
                    yield chunk, speaker_one_hot, spec_max, spec_min
    
    def extract_wave(self, wav_paths, transcriptions):
        xx = torch.tensor([[]])
        for idx, (text, wav_path) in enumerate(zip(transcriptions.collect_files(), wav_paths.collect_files())):
            x, sr = torchaudio.load(wav_path.replace("wav24kHz16bit/", "trimed_pau/"))
            xx = torch.cat((xx, x), dim=1)        
        return xx