import torch
import math
from torch import nn
import pytorch_lightning as pl



params = dict()

# UttrEnc setting
params['UTTR_ENC_NUM_LAYERS'] = 4

params['UTTR_ENC_CONV1_CHANNELS'] = 1
params['UTTR_ENC_CONV2_CHANNELS'] = 16
params['UTTR_ENC_CONV3_CHANNELS']= 32
params['UTTR_ENC_CONV4_CHANNELS'] = 32
params['UTTR_ENC_CONV5_CHANNELS'] = 16

params['UTTR_ENC_CONV1_KERNEL'] = (3, 9)
params['UTTR_ENC_CONV2_KERNEL'] = (4, 8)
params['UTTR_ENC_CONV3_KERNEL'] = (4, 8)
params['UTTR_ENC_CONV4_KERNEL'] = (9, 5)

params['UTTR_ENC_CONV1_STRIDE'] = (1, 1)
params['UTTR_ENC_CONV2_STRIDE'] = (2, 2)
params['UTTR_ENC_CONV3_STRIDE'] = (2, 2)
params['UTTR_ENC_CONV4_STRIDE'] = (9, 1)

for i in range(1, params['UTTR_ENC_NUM_LAYERS']+1):
    params[f'UTTR_ENC_CONV{i}_PADDING'] = tuple([math.floor((params[f'UTTR_ENC_CONV{i}_KERNEL'][j]-params[f'UTTR_ENC_CONV{i}_STRIDE'][j])/2) for j in range(2)])
    
# UttrDec setting
params['UTTR_DEC_NUM_LAYERS'] = 4

params['UTTR_DEC_CONV1_CHANNELS'] = 8
params['UTTR_DEC_CONV2_CHANNELS'] = 16
params['UTTR_DEC_CONV3_CHANNELS']= 16
params['UTTR_DEC_CONV4_CHANNELS'] = 8
params['UTTR_DEC_CONV5_CHANNELS'] = 2

params['UTTR_DEC_CONV1_KERNEL'] = (9, 5)
params['UTTR_DEC_CONV2_KERNEL'] = (4, 8)
params['UTTR_DEC_CONV3_KERNEL'] = (4, 8)
params['UTTR_DEC_CONV4_KERNEL'] = (3, 9)

params['UTTR_DEC_CONV1_STRIDE'] = (9, 1)
params['UTTR_DEC_CONV2_STRIDE'] = (2, 2)
params['UTTR_DEC_CONV3_STRIDE'] = (2, 2)
params['UTTR_DEC_CONV4_STRIDE'] = (1, 1)

for i in range(1, params['UTTR_DEC_NUM_LAYERS']+1):
    params[f'UTTR_DEC_CONV{i}_PADDING'] = tuple([math.ceil((params[f'UTTR_DEC_CONV{i}_KERNEL'][j]-params[f'UTTR_DEC_CONV{i}_STRIDE'][j])/2) for j in range(2)])
    params[f'UTTR_DEC_CONV{i}_OUT_PADDING'] = tuple([(params[f'UTTR_DEC_CONV{i}_KERNEL'][j]-params[f'UTTR_DEC_CONV{i}_STRIDE'][j])%2 for j in range(2)])
    
# FaceEnc setting
params['FACE_ENC_CONV_LAYERS'] = 5
params['FACE_ENC_LINEAR_LAYERS'] = 2

params['FACE_ENC_CONV1_CHANNELS'] = 3
params['FACE_ENC_CONV2_CHANNELS'] = 32
params['FACE_ENC_CONV3_CHANNELS']= 64
params['FACE_ENC_CONV4_CHANNELS'] = 128
params['FACE_ENC_CONV5_CHANNELS'] = 128
params['FACE_ENC_CONV6_CHANNELS'] = 256

params['FACE_ENC_LINEAR1_CHANNELS'] = 256
params['FACE_ENC_LINEAR2_CHANNELS'] = 16
params['FACE_ENC_LINEAR3_CHANNELS'] = 16

params['FACE_ENC_CONV1_KERNEL'] = (6, 6)
params['FACE_ENC_CONV2_KERNEL'] = (6, 6)
params['FACE_ENC_CONV3_KERNEL'] = (4, 4)
params['FACE_ENC_CONV4_KERNEL'] = (4, 4)
params['FACE_ENC_CONV5_KERNEL'] = (2, 2)

params['FACE_ENC_CONV1_STRIDE'] = (2, 2)
params['FACE_ENC_CONV2_STRIDE'] = (2, 2)
params['FACE_ENC_CONV3_STRIDE'] = (2, 2)
params['FACE_ENC_CONV4_STRIDE'] = (2, 2)
params['FACE_ENC_CONV5_STRIDE'] = (2, 2)



for i in range(1, params['FACE_ENC_CONV_LAYERS']+1):
    params[f'FACE_ENC_CONV{i}_PADDING'] = tuple([math.floor((params[f'FACE_ENC_CONV{i}_KERNEL'][j]-params[f'FACE_ENC_CONV{i}_STRIDE'][j])/2) for j in range(2)])
    
# FaceDec setting
params['FACE_DEC_LINEAR_LAYERS'] = 2
params['FACE_DEC_CONV_LAYERS'] = 5

params['FACE_DEC_LINEAR1_CHANNELS'] = 8
params['FACE_DEC_LINEAR2_CHANNELS'] = 128
params['FACE_DEC_LINEAR3_CHANNELS'] = 2048

params['FACE_DEC_CONV1_CHANNELS']= 128
params['FACE_DEC_CONV2_CHANNELS'] = 128
params['FACE_DEC_CONV3_CHANNELS'] = 64
params['FACE_DEC_CONV4_CHANNELS'] = 32
params['FACE_DEC_CONV5_CHANNELS'] = 6
params['FACE_DEC_CONV6_CHANNELS'] = 6

params['FACE_DEC_CONV1_KERNEL'] = (3, 3)
params['FACE_DEC_CONV2_KERNEL'] = (6, 6)
params['FACE_DEC_CONV3_KERNEL'] = (6, 6)
params['FACE_DEC_CONV4_KERNEL'] = (6, 6)
params['FACE_DEC_CONV5_KERNEL'] = (5, 5)

params['FACE_DEC_CONV1_STRIDE'] = (2, 2)
params['FACE_DEC_CONV2_STRIDE'] = (2, 2)
params['FACE_DEC_CONV3_STRIDE'] = (2, 2)
params['FACE_DEC_CONV4_STRIDE'] = (2, 2)
params['FACE_DEC_CONV5_STRIDE'] = (2, 2) # 元論文の実装と違う



for i in range(1, params['FACE_DEC_CONV_LAYERS']+1):
    params[f'FACE_DEC_CONV{i}_PADDING'] = tuple([math.ceil((params[f'FACE_DEC_CONV{i}_KERNEL'][j]-params[f'FACE_DEC_CONV{i}_STRIDE'][j])/2) for j in range(2)])
    params[f'FACE_DEC_CONV{i}_OUT_PADDING'] = tuple([(params[f'FACE_DEC_CONV{i}_KERNEL'][j]-params[f'FACE_DEC_CONV{i}_STRIDE'][j])%2 for j in range(2)])
    
# VoiceEnc setting
params['VOICE_ENC_NUM_LAYERS'] = 7

params['VOICE_ENC_CONV1_CHANNELS'] = 1
params['VOICE_ENC_CONV2_CHANNELS'] = 32
params['VOICE_ENC_CONV3_CHANNELS']= 64
params['VOICE_ENC_CONV4_CHANNELS'] = 128
params['VOICE_ENC_CONV5_CHANNELS'] = 128
params['VOICE_ENC_CONV6_CHANNELS'] = 128
params['VOICE_ENC_CONV7_CHANNELS'] = 64
params['VOICE_ENC_CONV8_CHANNELS'] = 16

params['VOICE_ENC_CONV1_KERNEL'] = (3, 9)
params['VOICE_ENC_CONV2_KERNEL'] = (4, 8)
params['VOICE_ENC_CONV3_KERNEL'] = (4, 8)
params['VOICE_ENC_CONV4_KERNEL'] = (4, 8)
params['VOICE_ENC_CONV5_KERNEL'] = (4, 5)
params['VOICE_ENC_CONV6_KERNEL'] = (1, 5)
params['VOICE_ENC_CONV7_KERNEL'] = (1, 5)

params['VOICE_ENC_CONV1_STRIDE'] = (1, 1)
params['VOICE_ENC_CONV2_STRIDE'] = (2, 2)
params['VOICE_ENC_CONV3_STRIDE'] = (2, 2)
params['VOICE_ENC_CONV4_STRIDE'] = (2, 2)
params['VOICE_ENC_CONV5_STRIDE'] = (4, 1)
params['VOICE_ENC_CONV6_STRIDE'] = (1, 1)
params['VOICE_ENC_CONV7_STRIDE'] = (1, 1)

for i in range(1, params['VOICE_ENC_NUM_LAYERS']+1):
    params[f'VOICE_ENC_CONV{i}_PADDING'] = tuple([math.floor((params[f'VOICE_ENC_CONV{i}_KERNEL'][j]-params[f'VOICE_ENC_CONV{i}_STRIDE'][j])/2) for j in range(2)])
    
params["LAMBDA1"] = 0.01
params["LAMBDA2"] = 1
params["LAMBDA3"] = 0.001
params["LAMBDA4"] = 0.01
params["LAMBDA5"] = 1
    
params["LR"] = 1e-5
    
    
class GLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2):
        return x1 * torch.sigmoid(x2)
    

class UttrEncoder(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.model = nn.ModuleDict()
        self.save_hyperparameters(params)
        
        self.model['lr'] = GLU()
        
        NUM_LAYERS = self.hparams['UTTR_ENC_NUM_LAYERS']
        for i in range(1, NUM_LAYERS):
            self.model[f'conv{i}a'] = nn.Conv2d(
                self.hparams[f'UTTR_ENC_CONV{i}_CHANNELS'],
                self.hparams[f'UTTR_ENC_CONV{i+1}_CHANNELS'],
                self.hparams[f'UTTR_ENC_CONV{i}_KERNEL'],
                self.hparams[f'UTTR_ENC_CONV{i}_STRIDE'],
                self.hparams[f'UTTR_ENC_CONV{i}_PADDING'],
                bias=False, padding_mode='replicate'
            )
            self.model[f'bn{i}a'] = nn.BatchNorm2d(self.hparams[f'UTTR_ENC_CONV{i+1}_CHANNELS'])
            
            self.model[f'conv{i}b'] = nn.Conv2d(
                self.hparams[f'UTTR_ENC_CONV{i}_CHANNELS'],
                self.hparams[f'UTTR_ENC_CONV{i+1}_CHANNELS'],
                self.hparams[f'UTTR_ENC_CONV{i}_KERNEL'],
                self.hparams[f'UTTR_ENC_CONV{i}_STRIDE'],
                self.hparams[f'UTTR_ENC_CONV{i}_PADDING'],
                bias=False, padding_mode='replicate'
            )
            self.model[f'bn{i}b'] = nn.BatchNorm2d(self.hparams[f'UTTR_ENC_CONV{i+1}_CHANNELS'])
            
        self.model[f'conv{NUM_LAYERS}'] =  nn.Conv2d(
            self.hparams[f'UTTR_ENC_CONV{NUM_LAYERS}_CHANNELS'],
            self.hparams[f'UTTR_ENC_CONV{NUM_LAYERS+1}_CHANNELS'],
            self.hparams[f'UTTR_ENC_CONV{NUM_LAYERS}_KERNEL'],
            self.hparams[f'UTTR_ENC_CONV{NUM_LAYERS}_STRIDE'],
            self.hparams[f'UTTR_ENC_CONV{NUM_LAYERS}_PADDING'],
            bias=False, padding_mode='replicate'
        )
    
    def forward(self, x):
        NUM_LAYERS = self.hparams['UTTR_ENC_NUM_LAYERS']
        for i in range(1, NUM_LAYERS):
            x1 = self.model[f'conv{i}a'](x)
            x1 = self.model[f'bn{i}a'](x1)
            x2 = self.model[f'conv{i}b'](x)
            x2 = self.model[f'bn{i}b'](x1)
            x = self.model['lr'](x1, x2)
        x = self.model[f'conv{NUM_LAYERS}'](x)
        return x
    
    
    def test_input(self):
        print("input")
        x = torch.ones(64, 1, 36, 40)
        print(x.shape)
        print("encoder out mean_shape")
        print(self.forward(x).shape)

        
class UttrDecoder(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.model = nn.ModuleDict()
        self.save_hyperparameters(params)
        
        self.model['lr'] = GLU()
        
        NUM_LAYERS= self.hparams['UTTR_DEC_NUM_LAYERS']
        
        for i in range(1, NUM_LAYERS):
            self.model[f'deconv{i}a'] = nn.ConvTranspose2d(
                self.hparams[f'UTTR_DEC_CONV{i}_CHANNELS']+int(self.hparams['UTTR_ENC_CONV5_CHANNELS']/2),
                self.hparams[f'UTTR_DEC_CONV{i+1}_CHANNELS'],
                self.hparams[f'UTTR_DEC_CONV{i}_KERNEL'],
                self.hparams[f'UTTR_DEC_CONV{i}_STRIDE'],
                self.hparams[f'UTTR_DEC_CONV{i}_PADDING'],
                bias=False, padding_mode='zeros'
            )
            self.model[f'bn{i}a'] = nn.BatchNorm2d(self.hparams[f'UTTR_DEC_CONV{i+1}_CHANNELS'])
            
            self.model[f'deconv{i}b'] = nn.ConvTranspose2d(
                self.hparams[f'UTTR_DEC_CONV{i}_CHANNELS']+int(self.hparams['UTTR_ENC_CONV5_CHANNELS']/2),
                self.hparams[f'UTTR_DEC_CONV{i+1}_CHANNELS'],
                self.hparams[f'UTTR_DEC_CONV{i}_KERNEL'],
                self.hparams[f'UTTR_DEC_CONV{i}_STRIDE'],
                self.hparams[f'UTTR_DEC_CONV{i}_PADDING'],
                self.hparams[f'UTTR_DEC_CONV{i}_OUT_PADDING'],
                bias=False, padding_mode='zeros'
            )
            self.model[f'bn{i}b'] = nn.BatchNorm2d(self.hparams[f'UTTR_DEC_CONV{i+1}_CHANNELS'])
            
        self.model[f'deconv{NUM_LAYERS}'] =  nn.ConvTranspose2d(
            self.hparams[f'UTTR_DEC_CONV{NUM_LAYERS}_CHANNELS']+int(self.hparams['UTTR_ENC_CONV5_CHANNELS']/2),
            self.hparams[f'UTTR_DEC_CONV{NUM_LAYERS+1}_CHANNELS'],
            self.hparams[f'UTTR_DEC_CONV{NUM_LAYERS}_KERNEL'],
            self.hparams[f'UTTR_DEC_CONV{NUM_LAYERS}_STRIDE'],
            self.hparams[f'UTTR_DEC_CONV{NUM_LAYERS}_PADDING'],
            bias=False, padding_mode='zeros'
        )
    
    def forward(self, x, y):
        NUM_LAYERS= self.hparams['UTTR_DEC_NUM_LAYERS']
        c = y.repeat(1, 1, x.shape[2]//y.shape[2], x.shape[3]//y.shape[3])
        z = torch.cat((x, c), dim=1)
        for i in range(1, NUM_LAYERS):
            z1 = self.model[f'deconv{i}a'](z)
            z1 = self.model[f'bn{i}a'](z1)
            z2 = self.model[f'deconv{i}b'](z)
            z2 = self.model[f'bn{i}b'](z1)
            x = self.model['lr'](z1, z2)
            c = y.repeat(1, 1, x.shape[2]//y.shape[2], x.shape[3]//y.shape[3])
            z = torch.cat((x, c), dim=1)
        x = self.model[f'deconv{NUM_LAYERS}'](z)
        return x
    
    
    def test_input(self):
        print("input")
        x = torch.ones(64, 8, 1, 10)
        print(x.shape)
        y = torch.ones(64, 8, 1, 1)
        print(y.shape)
        print("decoder out mean_shape")
        print(self.forward(x, y).shape)
        

class FaceEncoder(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.model = nn.ModuleDict()
        self.save_hyperparameters(params)
        
        self.model['lr'] = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        for i in range(1, self.hparams['FACE_ENC_CONV_LAYERS']+1):
            self.model[f'conv{i}'] = nn.Conv2d(
                self.hparams[f'FACE_ENC_CONV{i}_CHANNELS'],
                self.hparams[f'FACE_ENC_CONV{i+1}_CHANNELS'],
                self.hparams[f'FACE_ENC_CONV{i}_KERNEL'],
                self.hparams[f'FACE_ENC_CONV{i}_STRIDE'],
                self.hparams[f'FACE_ENC_CONV{i}_PADDING'],
                bias=False, padding_mode='replicate'
            )
            self.model[f'bn{i}'] = nn.BatchNorm2d(self.hparams[f'FACE_ENC_CONV{i+1}_CHANNELS'])
        
        for i in range(1, self.hparams['FACE_ENC_LINEAR_LAYERS']+1):
            self.model[f'linear{i}'] = nn.Linear(
                self.hparams[f'FACE_ENC_LINEAR{i}_CHANNELS'],
                self.hparams[f'FACE_ENC_LINEAR{i+1}_CHANNELS'],
                bias=False,
            )
            
    
    def forward(self, x):
        for i in range(1, self.hparams['FACE_ENC_CONV_LAYERS']+1):
            x = self.model[f'conv{i}'](x)
            if i != 1:
                x = self.model[f'bn{i}'](x)
            x = self.model['lr'](x)
        x = x.view(x.shape[0], -1)
        
        for i in range(1, self.hparams['FACE_ENC_LINEAR_LAYERS']+1):
            x = self.model[f'linear{i}'](x)
            x = self.model['lr'](x)
        
        x = x.unsqueeze(-1).unsqueeze(-1)
        return x
    
    
    def test_input(self):
        print("input")
        x = torch.ones(64, 3, 32, 32)
        print(x.shape)
        print("encoder out mean_shape")
        print(self.forward(x).shape)
        

class FaceDecoder(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.model = nn.ModuleDict()
        self.save_hyperparameters(params)
        
        self.model['lr'] = torch.nn.Softplus()
        
        for i in range(1, self.hparams['FACE_DEC_LINEAR_LAYERS']+1):
            self.model[f'linear{i}'] = nn.Linear(
                self.hparams[f'FACE_DEC_LINEAR{i}_CHANNELS'],
                self.hparams[f'FACE_DEC_LINEAR{i+1}_CHANNELS'],
                bias=False,
            )
        
        for i in range(1, self.hparams['FACE_DEC_CONV_LAYERS']):
            self.model[f'deconv{i}'] = nn.ConvTranspose2d(
                self.hparams[f'FACE_DEC_CONV{i}_CHANNELS'],
                self.hparams[f'FACE_DEC_CONV{i+1}_CHANNELS'],
                self.hparams[f'FACE_DEC_CONV{i}_KERNEL'],
                self.hparams[f'FACE_DEC_CONV{i}_STRIDE'],
                self.hparams[f'FACE_DEC_CONV{i}_PADDING'],
                self.hparams[f'FACE_DEC_CONV{i}_OUT_PADDING'],
                bias=False, padding_mode='zeros'
            )
            self.model[f'bn{i}'] = nn.BatchNorm2d(self.hparams[f'FACE_DEC_CONV{i+1}_CHANNELS'])
        
        i = self.hparams["FACE_DEC_CONV_LAYERS"]
        self.model[f'conv{i}']  = nn.Conv2d(
            self.hparams[f'FACE_DEC_CONV{i}_CHANNELS'],
            self.hparams[f'FACE_DEC_CONV{i+1}_CHANNELS'],
            self.hparams[f'FACE_DEC_CONV{i}_KERNEL'],
            self.hparams[f'FACE_DEC_CONV{i}_STRIDE'],
            self.hparams[f'FACE_DEC_CONV{i}_PADDING'],
            bias=False, padding_mode='replicate'
        )
            
            
    
    def forward(self, x):
        for i in range(1, params['FACE_DEC_LINEAR_LAYERS']+1):
            x = self.model[f'linear{i}'](x)
            x = self.model['lr'](x)
        h = int(math.sqrt(params[f'FACE_DEC_LINEAR{params["FACE_DEC_LINEAR_LAYERS"]+1}_CHANNELS'] / params[f'FACE_DEC_CONV1_CHANNELS']))
        x = x.view(x.shape[0], params[f'FACE_DEC_CONV1_CHANNELS'], h, h)
        for i in range(1, params['FACE_DEC_CONV_LAYERS']):
            x = self.model[f'deconv{i}'](x)
            if i != params['FACE_DEC_CONV_LAYERS']+1:
                x = self.model[f'bn{i}'](x)
            x = self.model['lr'](x)
        
        i = params['FACE_DEC_CONV_LAYERS']
        x = self.model[f'conv{i}'](x)
            
        return x
    
    
    def test_input(self):
        print("input")
        x = torch.ones(64, 8)
        print(x.shape)
        print("decoder out mean_shape")
        print(self.forward(x).shape)

        
class VoiceEncoder(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.model = nn.ModuleDict()
        self.save_hyperparameters(params)
        
        self.model['lr'] = GLU()
        
        NUM_LAYERS = self.hparams['VOICE_ENC_NUM_LAYERS']
        
        for i in range(1, NUM_LAYERS):
            self.model[f'conv{i}a'] = nn.Conv2d(
                self.hparams[f'VOICE_ENC_CONV{i}_CHANNELS'],
                self.hparams[f'VOICE_ENC_CONV{i+1}_CHANNELS'],
                self.hparams[f'VOICE_ENC_CONV{i}_KERNEL'],
                self.hparams[f'VOICE_ENC_CONV{i}_STRIDE'],
                self.hparams[f'VOICE_ENC_CONV{i}_PADDING'],
                bias=False, padding_mode='replicate'
            )
            self.model[f'bn{i}a'] = nn.BatchNorm2d(self.hparams[f'VOICE_ENC_CONV{i+1}_CHANNELS'])
            
            self.model[f'conv{i}b'] = nn.Conv2d(
                self.hparams[f'VOICE_ENC_CONV{i}_CHANNELS'],
                self.hparams[f'VOICE_ENC_CONV{i+1}_CHANNELS'],
                self.hparams[f'VOICE_ENC_CONV{i}_KERNEL'],
                self.hparams[f'VOICE_ENC_CONV{i}_STRIDE'],
                self.hparams[f'VOICE_ENC_CONV{i}_PADDING'],
                bias=False, padding_mode='replicate'
            )
            self.model[f'bn{i}b'] = nn.BatchNorm2d(self.hparams[f'VOICE_ENC_CONV{i+1}_CHANNELS'])
            
        self.model[f'conv{NUM_LAYERS}'] =  nn.Conv2d(
            self.hparams[f'VOICE_ENC_CONV{NUM_LAYERS}_CHANNELS'],
            self.hparams[f'VOICE_ENC_CONV{NUM_LAYERS+1}_CHANNELS'],
            self.hparams[f'VOICE_ENC_CONV{NUM_LAYERS}_KERNEL'],
            self.hparams[f'VOICE_ENC_CONV{NUM_LAYERS}_STRIDE'],
            self.hparams[f'VOICE_ENC_CONV{NUM_LAYERS}_PADDING'],
            bias=False, padding_mode='replicate'
        )
    
    def forward(self, x):
        NUM_LAYERS = self.hparams['VOICE_ENC_NUM_LAYERS']
        for i in range(1, NUM_LAYERS):
            x1 = self.model[f'conv{i}a'](x)
            x1 = self.model[f'bn{i}a'](x1)
            x2 = self.model[f'conv{i}b'](x)
            x2 = self.model[f'bn{i}b'](x1)
            x = self.model['lr'](x1, x2)
        x = self.model[f'conv{NUM_LAYERS}'](x)
        return x
    
    
    def test_input(self):
        print("input")
        x = torch.ones(64, 1, 36, 40)
        print(x.shape)
        print("encoder out mean_shape")
        print(self.forward(x).shape)


params['NUM_EMBEDDINGS'] = 16
params['EMBEDDINGS_DIM'] = 8
params['BETA'] = 0.25
        
class VectorQuantizer(pl.LightningModule):
    """
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        
        self.n_e = self.hparams['NUM_EMBEDDINGS']
        self.e_dim = self.hparams['EMBEDDINGS_DIM']
        self.beta = self.hparams['BETA']

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
    
    def forward(self, z):
         # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        return z_q.permute(0, 3, 1, 2).contiguous()


    def loss(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.sum((z_q.detach()-z)**2) + self.beta * \
            torch.sum((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
        
        
        
class Model(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.ue = UttrEncoder(self.hparams)
        self.fe = FaceEncoder(self.hparams)
        self.ve = VoiceEncoder(self.hparams)
        self.ud = UttrDecoder(self.hparams)
        self.fd = FaceDecoder(self.hparams)
        
    def forward(self, x, y):
        z, _ = torch.chunk(self.ue(x), 2, dim=1)
        c, _ = torch.chunk(self.fe(y), 2, dim=1)
        x, _ = torch.chunk(self.ud(z, c), 2, dim=1)
        return x
    
    def rc_image(self, y):
        c, _ = torch.chunk(self.fe(y), 2, dim=1)
        y, _ = torch.chunk(self.fd(c.squeeze(-1).squeeze(-1)), 2, dim=1)
        return y.to(torch.uint8).squeeze(0)
    
    def loss_function(self, x, y):
        mu, log_var = torch.chunk(self.ue(x), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        uttr_kl = self._KL_divergence(mu, log_var)
        z = self._sample_z(mu, log_var)
        
        mu, log_var = torch.chunk(self.fe(y), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        face_kl = self._KL_divergence(mu, log_var)
        c = self._sample_z(mu, log_var)
    
        mu, log_var = torch.chunk(self.ud(z, c), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        uttr_rc = self._reconstruction(x, mu, log_var)
        x_hat = self._sample_z(mu, log_var)
        
        mu, log_var = torch.chunk(self.fd(c.squeeze(-1).squeeze(-1)), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        face_rc = self._reconstruction(y, mu, log_var)
        
        mu, log_var = torch.chunk(self.ve(x_hat), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        voice_rc = []
        
        for i, j in zip(torch.tensor_split(mu, mu.shape[-1], dim=-1),torch.tensor_split(log_var, log_var.shape[-1], dim=-1)):
            voice_rc.append(self._reconstruction(c, i, j))
            
        voice_rc = torch.sum(torch.stack(voice_rc)).to(self.device)/len(voice_rc)
        
        return  uttr_rc, face_rc, voice_rc, uttr_kl, face_kl
    
    def training_step(self, batch, batch_idx):
        x, y = batch[1], batch[0]
        uttr_rc, face_rc, voice_rc, uttr_kl, face_kl = self.loss_function(x, y)
        uttr_loss = self.hparams["LAMBDA1"]*uttr_rc 
        uttr_loss += self.hparams["LAMBDA4"]*uttr_kl
        
        face_loss = self.hparams["LAMBDA2"]*face_rc
        face_loss += self.hparams["LAMBDA3"]*voice_rc
        face_loss += self.hparams["LAMBDA5"]*face_kl
        
        loss_schedule = self.current_epoch//200
        loss = uttr_loss/(2**loss_schedule) + face_loss/(4**loss_schedule)
        
        self.log("uttr_rc", uttr_rc)
        self.log("face_rc", face_rc)
        self.log("voice_rc", voice_rc)
        self.log("uttr_kl", uttr_kl)
        self.log("face_kl", face_kl)
        
        self.log("train_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[1], batch[0]
        uttr_rc, face_rc, voice_rc, uttr_kl, face_kl = self.loss_function(x, y)
        loss = self.hparams["LAMBDA1"]*uttr_rc 
        loss += self.hparams["LAMBDA2"]*face_rc
        loss += self.hparams["LAMBDA3"]*voice_rc
        loss += self.hparams["LAMBDA4"]*uttr_kl
        loss += self.hparams["LAMBDA5"]*face_kl
        
        self.log("val_loss", loss)

        
    
    def test_input(self):
        print("input")
        x = torch.ones(64, 1, 36, 40)
        y = torch.ones(64, 3, 32, 32)
        print(x.shape)
        print(y.shape)
        print("encoder out mean_shape")
        print(self.forward(x, y).shape)
        
    def test_loss(self):
        print("input")
        x = torch.ones(64, 1, 36, 40)
        y = torch.ones(64, 3, 32, 32)
        print(x.shape)
        print(y.shape)
        print("output loss_function")
        print(self.loss_function(x, y))
        
    def _KL_divergence(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2)  - log_var.exp())
    
    def _reconstruction(self, x, mu, log_var):
        return torch.sum(log_var + torch.square(x-mu)/log_var.exp())*0.5
    
    def _sample_z(self, mu, log_var):
        epsilon = torch.randn(mu.shape, device=self.device)
        return mu + log_var.exp() * epsilon
     
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams["LR"])

        return [opt], []

    
params['FACE_ENC_LINEAR3_CHANNELS'] = 8
class VQModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.ue = UttrEncoder(self.hparams)
        self.fe = FaceEncoder(self.hparams)
        self.ve = VoiceEncoder(self.hparams)
        self.ud = UttrDecoder(self.hparams)
        self.fd = FaceDecoder(self.hparams)
        
        self.vq = VectorQuantizer(self.hparams)
        
    def forward(self, x, y):
        z, _ = torch.chunk(self.ue(x), 2, dim=1)
        c_e = self.fe(y)
        c_q = self.vq(c_e)
        x, _ = torch.chunk(self.ud(z, c_q), 2, dim=1)
        return x
    
    def rc_image(self, y):
        c_e = self.fe(y)
        c_q = self.vq(c_e)
        print(c_q)
        y, _ = torch.chunk(self.fd(c_q.squeeze(-1).squeeze(-1)), 2, dim=1)
        return y.to(torch.uint8).squeeze(0)
    
    def loss_function(self, x, y):
        mu, log_var = torch.chunk(self.ue(x), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        uttr_kl = self._KL_divergence(mu, log_var)
        z = self._sample_z(mu, log_var)
        
        c_e = self.fe(y)
        vq_loss, c_q, _, _, _  = self.vq.loss(c_e)   
    
        mu, log_var = torch.chunk(self.ud(z, c_q), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        uttr_rc = self._reconstruction(x, mu, log_var)
        x_hat = self._sample_z(mu, log_var)
        
        mu, log_var = torch.chunk(self.fd(c_q.squeeze(-1).squeeze(-1)), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        face_rc = self._reconstruction(y, mu, log_var)
        
        mu, log_var = torch.chunk(self.ve(x_hat), 2, dim=1)
        log_var = torch.sigmoid(log_var)
        voice_rc = []
        
        for i, j in zip(torch.tensor_split(mu, mu.shape[-1], dim=-1),torch.tensor_split(log_var, log_var.shape[-1], dim=-1)):
            voice_rc.append(self._reconstruction(c_q, i, j))
            
        voice_rc = torch.sum(torch.stack(voice_rc)).to(self.device)/len(voice_rc)
        
        return  uttr_rc, face_rc, voice_rc, uttr_kl, vq_loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch[1], batch[0]
        uttr_rc, face_rc, voice_rc, uttr_kl, vq_loss = self.loss_function(x, y)
        uttr_loss = self.hparams["LAMBDA1"]*uttr_rc 
        uttr_loss += self.hparams["LAMBDA4"]*uttr_kl
        
        face_loss = self.hparams["LAMBDA2"]*face_rc
        face_loss += self.hparams["LAMBDA3"]*voice_rc
        face_loss += self.hparams["LAMBDA5"]*vq_loss
        
        loss_schedule = self.current_epoch//200
        loss = uttr_loss/(2**loss_schedule) + face_loss/(2**loss_schedule)
        
        self.log("uttr_rc", uttr_rc)
        self.log("face_rc", face_rc)
        self.log("voice_rc", voice_rc)
        self.log("uttr_kl", uttr_kl)
        self.log("vq_loss", vq_loss)
        
        self.log("train_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[1], batch[0]
        uttr_rc, face_rc, voice_rc, uttr_kl, vq_loss = self.loss_function(x, y)
        loss = self.hparams["LAMBDA1"]*uttr_rc 
        loss += self.hparams["LAMBDA2"]*face_rc
        loss += self.hparams["LAMBDA3"]*voice_rc
        loss += self.hparams["LAMBDA4"]*uttr_kl
        loss += self.hparams["LAMBDA5"]*vq_loss
        
        self.log("val_loss", loss)

        
    
    def test_input(self):
        print("input")
        x = torch.ones(64, 1, 36, 40)
        y = torch.ones(64, 3, 32, 32)
        print(x.shape)
        print(y.shape)
        print("encoder out mean_shape")
        print(self.forward(x, y).shape)
        
    def test_loss(self):
        print("input")
        x = torch.ones(64, 1, 36, 40)
        y = torch.ones(64, 3, 32, 32)
        print(x.shape)
        print(y.shape)
        print("output loss_function")
        print(self.loss_function(x, y))
        
    def _KL_divergence(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2)  - log_var.exp())
    
    def _reconstruction(self, x, mu, log_var):
        return torch.sum(log_var + torch.square(x-mu)/log_var.exp())*0.5
    
    def _sample_z(self, mu, log_var):
        epsilon = torch.randn(mu.shape, device=self.device)
        return mu + log_var.exp() * epsilon
     
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams["LR"])

        return [opt], []

    
if __name__ == "__main__":
    model = VQModel(params)
    model.test_loss()