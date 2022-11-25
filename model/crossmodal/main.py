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
    
params["LAMBDA1"] = 1
params["LAMBDA2"] = 1
params["LAMBDA3"] = 0.4
params["LAMBDA4"] = 0.5
params["LAMBDA5"] = 1
    
params["LR"] = 1e-3
    
    
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
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.sigmoid(log_var)
        return mean, log_var
    
    
    def test_input(self):
        print("input")
        x = torch.ones(64, 1, 36, 40)
        print(x.shape)
        print("encoder out mean_shape")
        print(self.forward(x)[0].shape)

        
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
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.sigmoid(log_var)
        return mean, log_var
    
    
    def test_input(self):
        print("input")
        x = torch.ones(64, 8, 1, 10)
        print(x.shape)
        y = torch.ones(64, 8, 1, 1)
        print(y.shape)
        print("decoder out mean_shape")
        print(self.forward(x, y)[0].shape)
        

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
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.sigmoid(log_var)
        return mean, log_var
    
    
    def test_input(self):
        print("input")
        x = torch.ones(64, 3, 32, 32)
        print(x.shape)
        print("encoder out mean_shape")
        print(self.forward(x)[0].shape)
        

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
            
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.sigmoid(log_var)
        return mean, log_var
    
    
    def test_input(self):
        print("input")
        x = torch.ones(64, 8)
        print(x.shape)
        print("decoder out mean_shape")
        print(self.forward(x)[0].shape)

        
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
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.sigmoid(log_var)
        return mean, log_var
    
    
    def test_input(self):
        print("input")
        x = torch.ones(64, 1, 36, 40)
        print(x.shape)
        print("encoder out mean_shape")
        print(self.forward(x)[0].shape)

        
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
        z = self.ue(x)
        c = self.fe(y)
        x = self.ud(z[0], c[0])
        return x[0]
    
    def rc_image(self, y):
        c = self.fe(y)
        y = self.fd(c[0].squeeze(-1).squeeze(-1))
        return y[0].to(torch.uint8).squeeze(0)
    
    def speech_to_face(self, x, y):
        x_hat = self.forward(x, y)
        c_hat = self.ve(x_hat)
        c_hat = torch.tensor_split(c_hat[0], c_hat[0].shape[-1], dim=-1)
        y_hat = self.fd(c_hat[0].squeeze(-1).squeeze(-1))
        return y_hat[0].to(torch.uint8).squeeze(0)
    
    def loss_function(self, x, y):
        mu, log_var = self.ue(x)
        uttr_kl = self._KL_divergence(mu, log_var)
        z = self._sample_z(mu, log_var)
        
        mu, log_var = self.fe(y)
        face_kl = self._KL_divergence(mu, log_var)
        c = self._sample_z(mu, log_var)
    
        mu, log_var = self.ud(z, c)
        uttr_rc = self._reconstruction(x, mu, log_var)
        x_hat = self._sample_z(mu, log_var)
        
        mu, log_var = self.fd(c.squeeze(-1).squeeze(-1))
        face_rc = self._reconstruction(y, mu, log_var)
        
        mu, log_var = self.ve(x_hat)
        c_hat = self._sample_z(mu, log_var)
        
        voice_rc = []
        
        for i in torch.tensor_split(c_hat, c_hat.shape[-1], dim=-1):
            mu, log_var = self.fd(i.squeeze(-1).squeeze(-1))
            voice_rc.append(self._reconstruction(y, mu, log_var))
            
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
    
if __name__ == "__main__":
    model = Model(params)
    model.test_loss()