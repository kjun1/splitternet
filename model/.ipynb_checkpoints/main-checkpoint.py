from .child import Encoder, Decoder, Discriminator
import pytorch_lightning as pl
import torch
from torch import nn
import itertools

class SplitterNet(pl.LightningModule):
    def __init__(self):
        super(SplitterNet, self).__init__()
        self.content_encoder = Encoder()
        self.attribute_encoder = Encoder()
        self.decoder = Decoder()
        self.content_discriminator = Discriminator()
        self.attribute_discriminator = Discriminator()

    def forward(self, x, y):
        z = self.content_encoder(x)
        mu1 = z.chunk(2, dim=1)[0]
        z = self.attribute_encoder(y)
        mu2 = z.chunk(2, dim=1)[0]
        # mu1 mu2 固める
        out = torch.cat([mu1, mu2], dim=1)
        out = self.decoder(out)

        return out

    def gen_loss(self, x, y):
        out = self.content_encoder(x)
        mu, log_var = out.chunk(2, dim=1)
        
        content_kl = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        eps = torch.randn_like(torch.exp(log_var))
        z1 = mu + torch.exp(log_var / 2) * eps
        
        y_hat = self.content_discriminator(z1)
        content_dis =  nn.functional.binary_cross_entropy(y_hat, y)
        
        out = self.attribute_encoder(x)
        mu, log_var = out.chunk(2, dim=1)
        
        attribute_kl = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        eps = torch.randn_like(torch.exp(log_var))
        z2 = mu + torch.exp(log_var / 2) * eps
        
        y_hat = self.content_discriminator(z2)
        attribute_dis =  nn.functional.binary_cross_entropy(y_hat, y)
        
        z = torch.cat([z1, z2], dim=1)
        x_hat = self.decoder(z)
        
        reconstruction = nn.functional.mse_loss(x, x_hat, reduction='sum')
        
        out = self.content_encoder(x_hat)
        mu, log_var = out.chunk(2, dim=1)
        
        eps = torch.randn_like(torch.exp(log_var))
        z1_hat = mu + torch.exp(log_var / 2) * eps
        z1_reconstruction = nn.functional.mse_loss(z1, z1_hat, reduction='sum')
        
        loss = reconstruction + z1_reconstruction - content_dis + 0.5*attribute_dis + 0.05*content_kl + attribute_kl

        
        return loss
    
    def dis_loss(self, x, y, dis_idx):
        if dis_idx == 0:
            encoder = self.content_encoder
            discriminator = self.content_discriminator
        if dis_idx == 1:
            encoder = self.attribute_encoder
            discriminator = self.attribute_discriminator
        
        out = encoder(x)
        mu, log_var = out.chunk(2, dim=1)
        
        eps = torch.randn_like(torch.exp(log_var))
        z = mu + torch.exp(log_var / 2) * eps
        
        y_hat = discriminator(z)
        
        loss = nn.functional.binary_cross_entropy(y_hat, y)
        
        return loss
        
    
    # train ループを定義
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch[0], batch[1].float()
        if optimizer_idx==0:
            loss = self.dis_loss(x, y, dis_idx=optimizer_idx)
        
        if optimizer_idx==1:
            loss = self.dis_loss(x, y, dis_idx=optimizer_idx)
        
        if optimizer_idx==2:
            loss = self.gen_loss(x, y)
        
        self.log("train_loss", loss)
        return loss
    
    def test_input(self):
        x = torch.ones(64, 1, 32, 32)
        y = torch.ones(64, 100)
        print("encoder out *2")
        print(self.content_encoder(x).shape)
        print("decoder out")
        print(self.forward(x,x).shape)
        try:
            self.dis_loss(x, y, dis_idx=0)
        except:
            print("content discriminator error")
            
        try:
            self.dis_loss(x, y, dis_idx=1)
        except:
            print("attribute discriminator error")
        
        try:
            self.forward(x, x)
        except:
            print("generator error")
        
        

    def configure_optimizers(self):
        Dc_opt = torch.optim.Adam(self.content_discriminator.parameters(), lr=1e-3)
        Da_opt = torch.optim.Adam(self.attribute_discriminator.parameters(), lr=1e-3)
        params = [self.content_encoder.parameters(), self.attribute_encoder.parameters(), self.decoder.parameters()]
        G_opt = Dc_opt = torch.optim.Adam(itertools.chain(*params), lr=1e-3)
        return [Dc_opt, Da_opt, G_opt], []
