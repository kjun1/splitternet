from .child import AttributeEncoder, ContentEncoder, Decoder, AttributeDiscriminator, ContentDiscriminator
import pytorch_lightning as pl
import torch
from torch import nn
import itertools

class SplitterNet(pl.LightningModule):
    def __init__(self):
        super(SplitterNet, self).__init__()
        self.content_encoder = ContentEncoder()
        self.attribute_encoder = AttributeEncoder()
        self.decoder = Decoder()
        self.content_discriminator = ContentDiscriminator()
        self.attribute_discriminator = AttributeDiscriminator()
        self.Da_lr = 1e-8
        self.Dc_lr = 1e-8
        self.G_lr = 1e-4
        self.h = [1.0, 0.03, -1.5, 1.5, 1.55, 1.525] #[1.0, 0.03, -1.0, 1.0, 0.60, 0.55]

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
        
        reconstruction = nn.functional.l1_loss(x, x_hat, reduction='sum')
        
        out = self.attribute_encoder(x_hat)
        mu, log_var = out.chunk(2, dim=1)
        
        eps = torch.randn_like(torch.exp(log_var))
        z2_hat = mu + torch.exp(log_var / 2) * eps
        z2_reconstruction = nn.functional.mse_loss(z2, z2_hat, reduction='sum')
        
        loss = [reconstruction, z2_reconstruction, content_dis, attribute_dis, content_kl, attribute_kl]

        
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
            reconstruction, z2_reconstruction, content_dis, attribute_dis, content_kl, attribute_kl= self.gen_loss(x, y)
            loss = self.h[0]*reconstruction + self.h[1]*z2_reconstruction + self.h[2]*content_dis + self.h[3]*attribute_dis + self.h[4]*content_kl + self.h[5]*attribute_kl
            self.log("reconstruction", reconstruction)
            self.log("z2_reconstruction", z2_reconstruction)
            self.log("content_dis", content_dis)
            self.log("attribute_dis", attribute_dis)
            self.log("content_kl", content_kl)
            self.log("attribute_kl", attribute_kl)
            self.log("train_loss", loss)
        return loss
    
    # define valid loop
    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1].float()
        reconstruction, z2_reconstruction, content_dis, attribute_dis, content_kl, attribute_kl = self.gen_loss(x, y)
        loss = self.h[0]*reconstruction #+ self.h[1]*z1_reconstruction + self.h[2]*content_dis + self.h[3]*attribute_dis + self.h[4]*content_kl + self.h[5]*attribute_kl
        self.log("val_loss", loss)
    
    def test_input(self):
        x = torch.ones(64, 1, 32, 32)
        y = torch.ones(64, 51)
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
        Dc_opt = torch.optim.AdamW(self.content_discriminator.parameters(), lr=self.Dc_lr)
        Da_opt = torch.optim.AdamW(self.attribute_discriminator.parameters(), lr=self.Da_lr)
        params = [self.content_encoder.parameters(), self.attribute_encoder.parameters(), self.decoder.parameters()]
        G_opt = torch.optim.AdamW(itertools.chain(*params), lr=self.G_lr)
        
        #Dc_sch = torch.optim.lr_scheduler.LinearLR(Dc_opt, start_factor=self.Dc_lr, end_factor=1e-6, total_iters=1000)
        #Da_sch = torch.optim.lr_scheduler.LinearLR(Da_opt, start_factor=self.Da_lr, end_factor=1e-6, total_iters=1000)
        #G_sch = torch.optim.lr_scheduler.LinearLR(G_opt, start_factor=self.G_lr, end_factor=1e-6, total_iters=1000)
        return [Dc_opt, Da_opt, G_opt], []#[Dc_sch, Da_sch, G_sch]
