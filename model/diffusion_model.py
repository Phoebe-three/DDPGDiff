import torch
import torch.nn as nn
import math
from physical_modules import PCGA, EDCM
from base_UNet import ComplexConv2d, ComplexBatchNorm2d, ComplexLeakyReLU, ComplexUpsample, ComplexMaxPool2d
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ComplexCondConvBlock(nn.Module):
    """Time Embedding Block"""

    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super(ComplexCondConvBlock, self).__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.relu1 = ComplexLeakyReLU()

        self.conv2 = ComplexConv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = ComplexBatchNorm2d(out_channels)
        self.relu2 = ComplexLeakyReLU()

        self.PCGA = PCGA(out_channels)
        #

        if time_emb_dim is not None:
            self.time_mlp_real = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
            self.time_mlp_imag = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp_real = None
            self.time_mlp_imag = None

    def forward(self, x, t_emb=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        if t_emb is not None and self.time_mlp_real is not None:
            t_real = self.time_mlp_real(t_emb).unsqueeze(-1).unsqueeze(-1)
            t_imag = self.time_mlp_imag(t_emb).unsqueeze(-1).unsqueeze(-1)
            x = x + torch.complex(t_real, t_imag)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.PCGA(x)
        return x

class ComplexConditionEncoder(nn.Module):
    def __init__(self, input_channels=1, base_channels=16):
        super().__init__()
        self.cond1 = ComplexConv2d(input_channels, base_channels, 3, padding=1)
        self.cond2 = ComplexConv2d(base_channels, base_channels * 2, 3, padding=1)
        self.cond3 = ComplexConv2d(base_channels * 2, base_channels * 4, 3, padding=1)
        self.cond4 = ComplexConv2d(base_channels * 4, base_channels * 8, 3, padding=1)

        self.pool1 = ComplexMaxPool2d(2)
        self.pool2 = ComplexMaxPool2d(2)
        self.pool3 = ComplexMaxPool2d(2)

        self.act1 = ComplexLeakyReLU()
        self.act2 = ComplexLeakyReLU()
        self.act3 = ComplexLeakyReLU()
        self.act4 = ComplexLeakyReLU()

    def forward(self, condition):
        c1 = self.act1(self.cond1(condition))
        c2 = self.act2(self.cond2(self.pool1(c1)))
        c3 = self.act3(self.cond3(self.pool2(c2)))
        c4 = self.act4(self.cond4(self.pool3(c3)))
        return c1, c2, c3, c4

class DDNPNet(nn.Module):
    def __init__(self,input_channels=1,
                 output_channels=1,
                 base_channels=16,
                 time_dim=64,
                 dx=1,
                 dy=2,
                 F0=78.8e9,
                 branch=True):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(time_dim),nn.Linear(time_dim,time_dim*4),nn.GELU(),nn.Linear(time_dim*4,time_dim))
        self.enc1 = ComplexCondConvBlock(input_channels,base_channels,time_dim)
        self.pool1 = ComplexMaxPool2d(2)

        self.enc2 = ComplexCondConvBlock(base_channels,base_channels*2,time_dim)
        self.pool2 = ComplexMaxPool2d(2)

        self.enc3 = ComplexCondConvBlock(base_channels*2,base_channels*4,time_dim)
        self.pool3 = ComplexMaxPool2d(2)

        self.bottleneck = ComplexCondConvBlock(base_channels*4,base_channels*8,time_dim)

        self.up3 = ComplexUpsample(scale_factor=2)
        self.dec3 = ComplexCondConvBlock(base_channels*8+base_channels*4,base_channels*4,time_dim)

        self.up2 = ComplexUpsample(scale_factor=2)
        self.dec2 = ComplexCondConvBlock(base_channels*4+base_channels*2,base_channels*2,time_dim)

        self.up1 = ComplexUpsample(scale_factor=2)
        self.dec1 = ComplexCondConvBlock(base_channels*2+base_channels,base_channels,time_dim)

        self.final_conv = ComplexConv2d(base_channels,output_channels,1)
        self.condition_encoder = ComplexConditionEncoder(input_channels,base_channels)

        self.cond_proj1 = ComplexConv2d(base_channels,base_channels,1)
        self.cond_proj2 = ComplexConv2d(base_channels*2,base_channels*2,1)
        self.cond_proj3 = ComplexConv2d(base_channels*4,base_channels*4,1)
        self.cond_proj4 = ComplexConv2d(base_channels*8,base_channels*8,1)

        self.branch = branch
        if self.branch:
            self.phys_branch = EDCM(in_channels=input_channels,
                                      out_channels=base_channels,
                                      dx=dx,
                                      dy=dy,
                                      F0=F0)
            self.fusion_layer = ComplexConv2d(base_channels*2, base_channels, 3, padding=1)

    def forward(self, x, time, condition):

        t_emb = self.time_mlp(time)
        c1,c2,c3,c4 = self.condition_encoder(condition)

        e1 = self.enc1(x,t_emb)
        e1 = e1 + self.cond_proj1(c1)

        e2 = self.enc2(self.pool1(e1),t_emb)
        e2 = e2 + self.cond_proj2(c2)

        e3 = self.enc3(self.pool2(e2),t_emb)
        e3 = e3 + self.cond_proj3(c3)

        e4 = self.bottleneck(self.pool3(e3),t_emb)
        e4 = e4 + self.cond_proj4(c4)

        d3 = self.up3(e4)

        if e3.shape[2:] != d3.shape[2:]:
            e3 = ComplexUpsample(size=d3.shape[2:])(e3)
        d3 = self.dec3(torch.cat([d3,e3],dim=1),t_emb)
        d2 = self.up2(d3)

        if e2.shape[2:] != d2.shape[2:]:
            e2 = ComplexUpsample(size=d2.shape[2:])(e2)
        d2 = self.dec2(torch.cat([d2, e2],dim=1),t_emb)
        d1 = self.up1(d2)

        if e1.shape[2:] != d1.shape[2:]:
            e1 = ComplexUpsample(size=d1.shape[2:])(e1)
        d1 = self.dec1(torch.cat([d1,e1],dim=1),t_emb)
        unet_noise_feat = d1

        if self.branch:
            if unet_noise_feat.shape[2:] != x.shape[2:]:
                unet_noise_feat = ComplexUpsample(size=x.shape[2:])(unet_noise_feat)
            phys_noise_feat = self.phys_branch(x)
            fused_feat = torch.cat([unet_noise_feat,phys_noise_feat],dim=1)
            fused_feat = self.fusion_layer(fused_feat)
            return self.final_conv(fused_feat)

        return self.final_conv(unet_noise_feat)

class ConditionalComplexDDPM(nn.Module):
    def __init__(self,model,timesteps=1000,beta_start=1e-4,beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start,beta_end,timesteps)
        alphas = 1.0-betas
        alphas_cumprod = torch.cumprod(alphas,dim=0)
        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.0-alphas_cumprod))

    def get_complex_noise(self,x):
        noise_real = torch.randn_like(x.real)/math.sqrt(2)
        noise_imag = torch.randn_like(x.imag)/math.sqrt(2)
        return torch.complex(noise_real,noise_imag)

    def forward(self,x_0,condition,R=None):
        t = torch.randint(0,self.timesteps,(x_0.shape[0],),device=x_0.device).long()
        noise = self.get_complex_noise(x_0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        x_t = sqrt_ab*x_0+sqrt_1mab*noise
        pred_noise = self.model(x_t, t, condition,R)
        loss_real = F.mse_loss(pred_noise.real,noise.real)
        loss_imag = F.mse_loss(pred_noise.imag,noise.imag)
        return loss_real+loss_imag







