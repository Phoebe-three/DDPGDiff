import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from base_UNet import ComplexConv2d, compute_phase_factors
from scipy.constants import speed_of_light as c

class PCGA(nn.Module):
    """
       PCGA
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.pad = kernel_size // 2

        # Fixed sliding window summation kernel (box filter)
        box = torch.ones(channels, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        self.register_buffer('box_kernel', box)

        self.conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        eps = 1e-8
        magnitude = torch.sqrt(x.real ** 2 + x.imag ** 2 + eps)


        B = x.shape[0]
        stacked = torch.cat([x.real, x.imag, magnitude], dim=0)  # [3B, C, H, W]
        box_kernel = self.box_kernel.to(dtype=stacked.dtype)
        filtered = F.conv2d(stacked, box_kernel, padding=self.pad, groups=self.channels)
        sum_real, sum_imag, sum_magnitude = filtered.split(B, dim=0)

        vector_sum_mag = torch.sqrt(sum_real ** 2 + sum_imag ** 2 + eps)
        coherence = vector_sum_mag / (sum_magnitude + eps)

        effective_magnitude = magnitude * coherence
        attention_weights = self.sigmoid(self.conv(effective_magnitude))

        real_attended = x.real * attention_weights
        imag_attended = x.imag * attention_weights

        return torch.complex(real_attended, imag_attended)


class EDCM(nn.Module):

    def __init__(self,dx,dy,F0,R,in_channels=1,out_channels=16):
        super(EDCM, self).__init__()

        # -----------------------------
        # physical parameters
        # -----------------------------
        self.dx = dx * 1e-3
        self.dy = dy * 1e-3
        self.F0 = F0
        self.R_default = R
        self.k0 = 2 * np.pi * F0 / c

        # -----------------------------
        # dynamic cache
        # -----------------------------
        self.cached_H = None
        self.cached_W = None
        self.cached_device = None

        self.kx_grid = None
        self.ky_grid = None
        self.kz_grid = None

        self.x_grid = None
        self.y_grid = None

        # -----------------------------
        # learnable physics relaxation
        # -----------------------------
        self.phase_relax = nn.Parameter(torch.tensor(1.0))
        # -----------------------------
        # network blocks
        # -----------------------------
        self.align_conv = ComplexConv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.init_conv = ComplexConv2d(in_channels,out_channels,kernel_size=3,padding=1)

        self.final_conv = ComplexConv2d(out_channels,in_channels,kernel_size=3,padding=1)

        self.isf_modulator = ComplexConv2d(1,in_channels,kernel_size=1)


    def _update_physical_grids(self,H, W,device):
        """
        Update grid
        """
        if (self.cached_H == H and self.cached_W == W and self.cached_device == device):
            return

        # ==========================
        # frequency domain grids
        # ==========================
        kx = torch.fft.fftshift(torch.fft.fftfreq(W,d=self.dx,device=device)) * 2*np.pi
        ky = torch.fft.fftshift(torch.fft.fftfreq(H,d=self.dy,device=device)) * 2*np.pi
        ky_grid,kx_grid = torch.meshgrid(ky,kx,indexing='ij')

        kz_sq = (4 * self.k0 ** 2 - kx_grid ** 2 - ky_grid ** 2)
        kz_sq = torch.clamp(kz_sq, min=0.0)

        kz_grid=torch.sqrt(kz_sq)

        self.kx_grid = kx_grid
        self.ky_grid = ky_grid
        self.kz_grid = kz_grid

        # ==========================
        # spatial grids for LSF
        # ==========================

        x_coord = (torch.arange( W,device=device)-W//2) * self.dx
        y_coord = (torch.arange(H,device=device)-H//2) * self.dy
        y_grid,x_grid = torch.meshgrid(y_coord,x_coord,indexing='ij')

        self.x_grid=x_grid
        self.y_grid=y_grid

        self.cached_H=H
        self.cached_W=W
        self.cached_device=device

    def _get_dynamic_phase_operator(self, R):

        phase_angle = (self.kz_grid.unsqueeze(0).unsqueeze(0)*R*self.phase_relax)
        phi_forward = torch.complex(torch.cos(phase_angle),torch.sin(phase_angle))
        phi_forward = torch.fft.ifftshift(phi_forward,dim=(-2,-1))
        phi_adjoint = torch.conj(phi_forward)

        return phi_forward,phi_adjoint

    def forward(self, x, R=None):

        B,C,H,W=x.shape
        device=x.device
        # -------------------------
        # update cache
        # -------------------------
        self._update_physical_grids(H, W, device)
        # -------------------------
        # process R
        # -------------------------
        if R is None:
            R = torch.full((B,1,1,1),self.R_default,device=device,dtype=x.real.dtype)

        elif R.dim()==1:
            R = R.reshape(B,1,1,1)

        phi_forward, phi_adjoint = (self._get_dynamic_phase_operator(R))

        X_freq = torch.fft.fft2(x)
        echo_domain = torch.fft.ifft2(X_freq*phi_forward)

        echo_domain = self.init_conv(echo_domain)

        r_dist = torch.sqrt(self.x_grid ** 2 + self.y_grid ** 2 + R ** 2 + 1e-8)

        K_x = self.x_grid/r_dist
        K_y = self.y_grid/r_dist

        isf_map = torch.complex(K_x, K_y)

        modulation_weight = self.isf_modulator(isf_map)
        echo_modulated = (echo_domain * modulation_weight)

        echo_noise_feat = self.align_conv(echo_modulated)

        echo_noise_feat = (echo_domain + echo_noise_feat)

        echo_noise_freq = torch.fft.fft2(echo_noise_feat)
        x_image_noise = torch.fft.ifft2(echo_noise_freq * phi_adjoint)

        return x_image_noise


