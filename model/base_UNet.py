import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from scipy.constants import speed_of_light as c

class ComplexConv2d(nn.Module):
    """Complex 2D Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        real = self.conv_real(x.real) - self.conv_imag(x.imag)
        imag = self.conv_real(x.imag) + self.conv_imag(x.real)
        return torch.complex(real, imag)


class ComplexBatchNorm2d(nn.Module):
    """Complex 2D BatchNorm"""

    def __init__(self, num_features, eps=1e-5):
        super(ComplexBatchNorm2d, self).__init__()
        self.bn_real = nn.BatchNorm2d(num_features, eps=eps)
        self.bn_imag = nn.BatchNorm2d(num_features, eps=eps)

    def forward(self, x):
        real = self.bn_real(x.real)
        imag = self.bn_imag(x.imag)
        return torch.complex(real, imag)


class ComplexLeakyReLU(nn.Module):
    """Complex LeakyReLU """

    def __init__(self, negative_slope=0.01):
        super(ComplexLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        real = F.leaky_relu(x.real, self.negative_slope)
        imag = F.leaky_relu(x.imag, self.negative_slope)
        return torch.complex(real, imag)


class ComplexUpsample(nn.Module):
    """Complex Upsample"""

    def __init__(self, size=None, scale_factor=None, mode='bilinear'):
        super(ComplexUpsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        real = F.interpolate(x.real, size=self.size, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=False)
        imag = F.interpolate(x.imag, size=self.size, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=False)
        return torch.complex(real, imag)


class ComplexMaxPool2d(nn.Module):
    """Complex MaxPool"""

    def __init__(self, kernel_size, stride=None, padding=0):
        super(ComplexMaxPool2d, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding, return_indices=False)

    def forward(self, x):
        real = self.pool(x.real)
        imag = self.pool(x.imag)
        return torch.complex(real, imag)


class ComplexConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ComplexConvBlock, self).__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.relu1 = ComplexLeakyReLU()

        self.conv2 = ComplexConv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = ComplexBatchNorm2d(out_channels)
        self.relu2 = ComplexLeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

def compute_phase_factors(F0, z0, dx, dy, nFFTspace):
    """compute phase factors"""
    k = 2 * np.pi * F0 / c
    z0 = z0

    dx_m = dx*1e-3
    dy_m = dy*1e-3

    # wave number axis
    kx = np.fft.fftshift(np.fft.fftfreq(nFFTspace, dx_m)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(nFFTspace, dy_m)) * 2 * np.pi

    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='xy')

    # kz
    kz = (2 * k) ** 2 - kx_grid ** 2 - ky_grid ** 2
    kz_sq = np.maximum(kz, 0)
    kz = np.sqrt(kz_sq)

    valid_mask = (kx_grid ** 2 + ky_grid ** 2) <= (2 * k) ** 2
    kz[~valid_mask] = 0

    # adjoint factor
    phi_adjoint = np.exp(-1j * kz * z0)
    phi_adjoint[~valid_mask] = 0

    # forward factor
    phi_forward = np.conj(phi_adjoint)

    phi_adjoint = np.fft.ifftshift(phi_adjoint)
    phi_forward = np.fft.ifftshift(phi_forward)

    return (torch.tensor(phi_forward, dtype=torch.complex64),
            torch.tensor(phi_adjoint, dtype=torch.complex64))

def observation_operator(image, sparse_echo_tensor, nFFTspace, phi_forward):
    """Projecting the image back to the echo"""
    batch_size, channels, xPointM, yPointM = sparse_echo_tensor.shape

    # padding
    pad_x_left = (nFFTspace - xPointM) // 2
    pad_y_top = (nFFTspace - yPointM) // 2

    image_fft = torch.fft.fft2(image)
    echo_sim_fft = image_fft * phi_forward.to(image.device)
    echo_sim = torch.fft.ifft2(echo_sim_fft)

    # cropping
    echo_sim_cropped = echo_sim[:, :, pad_x_left:pad_x_left + xPointM, pad_y_top:pad_y_top + yPointM]

    return echo_sim_cropped

