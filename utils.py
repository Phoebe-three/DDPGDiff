import numpy as np
import torch
from scipy.constants import speed_of_light as c
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def sparse_sampling(echo, strategy='URS', sparsity_rate=0.25):
    """
    :param echo: full-data (Ny, Nx)
    :param strategy:  ['URS', 'UCS']
    :param sparsity_rate:  (0.0 ~ 1.0)
    :return: sparse_echo , mask
    """
    N_V, N_H = echo.shape

    sample_axis = 0

    if strategy == 'UCS':
        num_samples = int(N_H * sparsity_rate)
    else:
        num_samples = int(N_V * sparsity_rate)




    if strategy == 'URS':

        indices = np.linspace(0, N_V - 1, num_samples, dtype=int)

    elif strategy == 'UCS':

        sample_axis = 1
        indices = np.linspace(0, N_H - 1, num_samples, dtype=int)

    else:
        raise ValueError("strategy must be 'URS' or 'UCS' ")

    sparse_echo = np.zeros_like(echo)
    mask = np.zeros_like(echo, dtype=bool)

    if indices is not None and len(indices) > 0:

        max_idx = N_H - 1 if sample_axis == 1 else N_V - 1
        indices = np.clip(indices, 0, max_idx)

        if sample_axis == 0:

            sparse_echo[indices, :] = echo[indices, :]
            mask[indices, :] = True
        else:

            sparse_echo[:, indices] = echo[:, indices]
            mask[:, indices] = True

    return sparse_echo, mask


def preprocess_echo(echo):
    # Normalization
    magnitude = np.abs(echo)
    phase = np.angle(echo)


    max_mag = np.max(magnitude)
    min_mag = np.min(magnitude)


    eps = 1e-12
    normalized_magnitude = (magnitude - min_mag) / (max_mag - min_mag + eps)


    normalized_echo = normalized_magnitude * np.exp(1j * phase)

    return normalized_echo

def rma(sarData, dx, dy, nFFTspace, F0, imSize, R, db=40, show=True, title=None):
    '''
    :param sarData: complex signals
    :return: reconstructed image
    '''
    # computer phase factors
    phasee_forward, phaseFactor, _ = compute_rma_phase_factors((nFFTspace, nFFTspace), F0, R, dx=dx, dy=dy)
    if isinstance(phaseFactor, torch.Tensor):
        phaseFactor = phaseFactor.cpu().numpy()

    # padding
    yPointM, xPointM = sarData.shape
    if yPointM < nFFTspace or xPointM < nFFTspace:
        # X direction
        pad_x_left = (nFFTspace - xPointM) // 2
        pad_x_right = nFFTspace - xPointM - pad_x_left
        sar_padded = np.pad(sarData, ((0, 0), (pad_x_left, pad_x_right)), 'constant')
        # Y direction
        pad_y_top = (nFFTspace - yPointM) // 2
        pad_y_bottom = nFFTspace - yPointM - pad_y_top
        sar_padded = np.pad(sar_padded, ((pad_y_top, pad_y_bottom), (0, 0)), 'constant')
    else:
        sar_padded = sarData

    sar_fft = np.fft.fft2(sar_padded)
    sar_image = np.fft.ifft2(sar_fft * phaseFactor)

    # cropping
    yPointT, xPointT = sar_image.shape
    xRangeT = dx * (np.arange(xPointT) - (xPointT - 1) / 2)
    yRangeT = dy * (np.arange(yPointT) - (yPointT - 1) / 2)
    x_mask = (xRangeT >= -imSize / 2) & (xRangeT <= imSize / 2)
    y_mask = (yRangeT >= -imSize / 2) & (yRangeT <= imSize / 2)
    sar_cropped = sar_image[np.ix_(y_mask, x_mask)]

    # imaging
    image = np.abs(sar_cropped)
    # transform to db
    img_db = db * np.log10(image / np.max(image))
    img_db = np.flipud(np.clip(img_db, -db, 0))

    if show:
        plt.figure()
        plt.imshow(img_db, extent=[
                xRangeT[x_mask][0], xRangeT[x_mask][-1],
                yRangeT[y_mask][0], yRangeT[y_mask][-1] ],
            cmap='jet', vmin=-db, vmax=0)
        plt.colorbar(label='dB')
        plt.xlabel('Horizontal (mm)')
        plt.ylabel('Vertical (mm)')
        plt.axis('equal')
        plt.show()

    return sar_image


def compute_rma_phase_factors(img_size, f0, R0, dx=1, dy=2):

    # wave number
    k0 = 2 * np.pi * f0 / c

    dx_m = dx * 1e-3  # mm to m
    dy_m = dy * 1e-3  # mm to m

    kx = np.fft.fftshift(np.fft.fftfreq(img_size[1], dx_m)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(img_size[0], dy_m)) * 2 * np.pi

    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='xy')

    kz_square = (2 * k0) ** 2 - kx_grid ** 2 - ky_grid ** 2

    kz = np.sqrt(np.maximum(kz_square, 0))

    valid_mask = (kx_grid ** 2 + ky_grid ** 2) <= (2 * k0) ** 2

    kz[~valid_mask] = 0
    # from echo to image
    phi_adjoint = np.exp(-1j * kz * R0)
    phi_adjoint[~valid_mask] = 0

    # from image to echo
    phi_forward = np.conj(phi_adjoint)

    phi_adjoint = np.fft.ifftshift(phi_adjoint)
    phi_forward = np.fft.ifftshift(phi_forward)

    return phi_forward, phi_adjoint, valid_mask