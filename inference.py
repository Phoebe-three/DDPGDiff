from model.diffusion_model import DDNPNet
import numpy as np
from scipy.constants import speed_of_light as c
import torch
from model.diffusion_model import ConditionalComplexDDPM
from utils import preprocess_echo, sparse_sampling, rma
from model.physical_modules import ERG_DDIM

# physical paramters
config = {
    'dx' : 1.2,
    'dy' : 2,
    'nFFTtime' : 512,
    'nFFTspace' : 512,
    'F0' : 77e9,
    'Fs' : 5.12 * 1e6,
    'Ts' : 1 / (5.12 * 1e6),
    'K' : 63.343e12,
    'tI' : 7e-10,
    'k' : 2 * np.pi * 77e9 / c,
    'num_sample' : 256,
    'imSize' : 400,
    'R': 300e-3
}

def inference(full_echo, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steps = 50
    project_beta = 0.25

    nFFTspace = config['nFFTspace']
    imSize = config['imSize']
    dx = config['dx']
    dy = config['dy']
    F0 = config['F0']
    R = config['R']
    nFFTtime = config['nFFTtime']
    K = config['K']
    fs = config['Fs']
    tI = config['tI']

    ddnpnet = DDNPNet(input_channels=1, output_channels=1, base_channels=32, time_dim=32, dx=dx, dy=dy, F0=F0, branch=True).to(device)
    ddpgdiff = ConditionalComplexDDPM(ddnpnet, timesteps=500).to(device)

    ddpgdiff.load_state_dict(torch.load(
        "pretrained.pth",
        map_location=device))
    ddpgdiff.eval()

    rma(full_echo, dx, dy, nFFTspace, F0, imSize, R, show=True)


    sparse_echo, mask = sparse_sampling(full_echo, strategy='URS', sparsity_rate=0.25)
    sparse_echo = preprocess_echo(sparse_echo)

    coarse_image = rma(sparse_echo, dx, dy, nFFTspace, F0, imSize, R, show=True)

    recon_image, recon_echo = ERG_DDIM(
        sparse_echo=sparse_echo,
        sparse_image=coarse_image,
        mask=mask,
        nFFTspace=nFFTspace,
        F0=F0,
        z0=R,
        dx=dx,
        dy=dy,
        pretrained_ddpm=ddpgdiff,
        ddim_steps=steps,
        projection_strength=project_beta
    )

    rma(recon_echo, dx, dy, nFFTspace, F0, imSize, R)

    return recon_image, recon_echo



