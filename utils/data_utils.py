"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
import numpy as np
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def cuda_unsqueeze(li_variables=None, device='cuda'):

    if li_variables is None:
        return None

    cuda_variables = []

    for var in li_variables:
        if not var is None:
            var = var.to(device).unsqueeze(0)
        cuda_variables.append(var)

    return cuda_variables


def convert_npy_code(latent):

    if latent.shape == (18, 512):
        latent = np.reshape(latent, (1, 18, 512))

    if latent.shape == (512,) or latent.shape == (1, 512):
        latent = np.reshape(latent, (1, 1, 512)).repeat(18, axis=1)
    return latent



def load_FS_latent(latent_path, device):
    dict = np.load(latent_path)
    latent_in = torch.from_numpy(dict['latent_in']).to(device)
    latent_F = torch.from_numpy(dict['latent_F']).to(device)

    return latent_in, latent_F

