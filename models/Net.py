import torch
from torch import nn
from models.stylegan2.model import Generator
import numpy as np
import os
from utils.model_utils import download_weight

class Net(nn.Module):

    def __init__(self, opts):
        super(Net, self).__init__()
        self.opts = opts
        self.generator = Generator(opts.size, opts.latent, opts.n_mlp, channel_multiplier=opts.channel_multiplier)
        self.cal_layer_num()
        self.load_weights()
        self.load_PCA_model()


    def load_weights(self):
        if not os.path.exists(self.opts.ckpt):
            print('Downloading StyleGAN2 checkpoint: {}'.format(self.opts.ckpt))
            download_weight(self.opts.ckpt)

        print('Loading StyleGAN2 from checkpoint: {}'.format(self.opts.ckpt))
        checkpoint = torch.load(self.opts.ckpt)
        device = self.opts.device
        self.generator.load_state_dict(checkpoint['g_ema'])
        self.latent_avg = checkpoint['latent_avg']
        self.generator.to(device)
        self.latent_avg = self.latent_avg.to(device)

        for param in self.generator.parameters():
            param.requires_grad = False
        self.generator.eval()


    def build_PCA_model(self, PCA_path):

        with torch.no_grad():
            latent = torch.randn((1000000, 512), dtype=torch.float32)
            # latent = torch.randn((10000, 512), dtype=torch.float32)
            self.generator.style.cpu()
            pulse_space = torch.nn.LeakyReLU(5)(self.generator.style(latent)).numpy()
            self.generator.style.to(self.opts.device)

        from utils.PCA_utils import IPCAEstimator

        transformer = IPCAEstimator(512)
        X_mean = pulse_space.mean(0)
        transformer.fit(pulse_space - X_mean)
        X_comp, X_stdev, X_var_ratio = transformer.get_components()
        np.savez(PCA_path, X_mean=X_mean, X_comp=X_comp, X_stdev=X_stdev, X_var_ratio=X_var_ratio)


    def load_PCA_model(self):
        device = self.opts.device

        PCA_path = self.opts.ckpt[:-3] + '_PCA.npz'

        if not os.path.isfile(PCA_path):
            self.build_PCA_model(PCA_path)

        PCA_model = np.load(PCA_path)
        self.X_mean = torch.from_numpy(PCA_model['X_mean']).float().to(device)
        self.X_comp = torch.from_numpy(PCA_model['X_comp']).float().to(device)
        self.X_stdev = torch.from_numpy(PCA_model['X_stdev']).float().to(device)



    # def make_noise(self):
    #     noises_single = self.generator.make_noise()
    #     noises = []
    #     for noise in noises_single:
    #         noises.append(noise.repeat(1, 1, 1, 1).normal_())
    #
    #     return noises

    def cal_layer_num(self):
        if self.opts.size == 1024:
            self.layer_num = 18
        elif self.opts.size == 512:
            self.layer_num = 16
        elif self.opts.size == 256:
            self.layer_num = 14

        self.S_index = self.layer_num - 11

        return


    def cal_p_norm_loss(self, latent_in):
        latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean).bmm(
            self.X_comp.T.unsqueeze(0)) / self.X_stdev
        p_norm_loss = self.opts.p_norm_lambda * (latent_p_norm.pow(2).mean())
        return p_norm_loss


    def cal_l_F(self, latent_F, F_init):
        return self.opts.l_F_lambda * (latent_F - F_init).pow(2).mean()


