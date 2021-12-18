import torch
import torch.nn as nn
from torch.nn import functional as F

import os

from losses.style.custom_loss import custom_loss, prepare_mask
from losses.style.vgg_activations import VGG16_Activations, VGG19_Activations, Vgg_face_dag


class StyleLoss(nn.Module):
    def __init__(self, VGG16_ACTIVATIONS_LIST=[21], normalize=False, distance="l2"):

        super(StyleLoss, self).__init__()

        self.vgg16_act = VGG16_Activations(VGG16_ACTIVATIONS_LIST)
        self.vgg16_act.eval()

        self.normalize = normalize
        self.distance = distance

    def get_features(self, model, x):

        return model(x)

    def mask_features(self, x, mask):

        mask = prepare_mask(x, mask)
        return x * mask

    def gram_matrix(self, x):
        """
        :x is an activation tensor
        """
        N, C, H, W = x.shape
        x = x.view(N * C, H * W)
        G = torch.mm(x, x.t())

        return G.div(N * H * W * C)

    def cal_style(self, model, x, x_hat, mask1=None, mask2=None):
        # Get features from the model for x and x_hat
        with torch.no_grad():
            act_x = self.get_features(model, x)
        for layer in range(0, len(act_x)):
            act_x[layer].detach_()

        act_x_hat = self.get_features(model, x_hat)

        loss = 0.0
        for layer in range(0, len(act_x)):

            # mask features if present
            if mask1 is not None:
                feat_x = self.mask_features(act_x[layer], mask1)
            else:
                feat_x = act_x[layer]
            if mask2 is not None:
                feat_x_hat = self.mask_features(act_x_hat[layer], mask2)
            else:
                feat_x_hat = act_x_hat[layer]

            """
            import ipdb; ipdb.set_trace()
            fx = feat_x[0, ...].detach().cpu().numpy()
            fx = (fx - fx.min()) / (fx.max() - fx.min())
            fx = fx * 255.
            fxhat = feat_x_hat[0, ...].detach().cpu().numpy()
            fxhat = (fxhat - fxhat.min()) / (fxhat.max() - fxhat.min())
            fxhat = fxhat * 255
            from PIL import Image
            import numpy as np
            for idx, img in enumerate(fx):
                img = fx[idx, ...]
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img.save('plot/feat_x/{}.png'.format(str(idx)))
                img = fxhat[idx, ...]
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img.save('plot/feat_x_hat/{}.png'.format(str(idx)))
            import ipdb; ipdb.set_trace()
            """

            # compute Gram matrix for x and x_hat
            G_x = self.gram_matrix(feat_x)
            G_x_hat = self.gram_matrix(feat_x_hat)

            # compute layer wise loss and aggregate
            loss += custom_loss(
                G_x, G_x_hat, mask=None, loss_type=self.distance, include_bkgd=True
            )

        loss = loss / len(act_x)

        return loss

    def forward(self, x, x_hat, mask1=None, mask2=None):
        x = x.cuda()
        x_hat = x_hat.cuda()

        # resize images to 256px resolution
        N, C, H, W = x.shape
        upsample2d = nn.Upsample(
            scale_factor=256 / H, mode="bilinear", align_corners=True
        )

        x = upsample2d(x)
        x_hat = upsample2d(x_hat)

        loss = self.cal_style(self.vgg16_act, x, x_hat, mask1=mask1, mask2=mask2)

        return loss
