import torch
import PIL
import os
from losses import masked_lpips

class BlendLossBuilder(torch.nn.Module):
    def __init__(self, opt):
        super(BlendLossBuilder, self).__init__()

        self.opt = opt
        self.parsed_loss = [[1.0, 'face'], [1.0, 'hair']]
        if opt.device == 'cuda':
            use_gpu = True
        else:
            use_gpu = False

        self.face_percept = masked_lpips.PerceptualLoss(
            model="net-lin", net="vgg", vgg_blocks=['1', '2', '3'], use_gpu=use_gpu
        )
        self.face_percept.eval()

        self.hair_percept = masked_lpips.PerceptualLoss(
            model="net-lin", net="vgg", vgg_blocks=['1', '2', '3'], use_gpu=use_gpu
        )
        self.hair_percept.eval()



    def _loss_face_percept(self, gen_im, ref_im, mask, **kwargs):

        return self.face_percept(gen_im, ref_im, mask=mask)

    def _loss_hair_percept(self, gen_im, ref_im, mask, **kwargs):

        return self.hair_percept(gen_im, ref_im, mask=mask)


    def forward(self, gen_im, im_1, im_3, mask_face, mask_hair):

        loss = 0
        loss_fun_dict = {
            'face': self._loss_face_percept,
            'hair': self._loss_hair_percept,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            if loss_type == 'face':
                var_dict = {
                    'gen_im': gen_im,
                    'ref_im': im_1,
                    'mask': mask_face
                }
            elif loss_type == 'hair':
                var_dict = {
                    'gen_im': gen_im,
                    'ref_im': im_3,
                    'mask': mask_hair
                }
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += weight*tmp_loss
        return loss, losses