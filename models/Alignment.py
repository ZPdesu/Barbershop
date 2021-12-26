import torch
from torch import nn
from models.Net import Net
import numpy as np
import os
from functools import partial
from utils.bicubic import BicubicDownSample
from tqdm import tqdm
import PIL
import torchvision
from PIL import Image
from utils.data_utils import convert_npy_code
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from losses.align_loss import AlignLossBuilder
import torch.nn.functional as F
import cv2
from utils.data_utils import load_FS_latent
from utils.seg_utils import save_vis_mask
from utils.model_utils import download_weight
from utils.data_utils import cuda_unsqueeze
from utils.image_utils import dilate_erosion_mask_tensor

toPIL = torchvision.transforms.ToPILImage()


class Alignment(nn.Module):

    def __init__(self, opts, net=None):
        super(Alignment, self).__init__()
        self.opts = opts
        if not net:
            self.net = Net(self.opts)
        else:
            self.net = net

        self.load_segmentation_network()
        self.load_downsampling()
        self.setup_align_loss_builder()

    def load_segmentation_network(self):
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.opts.device)

        if not os.path.exists(self.opts.seg_ckpt):
            download_weight(self.opts.seg_ckpt)
        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

    def load_downsampling(self):

        self.downsample = BicubicDownSample(factor=self.opts.size // 512)
        self.downsample_256 = BicubicDownSample(factor=self.opts.size // 256)

    def setup_align_loss_builder(self):
        self.loss_builder = AlignLossBuilder(self.opts)

    def create_target_segmentation_mask(self, img_path1, img_path2, sign, save_intermediate=True):

        device = self.opts.device

        im1 = self.preprocess_img(img_path1)
        down_seg1, _, _ = self.seg(im1)
        seg_target1 = torch.argmax(down_seg1, dim=1).long()

        ggg = torch.where(seg_target1 == 0, torch.zeros_like(seg_target1), torch.ones_like(seg_target1))


        hair_mask1 = torch.where(seg_target1 == 10, torch.ones_like(seg_target1), torch.zeros_like(seg_target1))
        seg_target1 = seg_target1[0].byte().cpu().detach()
        seg_target1 = torch.where(seg_target1 == 10, torch.zeros_like(seg_target1), seg_target1)

        im2 = self.preprocess_img(img_path2)
        down_seg2, _, _ = self.seg(im2)
        seg_target2 = torch.argmax(down_seg2, dim=1).long()

        ggg = torch.where(seg_target2 == 10, torch.ones_like(seg_target2), ggg)

        hair_mask2 = torch.where(seg_target2 == 10, torch.ones_like(seg_target2), torch.zeros_like(seg_target2))
        seg_target2 = seg_target2[0].byte().cpu().detach()


        OB_region = torch.where(
            (seg_target2 != 10) * (seg_target2 != 0) * (seg_target2 != 15) * (
                    seg_target1 == 0),
            255 * torch.ones_like(seg_target1), torch.zeros_like(seg_target1))


        new_target = torch.where(seg_target2 == 10, 10 * torch.ones_like(seg_target1), seg_target1)

        inpainting_region = torch.where((new_target != 0) * (new_target != 10), 255 * torch.ones_like(new_target),
                                        OB_region).numpy()
        tmp = torch.where(torch.from_numpy(inpainting_region) == 255, torch.zeros_like(new_target), new_target) / 10
        new_target_inpainted = (
                    cv2.inpaint(tmp.clone().numpy(), inpainting_region, 3, cv2.INPAINT_NS).astype(np.uint8) * 10)
        new_target_final = torch.where(OB_region, torch.from_numpy(new_target_inpainted), new_target)
        # new_target_final = new_target
        target_mask = new_target_final.unsqueeze(0).long().cuda()



        ############################# add auto-inpainting


        optimizer_align, latent_align = self.setup_align_optimizer()
        latent_end = latent_align[:, 6:, :].clone().detach()

        pbar = tqdm(range(80), desc='Create Target Mask Step1', leave=False)
        for step in pbar:
            optimizer_align.zero_grad()
            latent_in = torch.cat([latent_align[:, :6, :], latent_end], dim=1)
            down_seg, _ = self.create_down_seg(latent_in)

            loss_dict = {}

            if sign == 'realistic':
                ce_loss = self.loss_builder.cross_entropy_loss_wo_background(down_seg, target_mask)
                ce_loss += self.loss_builder.cross_entropy_loss_only_background(down_seg, ggg)
            else:
                ce_loss = self.loss_builder.cross_entropy_loss(down_seg, target_mask)


            loss_dict["ce_loss"] = ce_loss.item()
            loss = ce_loss


            loss.backward()
            optimizer_align.step()


        gen_seg_target = torch.argmax(down_seg, dim=1).long()
        free_mask = hair_mask1 * (1 - hair_mask2)
        target_mask = torch.where(free_mask==1, gen_seg_target, target_mask)
        previouse_target_mask = target_mask.clone().detach()

        ############################################

        target_mask = torch.where(OB_region.to(device).unsqueeze(0), torch.zeros_like(target_mask), target_mask)
        optimizer_align, latent_align = self.setup_align_optimizer()
        latent_end = latent_align[:, 6:, :].clone().detach()

        pbar = tqdm(range(80), desc='Create Target Mask Step2', leave=False)
        for step in pbar:
            optimizer_align.zero_grad()
            latent_in = torch.cat([latent_align[:, :6, :], latent_end], dim=1)
            down_seg, _ = self.create_down_seg(latent_in)

            loss_dict = {}

            if sign == 'realistic':
                ce_loss = self.loss_builder.cross_entropy_loss_wo_background(down_seg, target_mask)
                ce_loss += self.loss_builder.cross_entropy_loss_only_background(down_seg, ggg)
            else:
                ce_loss = self.loss_builder.cross_entropy_loss(down_seg, target_mask)

            loss_dict["ce_loss"] = ce_loss.item()
            loss = ce_loss

            loss.backward()
            optimizer_align.step()


        gen_seg_target = torch.argmax(down_seg, dim=1).long()
        # free_mask = hair_mask1 * (1 - hair_mask2)
        # target_mask = torch.where((free_mask == 1) * (gen_seg_target!=0), gen_seg_target, previouse_target_mask)
        target_mask = torch.where((OB_region.to(device).unsqueeze(0)) * (gen_seg_target != 0), gen_seg_target, previouse_target_mask)

        #####################  Save Visualization of Target Segmentation Mask
        if save_intermediate:
            save_vis_mask(img_path1, img_path2, sign, self.opts.output_dir, target_mask.squeeze().cpu())

        hair_mask_target = torch.where(target_mask == 10, torch.ones_like(target_mask), torch.zeros_like(target_mask))
        hair_mask_target = F.interpolate(hair_mask_target.float().unsqueeze(0), size=(512, 512), mode='nearest')

        return target_mask, hair_mask_target, hair_mask1, hair_mask2


    def preprocess_img(self, img_path):
        im = torchvision.transforms.ToTensor()(Image.open(img_path))[:3].unsqueeze(0).to(self.opts.device)
        im = (self.downsample(im).clamp(0, 1) - seg_mean) / seg_std
        return im

    def setup_align_optimizer(self, latent_path=None):
        if latent_path:
            latent_W = torch.from_numpy(convert_npy_code(np.load(latent_path))).to(self.opts.device).requires_grad_(True)
        else:
            latent_W = self.net.latent_avg.reshape(1, 1, 512).repeat(1, 18, 1).clone().detach().to(self.opts.device).requires_grad_(True)



        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }

        optimizer_align = opt_dict[self.opts.opt_name]([latent_W], lr=self.opts.learning_rate)

        return optimizer_align, latent_W



    def create_down_seg(self, latent_in):
        gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                       start_layer=0, end_layer=8)
        gen_im_0_1 = (gen_im + 1) / 2

        # get hair mask of synthesized image
        im = (self.downsample(gen_im_0_1) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(im)
        return down_seg, gen_im


    def dilate_erosion(self, free_mask, device, dilate_erosion=5):
        free_mask = F.interpolate(free_mask.cpu(), size=(256, 256), mode='nearest').squeeze()
        free_mask_D, free_mask_E = cuda_unsqueeze(dilate_erosion_mask_tensor(free_mask, dilate_erosion=dilate_erosion), device)
        return free_mask_D, free_mask_E

    def align_images(self, img_path1, img_path2, sign='realistic', align_more_region=False, smooth=5,
                     save_intermediate=True):

        ################## img_path1: Identity Image
        ################## img_path2: Structure Image

        device = self.opts.device
        output_dir = self.opts.output_dir
        target_mask, hair_mask_target, hair_mask1, hair_mask2 = \
            self.create_target_segmentation_mask(img_path1=img_path1, img_path2=img_path2, sign=sign,
                                                 save_intermediate=save_intermediate)

        im_name_1 = os.path.splitext(os.path.basename(img_path1))[0]
        im_name_2 = os.path.splitext(os.path.basename(img_path2))[0]

        latent_FS_path_1 = os.path.join(output_dir, 'FS', f'{im_name_1}.npz')
        latent_FS_path_2 = os.path.join(output_dir, 'FS', f'{im_name_2}.npz')

        latent_1, latent_F_1 = load_FS_latent(latent_FS_path_1, device)
        latent_2, latent_F_2 = load_FS_latent(latent_FS_path_2, device)

        latent_W_path_1 = os.path.join(output_dir, 'W+', f'{im_name_1}.npy')
        latent_W_path_2 = os.path.join(output_dir, 'W+', f'{im_name_2}.npy')

        optimizer_align, latent_align_1 = self.setup_align_optimizer(latent_W_path_1)

        pbar = tqdm(range(self.opts.align_steps1), desc='Align Step 1', leave=False)
        for step in pbar:
            optimizer_align.zero_grad()
            latent_in = torch.cat([latent_align_1[:, :6, :], latent_1[:, 6:, :]], dim=1)
            down_seg, _ = self.create_down_seg(latent_in)

            loss_dict = {}
            ##### Cross Entropy Loss
            ce_loss = self.loss_builder.cross_entropy_loss(down_seg, target_mask)
            loss_dict["ce_loss"] = ce_loss.item()
            loss = ce_loss

            # best_summary = f'BEST ({j+1}) | ' + ' | '.join(
            #     [f'{x}: {y:.4f}' for x, y in loss_dict.items()])

            #### TODO not finished

            loss.backward()
            optimizer_align.step()

        intermediate_align, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                                   start_layer=0, end_layer=3)
        intermediate_align = intermediate_align.clone().detach()

        ##############################################

        optimizer_align, latent_align_2 = self.setup_align_optimizer(latent_W_path_2)

        with torch.no_grad():
            tmp_latent_in = torch.cat([latent_align_2[:, :6, :], latent_2[:, 6:, :]], dim=1)
            down_seg_tmp, I_Structure_Style_changed = self.create_down_seg(tmp_latent_in)

            current_mask_tmp = torch.argmax(down_seg_tmp, dim=1).long()
            HM_Structure = torch.where(current_mask_tmp == 10, torch.ones_like(current_mask_tmp),
                                       torch.zeros_like(current_mask_tmp))
            HM_Structure = F.interpolate(HM_Structure.float().unsqueeze(0), size=(256, 256), mode='nearest')

        pbar = tqdm(range(self.opts.align_steps2), desc='Align Step 2', leave=False)
        for step in pbar:
            optimizer_align.zero_grad()
            latent_in = torch.cat([latent_align_2[:, :6, :], latent_2[:, 6:, :]], dim=1)
            down_seg, gen_im = self.create_down_seg(latent_in)

            Current_Mask = torch.argmax(down_seg, dim=1).long()
            HM_G_512 = torch.where(Current_Mask == 10, torch.ones_like(Current_Mask),
                                   torch.zeros_like(Current_Mask)).float().unsqueeze(0)
            HM_G = F.interpolate(HM_G_512, size=(256, 256), mode='nearest')

            loss_dict = {}

            ########## Segmentation Loss
            ce_loss = self.loss_builder.cross_entropy_loss(down_seg, target_mask)
            loss_dict["ce_loss"] = ce_loss.item()
            loss = ce_loss

            #### Style Loss
            H1_region = self.downsample_256(I_Structure_Style_changed) * HM_Structure
            H2_region = self.downsample_256(gen_im) * HM_G
            style_loss = self.loss_builder.style_loss(H1_region, H2_region, mask1=HM_Structure, mask2=HM_G)

            loss_dict["style_loss"] = style_loss.item()
            loss += style_loss

            # best_summary = f'BEST ({j+1}) | ' + ' | '.join(
            #     [f'{x}: {y:.4f}' for x, y in loss_dict.items()])

            loss.backward()
            optimizer_align.step()

        latent_F_out_new, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                                 start_layer=0, end_layer=3)
        latent_F_out_new = latent_F_out_new.clone().detach()

        free_mask = 1 - (1 - hair_mask1.unsqueeze(0)) * (1 - hair_mask_target)

        ##############################
        free_mask, _ = self.dilate_erosion(free_mask, device, dilate_erosion=smooth)
        ##############################

        free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')[0]
        interpolation_low = 1 - free_mask_down_32


        latent_F_mixed = intermediate_align + interpolation_low.unsqueeze(0) * (
                latent_F_1 - intermediate_align)

        if not align_more_region:
            free_mask = hair_mask_target
            ##########################
            _, free_mask = self.dilate_erosion(free_mask, device, dilate_erosion=smooth)
            ##########################
            free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')[0]
            interpolation_low = 1 - free_mask_down_32


        latent_F_mixed = latent_F_out_new + interpolation_low.unsqueeze(0) * (
                latent_F_mixed - latent_F_out_new)

        free_mask = F.interpolate((hair_mask2.unsqueeze(0) * hair_mask_target).float(), size=(256, 256), mode='nearest').cuda()
        ##########################
        _, free_mask = self.dilate_erosion(free_mask, device, dilate_erosion=smooth)
        ##########################
        free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')[0]
        interpolation_low = 1 - free_mask_down_32

        latent_F_mixed = latent_F_2 + interpolation_low.unsqueeze(0) * (
                latent_F_mixed - latent_F_2)

        gen_im, _ = self.net.generator([latent_1], input_is_latent=True, return_latents=False, start_layer=4,
                                       end_layer=8, layer_in=latent_F_mixed)
        self.save_align_results(im_name_1, im_name_2, sign, gen_im, latent_1, latent_F_mixed,
                                save_intermediate=save_intermediate)

    def save_align_results(self, im_name_1, im_name_2, sign, gen_im, latent_in, latent_F, save_intermediate=True):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        save_dir = os.path.join(self.opts.output_dir, 'Align_{}'.format(sign))
        os.makedirs(save_dir, exist_ok=True)

        latent_path = os.path.join(save_dir, '{}_{}.npz'.format(im_name_1, im_name_2))
        if save_intermediate:
            image_path = os.path.join(save_dir, '{}_{}.png'.format(im_name_1, im_name_2))
            save_im.save(image_path)
        np.savez(latent_path, latent_in=latent_in.detach().cpu().numpy(), latent_F=latent_F.detach().cpu().numpy())
