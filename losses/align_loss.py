import torch
from losses.style.style_loss import StyleLoss

class AlignLossBuilder(torch.nn.Module):
    def __init__(self, opt):
        super(AlignLossBuilder, self).__init__()

        self.opt = opt
        self.parsed_loss = [[opt.l2_lambda, 'l2'], [opt.percept_lambda, 'percep']]
        if opt.device == 'cuda':
            use_gpu = True
        else:
            use_gpu = False

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.style = StyleLoss(distance="l2", VGG16_ACTIVATIONS_LIST=[3, 8, 15, 22], normalize=False).to(opt.device)
        self.style.eval()


        tmp = torch.zeros(16).to(opt.device)
        tmp[0] = 1
        self.cross_entropy_wo_background = torch.nn.CrossEntropyLoss(weight=1 - tmp)
        self.cross_entropy_only_background = torch.nn.CrossEntropyLoss(weight=tmp)



    def cross_entropy_loss(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy(down_seg, target_mask)
        return loss


    def style_loss(self, im1, im2, mask1, mask2):
        loss = self.opt.style_lambda * self.style(im1 * mask1, im2 * mask2, mask1=mask1, mask2=mask2)
        return loss


    def cross_entropy_loss_wo_background(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy_wo_background(down_seg, target_mask)
        return loss

    def cross_entropy_loss_only_background(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy_only_background(down_seg, target_mask)
        return loss