import torch
import torch.nn as nn
from torch.nn import functional as F

mse_loss = nn.MSELoss(reduction="mean")


def custom_loss(x, y, mask=None, loss_type="l2", include_bkgd=True):
    """
    x, y: [N, C, H, W]
    Computes L1/L2 loss

    if include_bkgd is True:
        use traditional MSE and L1 loss
    else:
        mask out background info using :mask
        normalize loss with #1's in mask
    """
    if include_bkgd:
        # perform simple mse or l1 loss
        if loss_type == "l2":
            loss_rec = mse_loss(x, y)
        elif loss_type == "l1":
            loss_rec = F.l1_loss(x, y)

        return loss_rec

    Nx, Cx, Hx, Wx = x.shape
    Nm, Cm, Hm, Wm = mask.shape
    mask = prepare_mask(x, mask)

    x_reshape = torch.reshape(x, [Nx, -1])
    y_reshape = torch.reshape(y, [Nx, -1])
    mask_reshape = torch.reshape(mask, [Nx, -1])

    if loss_type == "l2":
        diff = (x_reshape - y_reshape) ** 2
    elif loss_type == "l1":
        diff = torch.abs(x_reshape - y_reshape)

    # diff: [N, Cx * Hx * Wx]
    # set elements in diff to 0 using mask
    masked_diff = diff * mask_reshape
    sum_diff = torch.sum(masked_diff, axis=-1)
    # count non-zero elements; add :mask_reshape elements
    norm_count = torch.sum(mask_reshape, axis=-1)
    diff_norm = sum_diff / (norm_count + 1.0)

    loss_rec = torch.mean(diff_norm)

    return loss_rec


def prepare_mask(x, mask):
    """
    Make mask similar to x.
    Mask contains values in [0, 1].
    Adjust channels and spatial dimensions.
    """
    Nx, Cx, Hx, Wx = x.shape
    Nm, Cm, Hm, Wm = mask.shape
    if Cm == 1:
        mask = mask.repeat(1, Cx, 1, 1)

    mask = F.interpolate(mask, scale_factor=Hx / Hm, mode="nearest")

    return mask
