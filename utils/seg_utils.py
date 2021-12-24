
import numpy as np
import os
import PIL
def vis_seg(pred):
    num_labels = 16

    color = np.array([[0, 0, 0],  ## 0
                      [102, 204, 255],  ## 1
                      [255, 204, 255],  ## 2
                      [255, 255, 153],  ## 3
                      [255, 255, 153],  ## 4
                      [255, 255, 102],  ## 5
                      [51, 255, 51],  ## 6
                      [0, 153, 255],  ## 7
                      [0, 255, 255],  ## 8
                      [0, 255, 255],  ## 9
                      [204, 102, 255],  ## 10
                      [0, 153, 255],  ## 11
                      [0, 255, 153],  ## 12
                      [0, 51, 0],
                      [102, 153, 255],  ## 14
                      [255, 153, 102],  ## 15
                      ])
    h, w = np.shape(pred)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     print(color.shape)
    for ii in range(num_labels):
        #         print(ii)
        mask = pred == ii
        rgb[mask, None] = color[ii, :]
    # Correct unk
    unk = pred == 255
    rgb[unk, None] = color[0, :]
    return rgb


def save_vis_mask(img_path1, img_path2, sign, output_dir, mask):
    im_name_1 = os.path.splitext(os.path.basename(img_path1))[0]
    im_name_2 = os.path.splitext(os.path.basename(img_path2))[0]
    vis_path = os.path.join(output_dir, 'vis_mask_{}_{}_{}.png'.format(im_name_1, im_name_2, sign))
    vis_mask = vis_seg(mask)
    PIL.Image.fromarray(vis_mask).save(vis_path)
