# import os
# import cv2
# import glob
# import numpy as np
#
#
# label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip',
#               'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
#
#
# folder_base = '/media/zhup/Data/CelebAMask-HQ/CelebAMaskHQ-mask-anno'
# folder_save = '/media/zhup/Data/CelebAMask-HQ/Mask_less'
# img_num = 30000
#
# correspond_list = [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#
# for k in range(img_num):
#     folder_num = int(k / 2000)
#     im_base = np.zeros((512, 512))
#     for idx, label in enumerate(label_list, 1):
#         filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
#         if (os.path.exists(filename)):
#             # print(label, idx)
#             im = cv2.imread(filename)
#             im = im[:, :, 0]
#             im_base[im != 0] = (correspond_list[idx])
#
#     filename_save = os.path.join(folder_save, str(k) + '.png')
#     print(filename_save)
#     cv2.imwrite(filename_save, im_base)
#
