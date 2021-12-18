from torch.utils.data import Dataset
from PIL import Image
import PIL
from utils import data_utils
import torchvision.transforms as transforms
import os

class ImagesDataset(Dataset):

    def __init__(self, opts, image_path=None):
        if not image_path:
            image_root = opts.input_dir
            self.image_paths = sorted(data_utils.make_dataset(image_root))
        elif type(image_path) == str:
            self.image_paths = [image_path]
        elif type(image_path) == list:
            self.image_paths = image_path

        self.image_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.opts = opts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        im_path = self.image_paths[index]
        im_H = Image.open(im_path).convert('RGB')
        im_L = im_H.resize((256, 256), PIL.Image.LANCZOS)
        im_name = os.path.splitext(os.path.basename(im_path))[0]
        if self.image_transform:
            im_H = self.image_transform(im_H)
            im_L = self.image_transform(im_L)

        return im_H, im_L, im_name



