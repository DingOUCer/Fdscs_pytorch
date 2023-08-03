import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines
import torchvision.transforms as transforms

def get_transforms(image,scale = None, disp=False):
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # return transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std),
    # ])
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    return  image_tensor * scale / 256.0  if disp else image_tensor/ 255.0



# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines

class KITTIDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        # data = np.array(data, dtype=np.float32) / 256.
        return data
    def load_disp2(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))


        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None
        """gwcnet"""
        # w, h = left_img.size
        # crop_w, crop_h = 512, 256
        #
        # x1 = random.randint(0, w - crop_w)
        # y1 = random.randint(0, h - crop_h)
        #
        # # random crop
        # left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        # right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        # disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
        #
        # # to tensor, normalize
        # processed = get_transform()
        # left_img = processed(left_img)
        # right_img = processed(right_img)
        """fdscs"""
        w, h = 1216, 352
        # define scale factor
        scale = torch.FloatTensor(1).uniform_(1.0,1.5).item()
        yscale = torch.FloatTensor(1).uniform_(1.0,1.5).item()
        oshp = torch.tensor(left_img.size,dtype=torch.float32) #tensor[1413,391]
        oshp = torch.stack([yscale * oshp[0],scale * oshp[1]]).to(torch.int32)

        # resize
        left_img = left_img.resize(oshp.tolist(),Image.BILINEAR) # 调整大小，线性插值
        right_img = right_img.resize(oshp.tolist(),Image.BILINEAR)
        disparity = disparity.resize((oshp[0].item(),oshp[1].item()),Image.NEAREST) # 最近邻插值
        # crop
        imshop = oshp # tensor[1413,391]
        yc = random.randint(0,imshop[1] - h + 1) # 0~391-370+1
        xc = random.randint(0,imshop[0] - w + 1) # 0~1413-1224+1
        # crop
        left_img = left_img.crop((xc,yc,xc + w,yc + h))
        right_img = right_img.crop((xc,yc,xc + w,yc + h))
        # disparity = disparity.crop((xc,yc,xc + w,yc + h))
        disparity = disparity.crop((xc,yc,xc + w,yc + h))

        left_img = get_transforms(left_img)
        right_img = get_transforms(right_img)
        disparity = get_transforms(disparity,scale = scale, disp=True)


        return {"left": left_img,
                "right": right_img,
                "disparity": disparity}
        # else:
        #     disparity = self.load_disp2(os.path.join(self.datapath, self.disp_filenames[index]))
        #     w, h = left_img.size
        #
        #     # normalize
        #     processed = get_transform()
        #     left_img = processed(left_img).numpy()
        #     right_img = processed(right_img).numpy()
        #
        #     # pad to size 1248x384
        #     top_pad = 384 - h
        #     right_pad = 1248 - w
        #     assert top_pad > 0 and right_pad > 0
        #     # pad images
        #     left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        #     right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
        #                            constant_values=0)
        #     # pad disparity gt
        #     if disparity is not None:
        #         assert len(disparity.shape) == 2
        #         disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        #         ##
        #         # disparity = get_transform(disparity, disp=True)
        #
        #     if disparity is not None:
        #         return {"left": left_img,
        #                 "right": right_img,
        #                 "disparity": disparity,
        #                 "top_pad": top_pad,
        #                 "right_pad": right_pad}
        #     else:
        #         return {"left": left_img,
        #                 "right": right_img,
        #                 "top_pad": top_pad,
        #                 "right_pad": right_pad,
        #                 "left_filename": self.left_filenames[index],
        #                 "right_filename": self.right_filenames[index]}
