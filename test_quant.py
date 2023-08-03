from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
# from models import __models__
from model_quant import FDSCS
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
"""
use the weight which is quantized to test
"""
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='FDSCS')
parser.add_argument('--maxdisp', type=int, default=128, help='maximum disparity')
parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/data1/dataset/KITTI/kitti_2012/', help='data path')
parser.add_argument('--testlist', default='./filenames/kitti12_test.txt', help='testing list')
parser.add_argument('--loadckpt', default='./quant_pth_test/fdscs_quant_weight_post_epoch2.pth',required=False, help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
# model = __models__[args.model](args.maxdisp)
model = FDSCS(args.maxdisp)
model.cuda()


# load parameters
print("loading model {}".format(args.loadckpt))
model = torch.jit.load(args.loadckpt)

# 训练一个浮点数模型


def test():
    os.makedirs('./predictions', exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            fn = os.path.join("predictions", fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            skimage.io.imsave(fn, disp_est_uint)


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]
    # disp_ests = torch.squeeze(disp_ests, 1) #[B,H,W]


if __name__ == '__main__':
    test()
