"""
run this code to qunatize the model
"""

from __future__ import print_function, division
import argparse
import torchvision.transforms as transforms
import skimage
import torch.backends.cudnn as cudnn
import time
from datasets import __datasets__
from model_quant import FDSCS
from utils import *
from PIL import Image
from torch.utils.data import DataLoader
cudnn.benchmark = True
from torch.quantization import get_default_qconfig, QConfig
import torch.nn.quantized
from torch.ao.quantization import (
    default_weight_observer,
    default_per_channel_weight_observer,
    get_default_qconfig_mapping,
    MinMaxObserver,
    QConfig,
    QConfigMapping,
)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='FDSCS')
parser.add_argument('--maxdisp', type=int, default=128, help='maximum disparity')
parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/data1/dataset/KITTI/kitti_2015/', help='data path')
parser.add_argument('--trainlist', default='./filenames/kitti15_train.txt', help='testing list')
parser.add_argument('--testlist', default='./filenames/kitti15_test.txt', help='testing list')
parser.add_argument('--loadckpt', default='./checkpoints/kitti/fdscs_quant500/checkpoint_004998.ckpt',required=False, help='load the weights from a specific checkpoint')
# parse arguments
args = parser.parse_args()
# dataset, dataloader
# StereoDataset = __datasets__[args.dataset]
# train_dataset = StereoDataset(args.datapath, args.trainlist, True)
# test_dataset = StereoDataset(args.datapath, args.testlist, False)
# TrainImgLoader = DataLoader(train_dataset, 1, shuffle=True, num_workers=4, drop_last=True)
# TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

Fdscs_model = FDSCS(args.maxdisp)

def save_params(model_int8, epoch):
    torch.jit.save(torch.jit.script(model_int8.eval()),
                   "quant_pth_test/fdscs_quant_weight_post_epoch{}.pth".format(epoch + 1))
    checkpoint_data = {'epoch': epoch+1, 'model': model_int8.state_dict()}
    torch.save(checkpoint_data, "quant_pth_test/fdscs_quant_weight_save_epoch{}.pth".format(epoch + 1))
    print("#######################量化模型保存成功！##########################")
    for k, v in model_int8.state_dict().items():
        # 打印量化结果，遍历模型的状态字典，保存权重相关的量化参数（如缩放因子、零点等）和权重的浮点表示形式
        if 'weight' in k:  # 权重相关的量化参数
            np.save('./para_test/' + k + '.scale', v.q_per_channel_scales())  # 保存缩放因子，不同通道的缩放因子不同
            np.save('./para_test/' + k + '.zero_point', v.q_per_channel_zero_points())  # 保存零点，不同通道的权重在量化过程中的零偏移量不同
            np.save('./para_test/' + k + '.int', v.int_repr())  # 保存权重的整数表示形式，将浮点权重乘以缩放因子，然后加上零点，再四舍五入取整
            np.save('./para_test/' + k, v.dequantize().numpy())  # 保存权重的浮点表示形式，还原量化后的权重
        elif 'bias' in k:  # 偏置相关的量化参数
            if v == None:
                continue
            np.save('./para_test/' + k, v.detach().numpy())
        elif 'zero_point' in k:  # 零点相关的量化参数
            np.save('./para_test/' + k, v.detach().numpy())
        elif 'scale' in k:  # 缩放因子相关的量化参数-
            np.save('./para_test/' + k, v.detach().numpy())

def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def deal_img(imgL, imgR, return_pad=False):
    imgL = Image.open(imgL).convert('RGB')
    imgR = Image.open(imgR).convert('RGB')
    w,h = imgL.size
    processed = get_transform()
    imgL = processed(imgL).numpy()
    imgR = processed(imgR).numpy()
    # pad to width and hight
    top_pad = 384 - h
    right_pad = 1248 - w
    assert top_pad >= 0 and right_pad >= 0
    # pad image
    imgL = np.lib.pad(imgL, ((0, 0), (top_pad, 0), (right_pad, 0)), mode='constant', constant_values=0)
    imgR = np.lib.pad(imgR, ((0, 0), (top_pad, 0), (right_pad, 0)), mode='constant', constant_values=0)
    imgL = torch.from_numpy(imgL).float().unsqueeze(0)
    imgR = torch.from_numpy(imgR).float().unsqueeze(0)
    return (imgL, imgR, top_pad, right_pad) if return_pad else (imgL, imgR)


def feeddata(epoch,img_num):
    """
    epoch: 要投喂的数据的轮次
    img_num: 要投喂的图片数量
    """
    state_dict = torch.load(args.loadckpt)
    Fdscs_model.load_state_dict(state_dict['model'])
    Fdscs_model.eval().fuse_model()
    print(Fdscs_model)
    # 1
    # backend = "fbgemm"  # 量化操作符在x86机器上通过FBGEMM后端运行 部署在ARM上时，可以使用qnnpack后端
    # torch.backends.quantized.engine = backend
    # Fdscs_model.qconfig = torch.quantization.get_default_qconfig(backend)  # 不同平台不同配置
    # 2
    my_qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8),
        weight=default_per_channel_weight_observer.with_args(dtype=torch.qint8),
    )

    Fdscs_model.qconfig = my_qconfig
    Fdscs_model_fp32 = torch.quantization.prepare(Fdscs_model, inplace=True)

    # 测试数据
    left_img_path = "testing/image_2/"
    left_img_list = os.listdir(left_img_path)
    right_img_path = "testing/image_3/"
    right_img_list = os.listdir(right_img_path)
    for i in range(epoch):
        for batch_idx in range(img_num):
            imgL, imgR = deal_img( left_img_path + left_img_list[batch_idx], right_img_path + right_img_list[batch_idx], return_pad=False)
            disp_ests = Fdscs_model_fp32(imgL,imgR)
            if(batch_idx + 1) == img_num:
                break
    model_int8 = torch.quantization.convert(Fdscs_model_fp32, inplace=True)
    save_params(model_int8, epoch)
    ## 原先的测试是在整个测试集上进行的，现在只在单张图片上进行测试
    # for batch_idx, sample in enumerate(TestImgLoader):
    #     with torch.no_grad():
    #         imgL, imgR = sample['left'], sample['right']
    #         start_time = time.time()
    #         disp_ests = model_int8(imgL, imgR)  # [B,1,H,W]
    #         disp_est  = torch.squeeze(disp_ests, 1)
    #         disp_est_np = tensor2numpy(disp_est)
    #         top_pad_np = tensor2numpy(sample["top_pad"])
    #         right_pad_np = tensor2numpy(sample["right_pad"])
    #         left_filenames = sample["left_filename"]
    #         print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
    #                                                 time.time() - start_time))
    #
    #         for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
    #             assert len(disp_est.shape) == 2
    #             disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
    #             fn = os.path.join("predictions", fn.split('/')[-1])
    #             print("saving to", fn, disp_est.shape)
    #             disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
    #             skimage.io.imsave(fn, disp_est_uint)
    for i in range(epoch):
        for batch_idx in range(img_num):
            imgL, imgR, top_pad, right_pad = deal_img(left_img_path + left_img_list[batch_idx], right_img_path + right_img_list[batch_idx],return_pad=True)
            with torch.no_grad():
                start_time = time.time()
                disp = model_int8(imgL, imgR)  # [B,1,H,W]
                disp = torch.squeeze(disp, 1)
                disp_np = tensor2numpy(disp)
                left_img_name = (left_img_path + left_img_list[batch_idx]).split('/')[-1].split('.')[0]
                print('time = {:3f}'.format(time.time() - start_time))
                for disp in disp_np:
                    assert len(disp.shape) == 2
                    disp = np.array(disp[top_pad:, :-right_pad], dtype=np.float32)
                    fn = os.path.join("test_disp", left_img_name + "disp" + ".png")
                    print("saving to", fn, disp.shape)
                    disp_uint = np.round(disp * 256).astype(np.uint16)
                    skimage.io.imsave(fn, disp_uint)



if __name__ == '__main__':
    feeddata(1,5)





























