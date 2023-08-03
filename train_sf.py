from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
# from models import __models__, model_loss
from model import FDSCS
from utils import *
from utils.visualization import *
from torch.utils.data import DataLoader
import gc
import argparse
from torch.optim.lr_scheduler import MultiStepLR
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Fast Deep Stereo with 2D Convolutional Processing of Cost Signatures(FDSCS)')
parser.add_argument('--maxdisp', type=int, default=128, help='maximum disparity')
parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/data1/dataset/SceneFlow/SceneFlow/FlyingThings3D/', help='data path')
parser.add_argument('--trainlist', default='./filenames/sceneflow_train.txt', help='training list')
parser.add_argument('--testlist', default='./filenames/sceneflow_test.txt', help='testing list')
parser.add_argument('--lr', type=float, default=0.0001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
parser.add_argument('--epochs', type=int, default=80, help='number of epochs to train')
# train sceneflow
parser.add_argument('--lrepochs', type=str, default='5:0.1', help='the epochs to decay lr: the downscale rate')
parser.add_argument('--logdir', default='./checkpoints/sceneflow/fdscsdebug192', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default='',help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)
print(args)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = FDSCS(args.maxdisp)  # initialize model with
model = nn.DataParallel(model)
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),weight_decay=0.00001)



error_func = disp_error_image_func()
# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))


def model_loss(d_gt,d, mask):
    # d_gt , mask = d_gt.squeeze(-1), mask.squeeze(-1)
    loss = torch.pow(torch.clamp_min(torch.abs(d_gt - d), 1.0), 1/8)
    # loss = (loss * mask).sum()
    return loss


def train():
    # print_freq = 100 # print training status every print_freq iterations
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # print('Epoch {}: learning rate {}'.format(epoch_idx, scheduler.get_last_lr()[0]))
        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            # print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
            #                                                                            batch_idx,
            #                                                                            len(TrainImgLoader), loss,
            #                                                                            time.time() - start_time))
            # if (batch_idx + 1) % print_freq == 0:
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()

    imgL, imgR, disp_gt =  sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()

    disp_ests = model(imgL, imgR) #[B,1,H,W]
    disp_ests = torch.squeeze(disp_ests, 1) #[B,H,W]
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    # mask = (disp_gt > 0).float() * (disp_gt < 255.0).float()
    # loss = model_loss(disp_gt, disp_ests, mask)

    # loss = model_loss(disp_ests, disp_gt, mask)
    loss = F.smooth_l1_loss(disp_ests[mask], disp_gt[mask], reduction='mean')
    # loss_fn = DisparityLoss(tau=1.0)
    # loss = loss_fn(disp_gt, disp_ests,mask)  # following paper loss
    # loss = torch.pow(torch.clamp(torch.abs(disp_ests - disp_gt), min=1.0), 0.125)
    # loss = (loss * mask.float()).sum()


    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            # error_func = disp_error_image_func()
            image_outputs["errormap"] = [error_func.forward(disp_ests, disp_gt)]
            scalar_outputs["EPE"] = [EPE_metric(disp_ests, disp_gt, mask)]
            scalar_outputs["D1"] = [D1_metric(disp_ests, disp_gt, mask)]
            scalar_outputs["Thres1"] = [Thres_metric(disp_ests, disp_gt, mask, 1.0)]
            scalar_outputs["Thres2"] = [Thres_metric(disp_ests, disp_gt, mask, 2.0)]
            scalar_outputs["Thres3"] = [Thres_metric(disp_ests, disp_gt, mask, 3.0)]
    loss.backward()
    optimizer.step()
    # scheduler.step()
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt =  sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda() # [B, H, W]
    disp_ests = model(imgL, imgR)  # [B,1,H,W]
    disp_ests = torch.squeeze(disp_ests, 1)  # [B,H,W]
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    # loss = model_loss(disp_ests, disp_gt, mask)
    loss = F.smooth_l1_loss(disp_ests[mask], disp_gt[mask], reduction='mean')
    # mask = (disp_gt > 0).float() * (disp_gt < 255.0).float()
    # loss = model_loss(disp_gt, disp_ests, mask)
    # loss_fn = DisparityLoss(tau=1.0)
    # loss = loss_fn(disp_gt, disp_ests, mask)
    # loss = torch.pow(torch.clamp(torch.abs(disp_ests - disp_gt), min=1.0), 0.125)
    # loss = (loss * mask.float()).sum()
    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_ests, disp_gt, mask)]
    scalar_outputs["EPE"] = [EPE_metric(disp_ests, disp_gt, mask)]
    scalar_outputs["Thres1"] = [Thres_metric(disp_ests, disp_gt, mask, 1.0)]
    scalar_outputs["Thres2"] = [Thres_metric(disp_ests, disp_gt, mask, 2.0)]
    scalar_outputs["Thres3"] = [Thres_metric(disp_ests, disp_gt, mask, 3.0)]

    if compute_metrics:
        image_outputs["errormap"] = [error_func.forward(disp_ests, disp_gt)]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    # for batch_idx, sample in enumerate(TestImgLoader):
    #     loss, scalar_outputs, image_outputs = test_sample(sample)
    train()