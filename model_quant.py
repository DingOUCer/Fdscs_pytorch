from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import DeQuantStub, QuantStub,fuse_modules





class CensusTransform(nn.Module):

    def __init__(self, kernel_size: int ):
        super().__init__()
        self._kernel_size = kernel_size
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        assert len(img.size()) == 4

        if self._kernel_size != 3 and self._kernel_size != 5:
            raise NotImplementedError
        n, c, h, w = img.size()
        # get the center idx of census filter 确定窗口的中心位置，以及边界大小（将计算范围限制在图像的有效区域内）
        census_center = margin = int((self._kernel_size - 1) / 2)
        # init census container  [B,1,H-k+1,W-k+1]，确定最后得到的census的大小
        census = torch.zeros((n, c, h - self._kernel_size + 1, w - self._kernel_size + 1), dtype=(torch.int32), device=img.device)
        center_points = img[:, :, margin:h - margin, margin:w - margin]  # 去除边界中心区域
        # offsets = [(u, v) for v in range(kernel_size) for u in range(kernel_size) if
        #            not u == census_center == v]  # 确定除了中心点以外的其他点的位置
        # offsets = []
        # for v in range(kernel_size):
        #     for u in range(kernel_size):
        #         if not u == census_center == v:
        #             offsets.append((u, v))
        offsets = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (0, 2), (1, 2),
                   (3, 2), (4, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
        for u, v in offsets:
            census = census * 2 + (
                    img[:, :, v:v + h - self._kernel_size + 1, u:u + w - self._kernel_size + 1] >= center_points).int()

        census = F.pad(census, (margin, margin, margin, margin), mode='constant', value=0.0)

        return census

class RgbToYcbcr(nn.Module):
    """Convert an image from RGB to YCbCr.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5
    """

    def __init__(self):
        super(RgbToYcbcr, self).__init__()



    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGB image to YCbCr.

                    Args:
                        image (torch.Tensor): RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

                    Returns:
                        torch.Tensor: YCbCr version of the image with shape :math:`(*, 3, H, W)`.

                    Examples:
                        >>> input = torch.rand(2, 3, 4, 5)
                        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5
                    """

        if not isinstance(image, torch.Tensor):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))

        if len(image.shape) < 3 or image.shape[(-3)] != 3:
            raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(image.shape))

        r = image[..., 0, :, :]
        g = image[..., 1, :, :]
        b = image[..., 2, :, :]
        delta = 0.5
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = (b - y) * 0.564 + delta
        cr = (r - y) * 0.713 + delta
        return torch.stack([y, cb, cr], -3)



class Hamming(nn.Module):

    def __init__(self, maxdisp: int):
        super(Hamming, self).__init__()
        self.maxdisp = maxdisp
        # self.cat4 = nn.quantized.FloatFunctional()

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if not len(left.size()) == 4:
            raise AssertionError
        if not len(right.size()) == 4:
            raise AssertionError

        if not isinstance(left, torch.Tensor):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(left)))
        if not isinstance(right, torch.Tensor):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(right)))
        if left.dtype != torch.int32:
            raise ValueError('Input dtype must be torch.int32. Got {}'.format(left.dtype))
        if right.dtype != torch.int32:
            raise ValueError('Input dtype must be torch.int32. Got {}'.format(right.dtype))

        n, c, h, w = left.size()
        hamming_list = list()
        for d in range(self.maxdisp):
            left_valid, right_valid = left[:, :, :, d:w], right[:, :, :, 0:w - d]
            hamming = torch.zeros((left_valid.shape), dtype=(torch.int32), device=(left.device))
            mask = torch.ones((left_valid.shape), dtype=(torch.int32), device=(left.device))
            left_valid = left_valid.__xor__(right_valid)
            for i in range(23, -1, -1):
                hamming = hamming.add(left_valid.__and__(mask))
                left_valid = left_valid >> 1

            hamming = torch.nn.functional.pad(hamming, (0, d, 0, 0), mode='constant', value=0.0 )
            hamming_list.append(hamming)

        # return self.cat4.cat(hamming_list, 1).float()
        return torch.cat(hamming_list, 1).float()


class AbsDiff(nn.Module):

    def __init__(self, maxdisp: int):
        super(AbsDiff, self).__init__()
        self.maxdisp = maxdisp
        # self.cat5 = nn.quantized.FloatFunctional()

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if not len(left.size()) == 4:
            raise AssertionError

        if not len(right.size()) == 4:
            raise AssertionError

        if not isinstance(left, torch.Tensor):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(left)))
        if not isinstance(right, torch.Tensor):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(right)))
        if left.dtype != torch.float32:
            raise ValueError('Input dtype must be torch.float32. Got {}'.format(left.dtype))
        if right.dtype != torch.float32:
            raise ValueError('Input dtype must be torch.float32. Got {}'.format(right.dtype))

        n, c, h, w = left.size()
        abs_diff_list = list()
        for d in range(self.maxdisp):
            left_valid, right_valid = left[:, :, :, d:w], right[:, :, :, 0:w - d]
            abs_diff = torch.abs(left_valid - right_valid)
            abs_diff = F.pad(abs_diff, (0, d, 0, 0), mode='constant', value=0.0)
            abs_diff_list.append(abs_diff)

        # return self.cat5.cat(abs_diff_list, 1)
        return torch.cat(abs_diff_list, 1)


class DoubleConv(nn.Module):
    """
    (convolution => [BN] => LeakyReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x_maxpool = self.maxpool(x)
        return x_maxpool, x


class Up(nn.Module):
    """
    Upscaling then double conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_up = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv = DoubleConv(out_channels * 2, out_channels)
        self.cat6 = nn.quantized.FloatFunctional()

    def forward(self, x1:torch.Tensor, x2:torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x1 = self.conv_up(x1)

        # x = torch.cat([x2, x1], dim=1)
        x = self.cat6.cat([x2, x1], 1)
        x = self.conv(x)

        return x


class Unify_size(nn.Module):
    def __init__(self):
        super(Unify_size, self).__init__()
        # self.cat1 = nn.quantized.FloatFunctional()

    def forward(self, Y_cencus, U_abs_diff, V_abs_diff):
        Y_cencus = (Y_cencus - 11.08282948) / 0.1949711
        U_abs_diff = (U_abs_diff - 0.02175535) / 35.91432953
        V_abs_diff = (V_abs_diff - 0.02679042) / 26.79782867
        # gether data to form a complete costs volume
        return torch.cat([Y_cencus, U_abs_diff, V_abs_diff], 1).float()
        # return self.cat1.cat([Y_cencus, U_abs_diff, V_abs_diff], 1).float()



class UNet(nn.Module):
    """It is re-designed by FDSCS work.

    Args:
        n_channels ([int]): input channels
    """

    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.down1 = Down(in_channels=35, out_channels=32)
        self.down2 = Down(in_channels=32, out_channels=48)
        self.down3 = Down(in_channels=48, out_channels=64)
        self.down4 = Down(in_channels=64, out_channels=80)
        self.inc = DoubleConv(in_channels=80, out_channels=96)
        self.up1 = Up(in_channels=96, out_channels=80)
        self.up2 = Up(in_channels=80, out_channels=64)
        self.up3 = Up(in_channels=64, out_channels=48)
        self.up4 = Up(in_channels=48, out_channels=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, x1 = self.down1(x)
        x, x2 = self.down2(x)
        x, x3 = self.down3(x)
        x, x4 = self.down4(x)
        x = self.inc(x)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x

def conv2d_bn(in_planes, out_planes, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_planes))


def conv2d(in_planes, out_planes, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))




# class post_process(nn.Module):
#     def __init__(self):
#         super(post_process, self).__init__()
#         self.add1 = nn.quantized.FloatFunctional()
#         # self.out_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)
#         # self.LeakyReLU = nn.LeakyReLU(0.1, inplace=True)
#
#     def forward(self, out: torch.Tensor) -> torch.Tensor:
#         out = self.add1.add(out, 128)
#         # out = self.LeakyReLU(out)
#         return out


class FDSCS(nn.Module):
    def __init__(self, maxdisp):
        super(FDSCS, self).__init__()
        self.maxdisp = maxdisp
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.yuv_to_rgb = YcbcrToRgb()
        self.rgb_to_yuv = RgbToYcbcr()
        self.census = CensusTransform(kernel_size=5)
        self.hamming = Hamming(maxdisp=(self.maxdisp))
        self.abs_diff = AbsDiff(maxdisp=(self.maxdisp))
        self.unifiy = Unify_size()

        self.enc0_conv2d_bn = conv2d_bn(in_planes=384, out_planes=192, kernel_size=1, stride=1, padding=0)
        self.enc1_conv2d_bn = conv2d_bn(in_planes=192, out_planes=96, kernel_size=1, stride=1, padding=0)
        self.enc2_conv2d_bn = conv2d_bn(in_planes=96, out_planes=48, kernel_size=1, stride=1, padding=0)
        self.enc3_conv2d_bn = conv2d_bn(in_planes=48, out_planes=32, kernel_size=1, stride=1, padding=0)
        self.cenc0_conv2d_bn = conv2d_bn(in_planes=35, out_planes=32, kernel_size=3, stride=1, padding=(1, 1))
        self.cenc1_conv2d_bn = conv2d_bn(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=(1, 1))
        self.cenc2_conv2d_bn = conv2d_bn(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=(1, 1))

        self.unet = UNet(n_channels=35)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.cat1 = nn.quantized.FloatFunctional()
        self.cat2 = nn.quantized.FloatFunctional()
        self.cat3 = nn.quantized.FloatFunctional()
        self.add1 = nn.quantized.FloatFunctional()

    # def lowrescv(self, left:torch.Tensor, right:torch.Tensor, imsz=None, maxdisp: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     left:[B, 3, H, W] => [16,3,256,512]
    #     right:[B, 3, H, W] => [16,3,256,512]
    #     """
    #     # half resolution
    #     left, right = self.avg_pool(left), self.avg_pool(right)
    #
    #     # rgb to yuv
    #     left, right = self.rgb_to_yuv(left), self.rgb_to_yuv(right)
    #
    #     # doing census and then hamming at the first channal--Y channal.
    #     Y_cencus = self.hamming(self.census(left[:, 0:1, :, :]), self.census(right[:, 0:1, :, :]))
    #
    #     # doing absolute diffrence at the U channal an the V channal.
    #     U_abs_diff = self.abs_diff(left[:, 1:2, :, :], right[:, 1:2, :, :])
    #     V_abs_diff = self.abs_diff(left[:, 2:3, :, :], right[:, 2:3, :, :])
    #
    #     # unifiy the scale
    #     Y_cencus = (Y_cencus - 11.08282948) / 0.1949711
    #     U_abs_diff = (U_abs_diff - 0.02175535) / 35.91432953
    #     V_abs_diff = (V_abs_diff - 0.02679042) / 26.79782867
    #
    #     # gether data to form a complete costs volume
    #     costs_volume = self.cat1.cat([Y_cencus, U_abs_diff, V_abs_diff], 1)
    #
    #     return left, costs_volume

    def forward(self, left, right):
        # left, right = self.quant(left), self.quant(right)
        # costs_volume
        # half_resolution_left, costs_volume = self.lowrescv(left, right)
        # half resolution
        left, right = self.avg_pool(left), self.avg_pool(right)

        # rgb to yuv
        left, right = self.rgb_to_yuv(left), self.rgb_to_yuv(right)

        # doing census and then hamming at the first channal--Y channal.
        Y_cencus = self.hamming(self.census(left[:, 0:1, :, :]), self.census(right[:, 0:1, :, :]))

        # doing absolute diffrence at the U channal an the V channal.
        U_abs_diff = self.abs_diff(left[:, 1:2, :, :], right[:, 1:2, :, :])
        V_abs_diff = self.abs_diff(left[:, 2:3, :, :], right[:, 2:3, :, :])

        # unifiy the scale
        costs_volume = self.unifiy(Y_cencus, U_abs_diff, V_abs_diff)

        costs_volume = self.quant(costs_volume)
        left = self.quant(left)

        # costs_signature
        costs_signature = costs_volume
        costs_signature = self.enc0_conv2d_bn(costs_signature)
        costs_signature = self.enc1_conv2d_bn(costs_signature)
        costs_signature = self.enc2_conv2d_bn(costs_signature)
        costs_signature = self.enc3_conv2d_bn(costs_signature)
        costs_signature = self.cat2.cat([left, costs_signature], 1) # [16, 35, 128, 256]

        # costs_signature

        costs_signature = self.cenc0_conv2d_bn(costs_signature)
        costs_signature = self.cenc1_conv2d_bn(costs_signature)
        costs_signature = self.cenc2_conv2d_bn(costs_signature)

        unet_input = self.cat3.cat([left, costs_signature], 1)

        # unet
        out = self.unet(unet_input)
        out = self.out_conv(out)

        # if out.dtype == torch.quint8:
        #     xq = torch.quantize_per_tensor(torch.tensor(128.0), scale=0.1, zero_point=128, dtype=torch.quint8)
        #     out = self.add1.add(out, xq)
        #     out = self.LeakyReLU(out)
        #     out = self.up(out)
        #     out = self.dequant(out)
        #     return out
        out = self.add1.add(out, self.quant(torch.tensor(128.0)))
        out = self.LeakyReLU(out)
        out = self.up(out)
        out = self.dequant(out)
        return out

    # 融合bn层和conv层
    def fuse_model(self):
        conv_name_dict = {}
        bn_name_dict = {}
        for idx, tu in enumerate(self.named_modules()):
            module_name, module = tu
            if isinstance(module, nn.Conv2d):
                conv_name_dict[idx] = module_name
            elif isinstance(module, nn.BatchNorm2d):
                bn_name_dict[idx] = module_name

        for idx, bn_name in bn_name_dict.items():
            assert (idx - 1) in conv_name_dict.keys()
            conv_name = conv_name_dict[idx - 1]
            fuse_modules(self, [conv_name, bn_name], inplace=True)