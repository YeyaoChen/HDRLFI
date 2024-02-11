import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import numpy as np
from math import exp
import Vgg19


# Functions
def gradient_cal(in_img):
    grad1 = in_img[:, :, :, :, 1:, :] - in_img[:, :, :, :, :-1, :]     # [b,ah,aw,c,h,w]
    grad2 = in_img[:, :, :, :, :, 1:] - in_img[:, :, :, :, :, :-1]
    return grad1, grad2


def epi_gradient_cal(in_epi):
    # [b*ah*h,c,aw,w] or [B*aw*w,c,ah,h]
    grad1 = in_epi[:, :, 1:, :] - in_epi[:, :, :-1, :]
    grad2 = in_epi[:, :, :, 1:] - in_epi[:, :, :, :-1]
    return grad1, grad2


def lfi2epi(lfi):
    bs, ah, aw, c, h, w = lfi.shape       # [b,ah,aw,c,h,w]

    # [b,ah,aw,c,h,w] --> [b*ah*h,c,aw,w] & [B*aw*w,c,ah,h]
    epi_h = lfi.permute(0, 1, 4, 3, 2, 5).reshape(bs*ah*h, c, aw, w)
    epi_v = lfi.permute(0, 2, 5, 3, 1, 4).reshape(bs*aw*w, c, ah, h)
    return epi_h, epi_v


# def L1_loss(X, Y):
#     eps = 1e-6
#     diff = torch.add(X, -Y)
#     error = torch.sqrt(diff * diff + eps)
#     Charbonnier_loss = torch.sum(error) / torch.numel(error)
#     return Charbonnier_loss

def L1_loss(X, Y):
    l1_loss = functional.l1_loss(X, Y)
    return l1_loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = functional.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = functional.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = functional.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = functional.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = functional.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


#################### Loss functions ####################
class L1_Reconstruction_loss(nn.Module):
    def __init__(self):
        super(L1_Reconstruction_loss, self).__init__()

    def forward(self, infer, ref):
        rec_loss = L1_loss(infer, ref)   # [b,c,h,w]
        return rec_loss


class SSIM_loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channels = 3
        self.window = create_window(window_size, self.channels)

    def forward(self, infer, ref):
        bs, ah, aw, c, h, w = infer.shape            # [b,ah,aw,c,h,w]
        infer = infer.reshape(bs*ah*aw, c, h, w)     # [b*ah*aw,c,h,w]
        ref = ref.reshape(bs*ah*aw, c, h, w)         # [b*ah*aw,c,h,w]

        channels = infer.shape[1]
        if channels == self.channels and self.window.data.type() == infer.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channels)

            if infer.is_cuda:
                window = window.cuda(infer.get_device())
            window = window.type_as(infer)

            self.window = window
            self.channels = channels

        ssim_value = _ssim(infer, ref, window, self.window_size, channels, self.size_average)
        return 1.0 - ssim_value


class Perceptual_loss(nn.Module):
    def __init__(self, device):
        super(Perceptual_loss, self).__init__()
        self.device = device

    def forward(self, infer, ref):
        vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)

        bs, ah, aw, c, h, w = infer.shape            # [b,ah,aw,c,h,w]
        infer = infer.reshape(bs*ah*aw, c, h, w)     # [b*ah*aw,c,h,w]
        ref = ref.reshape(bs*ah*aw, c, h, w)         # [b*ah*aw,c,h,w]

        infer_relu = vgg19(infer)
        ref_relu = vgg19(ref)

        per_loss = L1_loss(infer_relu[0], ref_relu[0]) + \
                   L1_loss(infer_relu[1], ref_relu[1]) + \
                   L1_loss(infer_relu[2], ref_relu[2]) + \
                   L1_loss(infer_relu[3], ref_relu[3]) + \
                   L1_loss(infer_relu[4], ref_relu[4])
        return per_loss/5.0


# class Detail_loss(nn.Module):
#     def __init__(self):
#         super(Detail_loss, self).__init__()
#
#     def forward(self, infer, ref):
#         infer_dx, infer_dy = gradient_cal(infer)
#         ref_dx, ref_dy = gradient_cal(ref)
#         detail_loss = L1_loss(infer_dx, ref_dx) + L1_loss(infer_dy, ref_dy)
#         return detail_loss


class Detail_loss(nn.Module):
    def __init__(self):
        super(Detail_loss, self).__init__()

    def forward(self, infer, ref):
        bs, ah, aw, c, h, w = infer.shape           # [b,ah,aw,c,h,w]
        infer = infer.reshape(bs*ah*aw, c, h, w)    # [b*ah*aw,c,h,w]
        ref = ref.reshape(bs*ah*aw, c, h, w)

        kh_win = torch.Tensor([[-1/2, 0, 1/2]]).to(infer.device)
        kv_win = torch.Tensor(([[-1/2], [0], [1/2]])).to(infer.device)

        kh_win = Variable(kh_win.expand(3, 3, 1, 3).contiguous())
        kv_win = Variable(kv_win.expand(3, 3, 3, 1).contiguous())

        infer_h = functional.conv2d(infer, kh_win, padding=1)
        ref_h = functional.conv2d(ref, kh_win, padding=1)

        infer_v = functional.conv2d(infer, kv_win,  padding=1)
        ref_v = functional.conv2d(ref, kv_win, padding=1)

        detail_loss = (L1_loss(infer_h, ref_h) + L1_loss(infer_v, ref_v)) / 2.0
        return detail_loss


class EPI_loss(nn.Module):
    def __init__(self):
        super(EPI_loss, self).__init__()

    def forward(self, infer, ref):
        infer_epi_h, infer_epi_v = lfi2epi(infer)
        infer_dx_h, infer_dy_h = epi_gradient_cal(infer_epi_h)
        infer_dx_v, infer_dy_v = epi_gradient_cal(infer_epi_v)

        ref_epi_h, ref_epi_v = lfi2epi(ref)
        ref_dx_h, ref_dy_h = epi_gradient_cal(ref_epi_h)
        ref_dx_v, ref_dy_v = epi_gradient_cal(ref_epi_v)
        epi_loss_h = L1_loss(infer_dx_h, ref_dx_h) + L1_loss(infer_dy_h, ref_dy_h)
        epi_loss_v = L1_loss(infer_dx_v, ref_dx_v) + L1_loss(infer_dy_v, ref_dy_v)
        epi_loss = epi_loss_h + epi_loss_v
        return epi_loss


def get_loss(opt):
    losses = {}
    if (opt.l1_weight > 0):
        losses['l1_loss'] = L1_Reconstruction_loss()

    if (opt.ssim_weight > 0):
        losses['ssim_loss'] = SSIM_loss()

    if (opt.perceptual_weight > 0):
        losses['perceptual_loss'] = Perceptual_loss(opt.device)

    if (opt.detail_weight > 0):
        losses['detail_loss'] = Detail_loss()

    if (opt.epi_weight > 0):
        losses['epi_loss'] = EPI_loss()
    return losses