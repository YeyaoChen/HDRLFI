import torch
import numpy as np
import scipy.io as sio
from scipy.signal import convolve2d
import os
import random
import math
import h5py
import OpenEXR, Imath
import pyexr


class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


# Write HDR image using OpenEXR
def writeEXR(img, file):
    try:
        img = np.squeeze(img)
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(file, header)
        R = (img[:, :, 0]).astype(np.float32).tostring()
        G = (img[:, :, 1]).astype(np.float32).tostring()
        B = (img[:, :, 2]).astype(np.float32).tostring()
        out.writePixels({'R': R, 'G': G, 'B': B})
        out.close()
    except Exception as e:
        raise IOException("Failed writing EXR: %s"%e)


def loadEXR(name_hdr):
    return pyexr.read_all(name_hdr)['default'][:, :, 0:3]


def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)


def mk_dir(required_dir):
    if not os.path.exists(required_dir):
        os.makedirs(required_dir)


def lfi2mlia(in_lfi):
    # [ah,aw,3,h,w] to [h*ah,w*aw,c]
    in_ah, in_aw, in_c, in_h, in_w = in_lfi.shape
    in_lfi = np.transpose(in_lfi, [3, 0, 4, 1, 2])   # [h,ah,w,aw,3]
    out_mlia = np.reshape(in_lfi, [in_h*in_ah, in_w*in_aw, in_c])
    return out_mlia


def log_transformation(in_hdr, param_u=5000.):
    out_tm = torch.log(torch.tensor(1.) + torch.tensor(param_u) * in_hdr) / torch.log(torch.tensor(1. + param_u))
    return out_tm


def inverse_log_transformation(log_img, log_param=5000.):
    scale_param = torch.log(torch.tensor(1.) + log_param)
    radia_img = (torch.exp(scale_param * log_img) - torch.tensor(1.)) / torch.tensor(log_param)
    return radia_img


def log_transformation_np(in_hdr, param_u=5000.):
    out_tm = np.log(1.0 + param_u * in_hdr) / np.log(1.0 + param_u)
    return out_tm


def inverse_log_transformation_np(log_img, log_param=5000.):
    scale_param = np.log(1.0 + log_param)
    radia_img = (np.exp(scale_param * log_img) - 1.0) / log_param
    return radia_img


def LDRtoHDR(in_ldr, expo, gamma=1/0.7):
    in_ldr = np.clip(in_ldr, 0, 1)
    out_hdr = in_ldr ** gamma
    out_hdr = out_hdr / expo
    return out_hdr


def HDRtoLDR(in_hdr, expo, inv_gamma=0.7):
    in_hdr = in_hdr * expo
    in_hdr = np.clip(in_hdr, 0, 1)
    out_ldr = in_hdr ** inv_gamma
    return out_ldr


def LDRtoLDR(in_ldr1, expo1, expo2, gamma=1/0.7):
    Radiance = LDRtoHDR(in_ldr1, expo1, gamma)
    out_ldr2 = HDRtoLDR(Radiance, expo2, 1.0/gamma)
    return out_ldr2


def tensor_LDRtoHDR(in_ldr, expo, gamma=1/0.7):
    in_ldr = torch.clamp(in_ldr, min=0, max=1)
    out_hdr = in_ldr ** gamma
    out_hdr = out_hdr / expo
    return out_hdr


def tensor_HDRtoLDR(in_hdr, expo, inv_gamma=0.7):
    in_hdr = in_hdr * expo
    in_hdr = torch.clamp(in_hdr, min=0, max=1)
    out_ldr = in_hdr ** inv_gamma
    return out_ldr


def tensor_LDRtoLDR(in_ldr1, expo1, expo2, gamma=1/0.7):
    Radiance = tensor_LDRtoHDR(in_ldr1, expo1, gamma)
    out_ldr2 = tensor_HDRtoLDR(Radiance, expo2, 1.0/gamma)
    return out_ldr2


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 10 * math.log10(pixel_max / mse)


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    g = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    sumg = g.sum()
    if sumg != 0:
        g /= sumg
    return g


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def calculate_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1.0):
    if not im1.shape == im2.shape:
        raise ValueError("Input Images must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'same')
    mu2 = filter2(im2, window, 'same')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'same') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'same') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'same') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(np.mean(ssim_map))


def calculate_quality(im1, im2):
    # [ah,aw,c,h,w]
    score1 = calculate_psnr(im1, im2)
    score2 = 0.
    an1, an2 = im1.shape[0:2]
    for a1 in range(an1):
        for a2 in range(an2):
            for ci in range(3):
                score2 = score2 + calculate_ssim(im1[a1, a2, ci, :, :], im2[a1, a2, ci, :, :])
    score2 = score2 / (an1 * an2 * 3.)
    return score1, score2


def ColorAugmentation(lfi, index):
    # [ah,aw,h,w,c]
    if index == 1:
        order = [0, 1, 2]
    elif index == 2:
        order = [0, 2, 1]
    elif index == 3:
        order = [1, 0, 2]
    elif index == 4:
        order = [1, 2, 0]
    elif index == 5:
        order = [2, 1, 0]
    else:
        order = [2, 0, 1]
    aug_lfi = lfi[:, :, :, :, order]
    return aug_lfi


def CropPatches_w(image, len, crop):
    # LFimage [1,ah,aw,c,h,w]: left [1,ah,aw,c,h,lw] / middles[n,ah,aw,c,h,mw] / right [1,ah,aw,c,h,rw]
    ah, aw, c, h, w = image.shape[1:]
    left = image[:, :, :, :, :, 0:len+crop]       # [1,ah,aw,c,h,lw]
    num = math.floor((w-len-crop)/len)
    middles = torch.Tensor(num, ah, aw, c, h, len+crop*2).to(image.device)
    for i in range(num):
        middles[i] = image[0, :, :, :, :, (i+1)*len-crop:(i+2)*len+crop]    # middles[n,ah,aw,c,h,mw]
    right = image[:, :, :, :, :, -(len+crop):]    # [1,ah,aw,c,h,rw]
    return left, middles, right


def MergePatches_w(left, middles, right, h, w, len, crop):
    # [B,ah,aw,c,h,w]
    b, ah, aw, c = left.shape[0:4]
    out = torch.Tensor(b, ah, aw, c, h, w).to(left.device)
    out[:, :, :, :, :, :len] = left[:, :, :, :, :, :-crop]      # left
    for i in range(middles.shape[0]):
        out[:, :, :, :, :, len*(i+1):len*(i+2)] = middles[i:i+1, :, :, :, :, crop:-crop]   # middle
    out[:, :, :, :, :, -len:] = right[:, :, :, :, :, crop:]   # right
    return out