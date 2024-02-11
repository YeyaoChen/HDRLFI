import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from Deform import DeformConv2d


# Sub-modules
class size_interp(nn.Module):
    def __init__(self):
        super(size_interp, self).__init__()

    def forward(self, x, tar_size):
        return functional.interpolate(input=x, size=tar_size, mode='bilinear')


class Ang_conv(nn.Module):
    def __init__(self, in_num, out_num, an):
        super(Ang_conv, self).__init__()
        self.an = an
        self.an2 = an ** 2
        self.an_conv = nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=an, stride=1, padding=0)

    def forward(self, in_x):
        ban2, c, h, w = in_x.shape    # [b*ah*aw,c,h,w]
        bs = ban2//self.an2

        # spatial --> angular
        in_x = in_x.reshape([bs, self.an2, c, h*w])           # [b,ah*aw,c,h*w]
        in_x = torch.transpose(in_x, 1, 3)                    # [b,h*w,c,ah*aw]
        in_x = in_x.reshape(bs*h*w, c, self.an, self.an)      # [b*h*w,c,ah,aw]

        # angular convolution and duplicate
        out_x = self.an_conv(in_x)                            # [b*h*w,c,1,1]
        out_x = out_x.repeat(1, 1, self.an, self.an)          # [b*h*w,c,ah,aw]

        # angular --> spatial
        an_c = out_x.shape[1]
        out_x = out_x.reshape(bs, h*w, an_c, self.an2)        # [b,h*w,c,ah*aw]
        out_x = torch.transpose(out_x, 1, 3)                  # [b,ah*aw,c,h*w]
        out_x = out_x.reshape(bs*self.an2, an_c, h, w)        # [b*ah*aw,c,h,w]
        return out_x


class Ang_embed(nn.Module):
    def __init__(self, in_num, out_num):
        super(Ang_embed, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=1, padding=1)

    def forward(self, in_x, in_y):
        gamma_par = self.conv1(in_y)
        beta_par = self.conv2(in_y)
        out_x = in_x * gamma_par + beta_par
        return out_x


class Res_block(nn.Module):
    def __init__(self, in_num, out_num):
        super(Res_block, self).__init__()
        self.res_conv1 = nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=1, padding=1)
        self.res_conv2 = nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, in_x):
        res_x = self.lrelu(self.res_conv1(in_x))
        res_x = self.res_conv2(res_x)
        out_x = in_x + res_x
        return out_x


class ResASPP(nn.Module):
    def __init__(self, in_num, out_num):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_2 = nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv_3 = nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=1, padding=4, dilation=4)
        self.conv_r = nn.Conv2d(in_channels=out_num*3, out_channels=out_num, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def __call__(self, in_x):
        res_x1 = self.lrelu(self.conv_1(in_x))
        res_x2 = self.lrelu(self.conv_2(in_x))
        res_x3 = self.lrelu(self.conv_3(in_x))

        res_x = self.conv_r(torch.cat((res_x1, res_x2, res_x3), dim=1))
        out_x = in_x + res_x
        return out_x


class Deform_align(nn.Module):
    def __init__(self, in_num, out_num):
        super(Deform_align, self).__init__()
        self.conv01 = nn.Conv2d(in_channels=in_num*2, out_channels=out_num, kernel_size=1, stride=1, padding=0)
        self.ASPP = ResASPP(in_num, out_num)
        self.conv02 = nn.Conv2d(in_channels=out_num, out_channels=2*9, kernel_size=1, stride=1, padding=0)
        self.deform = DeformConv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def __call__(self, in_aux, in_ref):
        cat_x = torch.cat((in_aux, in_ref), dim=1)
        off_x = self.lrelu(self.conv01(cat_x))
        off_x = self.ASPP(off_x)
        off_x = self.conv02(off_x)        # [b*ah*aw,18,h,w]
        align_x = self.deform(in_aux, off_x)
        return align_x


# Main modules
class Initial_conv(nn.Module):
    def __init__(self, in_num, out_num):
        super(Initial_conv, self).__init__()
        self.init_conv = nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, in_x):
        out_x = self.lrelu(self.init_conv(in_x))        # [b*ah*aw,c,h,w]
        return out_x


class Aux_encoder(nn.Module):
    def __init__(self, in_num, out_num):
        super(Aux_encoder, self).__init__()
        self.spa_conv1 = nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=2, padding=1)
        self.spa_conv2 = nn.Conv2d(in_channels=out_num, out_channels=out_num*2, kernel_size=3, stride=2, padding=1)
        self.spa_conv3 = nn.Conv2d(in_channels=out_num*2, out_channels=out_num*4, kernel_size=3, stride=2, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, in_x):
        spa_x1 = self.lrelu(self.spa_conv1(in_x))        # [b*ah*aw,2c,h/2,w/2]
        spa_x2 = self.lrelu(self.spa_conv2(spa_x1))      # [b*ah*aw,4c,h/4,w/4]
        spa_x3 = self.lrelu(self.spa_conv3(spa_x2))      # [b*ah*aw,8c,h/8,w/8]
        spa_x = [spa_x1, spa_x2, spa_x3]
        return spa_x


class Main_encoder(nn.Module):
    def __init__(self, in_num, out_num, an):
        super(Main_encoder, self).__init__()
        self.spa_conv1 = nn.Conv2d(in_channels=in_num, out_channels=out_num, kernel_size=3, stride=2, padding=1)
        self.spa_conv2 = nn.Conv2d(in_channels=out_num, out_channels=out_num*2, kernel_size=3, stride=2, padding=1)
        self.spa_conv3 = nn.Conv2d(in_channels=out_num*2, out_channels=out_num*4, kernel_size=3, stride=2, padding=1)
        self.align_conv11 = Deform_align(out_num, out_num)
        self.align_conv12 = Deform_align(out_num, out_num)
        self.align_conv21 = Deform_align(out_num*2, out_num*2)
        self.align_conv22 = Deform_align(out_num*2, out_num*2)
        self.align_conv31 = Deform_align(out_num*4, out_num*4)
        self.align_conv32 = Deform_align(out_num*4, out_num*4)
        self.fuse_conv1 = nn.Conv2d(in_channels=out_num*3, out_channels=out_num, kernel_size=1, stride=1, padding=0)
        self.fuse_conv2 = nn.Conv2d(in_channels=out_num*6, out_channels=out_num*2, kernel_size=1, stride=1, padding=0)
        self.fuse_conv3 = nn.Conv2d(in_channels=out_num*12, out_channels=out_num*4, kernel_size=1, stride=1, padding=0)
        self.aux_ang_conv11 = Ang_conv(out_num, out_num, an)
        self.aux_ang_conv12 = Ang_conv(out_num, out_num, an)
        self.aux_ang_conv21 = Ang_conv(out_num*2, out_num*2, an)
        self.aux_ang_conv22 = Ang_conv(out_num*2, out_num*2, an)
        self.aux_ang_conv31 = Ang_conv(out_num*4, out_num*4, an)
        self.aux_ang_conv32 = Ang_conv(out_num*4, out_num*4, an)
        self.ang_conv1 = Ang_conv(out_num, out_num, an)
        self.ang_conv2 = Ang_conv(out_num*2, out_num*2, an)
        self.ang_conv3 = Ang_conv(out_num*4, out_num*4, an)
        self.fuse_conv01 = nn.Conv2d(in_channels=out_num*3, out_channels=out_num, kernel_size=1, stride=1, padding=0)
        self.fuse_conv02 = nn.Conv2d(in_channels=out_num*6, out_channels=out_num*2, kernel_size=1, stride=1, padding=0)
        self.fuse_conv03 = nn.Conv2d(in_channels=out_num*12, out_channels=out_num*4, kernel_size=1, stride=1, padding=0)
        self.embed_conv1 = Ang_embed(out_num, out_num)
        self.embed_conv2 = Ang_embed(out_num*2, out_num*2)
        self.embed_conv3 = Ang_embed(out_num*4, out_num*4)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, in_x, aux_spa1, aux_spa2):
        # step1: spatial fusion
        spa_x1 = self.lrelu(self.spa_conv1(in_x))                            # [b*ah*aw,2c,h/2,w/2]
        align_spa11 = self.lrelu(self.align_conv11(aux_spa1[0], spa_x1))     # [b*ah*aw,2c,h/2,w/2]
        align_spa12 = self.lrelu(self.align_conv12(aux_spa2[0], spa_x1))     # [b*ah*aw,2c,h/2,w/2]
        cat_spa1 = torch.cat((spa_x1, align_spa11, align_spa12), dim=1)      # [b*ah*aw,2c*3,h/2,w/2]
        spa_fuse1 = self.fuse_conv1(cat_spa1)                                # [b*ah*aw,2c,h/2,w/2]

        # step1: angular embedding
        aux_ang11 = self.lrelu(self.aux_ang_conv11(align_spa11))             # [b*ah*aw,2c,h/2,w/2]
        aux_ang12 = self.lrelu(self.aux_ang_conv12(align_spa12))             # [b*ah*aw,2c,h/2,w/2]
        ang_x1 = self.lrelu(self.ang_conv1(spa_fuse1))                       # [b*ah*aw,2c,h/2,w/2]
        cat_ang1 = torch.cat((ang_x1, aux_ang11, aux_ang12), dim=1)          # [b*ah*aw,2c*3,h/2,w/2]
        ang_fused1 = self.fuse_conv01(cat_ang1)                              # [b*ah*aw,2c,h/2,w/2]
        fuse_x1 = self.embed_conv1(spa_fuse1, ang_fused1)                    # [b*ah*aw,2c,h/2,w/2]

        # step2: spatial fusion
        spa_x2 = self.lrelu(self.spa_conv2(fuse_x1))                         # [b*ah*aw,4c,h/4,w/4]
        align_spa21 = self.lrelu(self.align_conv21(aux_spa1[1], spa_x2))     # [b*ah*aw,4c,h/4,w/4]
        align_spa22 = self.lrelu(self.align_conv22(aux_spa2[1], spa_x2))     # [b*ah*aw,4c,h/4,w/4]
        cat_spa2 = torch.cat((spa_x2, align_spa21, align_spa22), dim=1)      # [b*ah*aw,4c*3,h/4,w/4]
        spa_fuse2 = self.fuse_conv2(cat_spa2)                                # [b*ah*aw,4c,h/4,w/4]

        # step2: angular embedding
        aux_ang21 = self.lrelu(self.aux_ang_conv21(align_spa21))             # [b*ah*aw,4c,h/4,w/4]
        aux_ang22 = self.lrelu(self.aux_ang_conv22(align_spa22))             # [b*ah*aw,4c,h/4,w/4]
        ang_x2 = self.lrelu(self.ang_conv2(spa_fuse2))                       # [b*ah*aw,4c,h/4,w/4]
        cat_ang2 = torch.cat((ang_x2, aux_ang21, aux_ang22), dim=1)          # [b*ah*aw,4c*3,h/4,w/4]
        ang_fused2 = self.fuse_conv02(cat_ang2)                              # [b*ah*aw,4c,h/4,w/4]
        fuse_x2 = self.embed_conv2(spa_fuse2, ang_fused2)                    # [b*ah*aw,4c,h/4,w/4]

        # step3: spatial fusion
        spa_x3 = self.lrelu(self.spa_conv3(fuse_x2))                         # [b*ah*aw,8c,h/8,w/8]
        align_spa31 = self.lrelu(self.align_conv31(aux_spa1[2], spa_x3))     # [b*ah*aw,8c,h/8,w/8]
        align_spa32 = self.lrelu(self.align_conv32(aux_spa2[2], spa_x3))     # [b*ah*aw,8c,h/8,w/8]
        cat_spa3 = torch.cat((spa_x3, align_spa31, align_spa32), dim=1)      # [b*ah*aw,8c*3,h/8,w/8]
        spa_fuse3 = self.fuse_conv3(cat_spa3)                                # [b*ah*aw,8c,h/8,w/8]

        # step3: angular embedding
        aux_ang31 = self.lrelu(self.aux_ang_conv31(align_spa31))             # [b*ah*aw,8c,h/8,w/8]
        aux_ang32 = self.lrelu(self.aux_ang_conv32(align_spa32))             # [b*ah*aw,8c,h/8,w/8]
        ang_x3 = self.lrelu(self.ang_conv3(spa_fuse3))                       # [b*ah*aw,8c,h/8,w/8]
        cat_ang3 = torch.cat((ang_x3, aux_ang31, aux_ang32), dim=1)          # [b*ah*aw,8c*3,h/8,w/8]
        ang_fused3 = self.fuse_conv03(cat_ang3)                              # [b*ah*aw,8c,h/8,w/8]
        fuse_x3 = self.embed_conv3(spa_fuse3, ang_fused3)                    # [b*ah*aw,8c,h/8,w/8]

        fused_x = [fuse_x1, fuse_x2, fuse_x3]
        return fused_x


class Main_decoder(nn.Module):
    def __init__(self, in_num):
        super(Main_decoder, self).__init__()
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=in_num, out_channels=in_num//2, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(in_channels=in_num//2, out_channels=in_num//4, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(in_channels=in_num//4, out_channels=in_num//8, kernel_size=4, stride=2, padding=1)
        self.cat_conv1 = nn.Conv2d(in_channels=in_num//2*2, out_channels=in_num//2, kernel_size=1, stride=1, padding=0)
        self.cat_conv2 = nn.Conv2d(in_channels=in_num//4*2, out_channels=in_num//4, kernel_size=1, stride=1, padding=0)
        self.cat_conv3 = nn.Conv2d(in_channels=in_num//8*2, out_channels=in_num//8, kernel_size=1, stride=1, padding=0)
        self.res_block1 = Res_block(in_num//2, in_num//2)
        self.res_block2 = Res_block(in_num//4, in_num//4)
        self.res_block3 = Res_block(in_num//8, in_num//8)
        self.rec_conv = nn.Conv2d(in_channels=in_num//8, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.feat_interp = size_interp()
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_x):
        dec_x1 = self.lrelu(self.dec_conv1(in_x[3]))      # [b*ah*aw,4c,h/4,w/4]
        dec_x1_size = [in_x[2].shape[2], in_x[2].shape[3]]
        dec_x1 = self.feat_interp(dec_x1, dec_x1_size)
        cat_x1 = torch.cat((dec_x1, in_x[2]), dim=1)      # [b*ah*aw,4c*2,h/4,w/4]
        red_x1 = self.cat_conv1(cat_x1)                   # [b*ah*aw,4c,h/4,w/4]
        red_x1 = self.res_block1(red_x1)                  # [b*ah*aw,4c,h/4,w/4]

        dec_x2 = self.lrelu(self.dec_conv2(red_x1))       # [b*ah*aw,2c,h/2,w/2]
        dec_x2_size = [in_x[1].shape[2], in_x[1].shape[3]]
        dec_x2 = self.feat_interp(dec_x2, dec_x2_size)
        cat_x2 = torch.cat((dec_x2, in_x[1]), dim=1)      # [b*ah*aw,2c*2,h/2,w/2]
        red_x2 = self.cat_conv2(cat_x2)                   # [b*ah*aw,2c,h/2,w/2]
        red_x2 = self.res_block2(red_x2)                  # [b*ah*aw,2c,h/2,w/2]

        dec_x3 = self.lrelu(self.dec_conv3(red_x2))       # [b*ah*aw,c,h,w]
        dec_x3_size = [in_x[0].shape[2], in_x[0].shape[3]]
        dec_x3 = self.feat_interp(dec_x3, dec_x3_size)
        cat_x3 = torch.cat((dec_x3, in_x[0]), dim=1)      # [b*ah*aw,c*2,h,w]
        red_x3 = self.cat_conv3(cat_x3)                   # [b*ah*aw,c,h,w]
        red_x3 = self.res_block3(red_x3)                  # [b*ah*aw,c,h,w]

        # rec_x = self.sigmoid(self.rec_conv(red_x3))       # [b*ah*aw,3,h,w]
        rec_x = self.rec_conv(red_x3)                     # [b*ah*aw,3,h,w]
        return rec_x


