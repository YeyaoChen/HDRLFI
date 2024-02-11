import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from modules import *


class Build_HDRLFINet(nn.Module):
    def __init__(self, par):
        super(Build_HDRLFINet, self).__init__()

        self.init_embed1 = Initial_conv(6, 32)
        self.init_embed2 = Initial_conv(6, 32)
        self.init_embed3 = Initial_conv(6, 32)
        self.aux_enc1 = Aux_encoder(32, 64)
        self.aux_enc2 = Aux_encoder(32, 64)
        self.main_enc = Main_encoder(32, 64, par.ang_res)
        self.main_dec = Main_decoder(256)

    def forward(self, in_lfs):
        in_lf1 = torch.cat((in_lfs[:, :, :, 0:3, :, :], in_lfs[:, :, :, 9:12, :, :]), dim=3)      # [b,ah,aw,6,h,w]
        in_lf2 = torch.cat((in_lfs[:, :, :, 3:6, :, :], in_lfs[:, :, :, 12:15, :, :]), dim=3)
        in_lf3 = torch.cat((in_lfs[:, :, :, 6:9, :, :], in_lfs[:, :, :, 15:18, :, :]), dim=3)

        b, ah, aw, c, sh, sw = in_lf2.shape
        in_lf1 = in_lf1.reshape([b*ah*aw, c, sh, sw])     # [b*ah*aw,6,h,w]
        in_lf2 = in_lf2.reshape([b*ah*aw, c, sh, sw])
        in_lf3 = in_lf3.reshape([b*ah*aw, c, sh, sw])

        ##########################  Mapping to feature space  ##########################
        init_feats1 = self.init_embed1(in_lf1)             # [b*ah*aw,32,h,w]
        init_feats2 = self.init_embed2(in_lf2)
        init_feats3 = self.init_embed3(in_lf3)

        #######################  Under/Over-exposure feature  #######################
        spa_feats1 = self.aux_enc1(init_feats1)
        spa_feats3 = self.aux_enc2(init_feats3)

        #############################  Reference feature  #############################
        ref_feats = self.main_enc(init_feats2, spa_feats1, spa_feats3)

        ##############################  Reconstruction  ##############################
        dec_feats = [init_feats2, ref_feats[0], ref_feats[1], ref_feats[2]]
        net_img = self.main_dec(dec_feats)        # [b*ah*aw,3,h,w]
        net_img = net_img.reshape(b, ah, aw, 3, sh, sw)
        return net_img
