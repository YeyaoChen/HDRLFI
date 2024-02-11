import torch
import torch.utils.data as data
import h5py
import numpy as np
import random
from utils import LDRtoHDR, ColorAugmentation, log_transformation_np


################################  Read Training data  ################################
class TrainSetLoader(data.Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader, self).__init__()
        self.psize = cfg.patch_size
        self.ang = cfg.ang_res
        self.crf_value = cfg.crf_gamma

        hf = h5py.File(cfg.dataset_path + '.h5')             # Input path
        self.UnderLF = hf.get('Under_Exposure_LFimgs')       # [ah,aw,h,w,3,N]
        self.UnderLF = self.UnderLF[:self.ang, :self.ang, :, :, :, :]
        self.NormalLF = hf.get('Normal_Exposure_LFimgs')     # [ah,aw,h,w,3,N]
        self.NormalLF = self.NormalLF[:self.ang, :self.ang, :, :, :, :]
        self.OverLF = hf.get('Over_Exposure_LFimgs')         # [ah,aw,h,w,3,N]
        self.OverLF = self.OverLF[:self.ang, :self.ang, :, :, :, :]
        self.GT = hf.get('Label_HDR_LFimgs')                 # [ah,aw,h,w,3,N]
        self.GT = self.GT[:self.ang, :self.ang, :, :, :, :]
        self.Expo = hf.get('Exposure_values')                # [3,N]

    def __getitem__(self, index):
        under_lfi = self.UnderLF[:, :, :, :, :, index]          # [ah,aw,h,w,3]
        normal_lfi = self.NormalLF[:, :, :, :, :, index]        # [ah,aw,h,w,3]
        over_lfi = self.OverLF[:, :, :, :, :, index]            # [ah,aw,h,w,3]
        gt_lfi = self.GT[:, :, :, :, :, index]            # [ah,aw,h,w,3]
        expo_value = self.Expo[:, index]                  # [3]

        me_lfi = np.concatenate([under_lfi, normal_lfi, over_lfi], axis=-1)
        me_lfi = me_lfi.astype(np.float32) / 65535.0      # [0,1]
        gt_lfi = gt_lfi.astype(np.float32)
        expo_value = expo_value.astype(np.float32)
        expo_value = 2 ** expo_value

        ###################  Crop to LF patch  ###################
        sai_h, sai_w, sai_c = gt_lfi.shape[2:5]

        xx = random.randrange(0, sai_h - self.psize)
        yy = random.randrange(0, sai_w - self.psize)
        me_lfi = me_lfi[:, :, xx:xx + self.psize, yy:yy + self.psize, :]       # [ah,aw,ph,pw,9]
        gt_lfi = gt_lfi[:, :, xx:xx + self.psize, yy:yy + self.psize, :]       # [ah,aw,ph,pw,3]

        ####################  4D Augmentation  ####################
        # flip
        if np.random.rand(1) > 0.5:
            me_lfi = np.flip(np.flip(me_lfi, 0), 2)
            gt_lfi = np.flip(np.flip(gt_lfi, 0), 2)
        if np.random.rand(1) > 0.5:
            me_lfi = np.flip(np.flip(me_lfi, 1), 3)
            gt_lfi = np.flip(np.flip(gt_lfi, 1), 3)
        # rotate
        r_ang = np.random.randint(1, 5)
        me_lfi = np.rot90(me_lfi, r_ang, (2, 3))
        me_lfi = np.rot90(me_lfi, r_ang, (0, 1))
        gt_lfi = np.rot90(gt_lfi, r_ang, (2, 3))
        gt_lfi = np.rot90(gt_lfi, r_ang, (0, 1))
        # color
        c_ang = np.random.randint(1, 7)
        me_lfi = np.concatenate([ColorAugmentation(me_lfi[:, :, :, :, 0:3], c_ang),
                                 ColorAugmentation(me_lfi[:, :, :, :, 3:6], c_ang),
                                 ColorAugmentation(me_lfi[:, :, :, :, 6:9], c_ang)], axis=-1)
        gt_lfi = ColorAugmentation(gt_lfi, c_ang)

        ##############################  Convert to linear hdr domain  ##############################
        me_hdr_lfi = np.concatenate([LDRtoHDR(me_lfi[:, :, :, :, 0:3], expo_value[0], gamma=self.crf_value),
                                     LDRtoHDR(me_lfi[:, :, :, :, 3:6], expo_value[1], gamma=self.crf_value),
                                     LDRtoHDR(me_lfi[:, :, :, :, 6:9], expo_value[2], gamma=self.crf_value)], axis=-1)
        # log hdr domain
        me_hdr_lfi = log_transformation_np(me_hdr_lfi, param_u=5000.)

        # Inputs (LDR + HDR)
        me_input = np.concatenate([me_lfi, me_hdr_lfi], axis=-1)     # [ah,aw,ph,pw,18]

        # gt: log hdr domain
        gt_lfi = log_transformation_np(gt_lfi, param_u=5000.)

        ##########################  Get input and label  ###########################
        me_input = np.transpose(me_input, [0, 1, 4, 2, 3])          # [ah,aw,18,h,w]
        gt_lfi = np.transpose(gt_lfi, [0, 1, 4, 2, 3])              # [ah,aw,3,h,w]

        ###########################  Convert to tensor  ###########################
        me_input = torch.from_numpy(me_input)
        gt_lfi = torch.from_numpy(gt_lfi)
        return me_input, gt_lfi

    def __len__(self):
        return self.UnderLF.shape[5]



###################################  Read Test data  ###################################
class TestSetLoader(data.Dataset):
    def __init__(self, cfg):
        super(TestSetLoader, self).__init__()
        self.ang = cfg.ang_res
        self.crf_value = cfg.crf_gamma

        hf = h5py.File(cfg.dataset_path + '.h5')           # Input path
        self.UnderLF = hf.get('Under_Exposure_LFimgs')     # [ah,aw,h,w,3,N]
        self.UnderLF = self.UnderLF[:self.ang, :self.ang, :, :, :, :]
        self.NormalLF = hf.get('Normal_Exposure_LFimgs')   # [ah,aw,h,w,3,N]
        self.NormalLF = self.NormalLF[:self.ang, :self.ang, :, :, :, :]
        self.OverLF = hf.get('Over_Exposure_LFimgs')       # [ah,aw,h,w,3,N]
        self.OverLF = self.OverLF[:self.ang, :self.ang, :, :, :, :]
        self.GT = hf.get('Label_HDR_LFimgs')               # [ah,aw,h,w,3,N]
        self.GT = self.GT[:self.ang, :self.ang, :, :, :, :]
        self.Expo = hf.get('Exposure_values')              # [3,N]

    def __getitem__(self, index):
        under_lfi = self.UnderLF[:, :, :, :, :, index]     # [ah,aw,h,w,3]
        normal_lfi = self.NormalLF[:, :, :, :, :, index]   # [ah,aw,h,w,3]
        over_lfi = self.OverLF[:, :, :, :, :, index]       # [ah,aw,h,w,3]
        gt_lfi = self.GT[:, :, :, :, :, index]             # [ah,aw,h,w,3]
        expo_value = self.Expo[:, index]                   # [3]

        me_lfi = np.concatenate([under_lfi, normal_lfi, over_lfi], axis=-1)
        me_lfi = me_lfi.astype(np.float32) / 65535.0       # [0,1]
        gt_lfi = gt_lfi.astype(np.float32)
        expo_value = expo_value.astype(np.float32)
        expo_value = 2 ** expo_value

        ##############################  Convert to linear hdr domain  ##############################
        me_hdr_lfi = np.concatenate([LDRtoHDR(me_lfi[:, :, :, :, 0:3], expo_value[0], gamma=self.crf_value),
                                     LDRtoHDR(me_lfi[:, :, :, :, 3:6], expo_value[1], gamma=self.crf_value),
                                     LDRtoHDR(me_lfi[:, :, :, :, 6:9], expo_value[2], gamma=self.crf_value)], axis=-1)
        # log hdr domain
        me_hdr_lfi = log_transformation_np(me_hdr_lfi, param_u=5000.)

        # Inputs (LDR + HDR)
        me_input = np.concatenate([me_lfi, me_hdr_lfi], axis=-1)     # [ah,aw,h,w,18]

        # gt: log hdr domain
        gt_lfi = log_transformation_np(gt_lfi, param_u=5000.)

        ##########################  Get input and label  ###########################
        me_input = np.transpose(me_input, [0, 1, 4, 2, 3])           # [ah,aw,18,h,w]
        gt_lfi = np.transpose(gt_lfi, [0, 1, 4, 2, 3])               # [ah,aw,3,h,w]

        ###########################  Convert to tensor  ###########################
        me_input = torch.from_numpy(me_input)
        gt_lfi = torch.from_numpy(gt_lfi)
        return me_input, gt_lfi

    def __len__(self):
        return self.UnderLF.shape[5]



#################################  Read Test data  #################################
class TestSetLoader_noGT(data.Dataset):
    def __init__(self, cfg):
        super(TestSetLoader_noGT, self).__init__()
        self.ang = cfg.ang_res
        self.crf_value = cfg.crf_gamma

        hf = h5py.File(cfg.dataset_path + '.h5')            # Input path
        self.UnderLF = hf.get('Under_Exposure_LFimgs')      # [ah,aw,h,w,3,N]
        self.UnderLF = self.UnderLF[:self.ang, :self.ang, :, :, :, :]
        self.NormalLF = hf.get('Normal_Exposure_LFimgs')    # [ah,aw,h,w,3,N]
        self.NormalLF = self.NormalLF[:self.ang, :self.ang, :, :, :, :]
        self.OverLF = hf.get('Over_Exposure_LFimgs')        # [ah,aw,h,w,3,N]
        self.OverLF = self.OverLF[:self.ang, :self.ang, :, :, :, :]
        self.Expo = hf.get('Exposure_values')               # [3,N]

    def __getitem__(self, index):
        under_lfi = self.UnderLF[:, :, :, :, :, index]     # [ah,aw,h,w,3]
        normal_lfi = self.NormalLF[:, :, :, :, :, index]   # [ah,aw,h,w,3]
        over_lfi = self.OverLF[:, :, :, :, :, index]       # [ah,aw,h,w,3]
        expo_value = self.Expo[:, index]                   # [3]

        me_lfi = np.concatenate([under_lfi, normal_lfi, over_lfi], axis=-1)
        me_lfi = me_lfi.astype(np.float32) / 65535.0       # [0,1]
        expo_value = expo_value.astype(np.float32)
        expo_value = 2 ** expo_value

        ##############################  Convert to linear hdr domain  ##############################
        me_hdr_lfi = np.concatenate([LDRtoHDR(me_lfi[:, :, :, :, 0:3], expo_value[0], gamma=self.crf_value),
                                     LDRtoHDR(me_lfi[:, :, :, :, 3:6], expo_value[1], gamma=self.crf_value),
                                     LDRtoHDR(me_lfi[:, :, :, :, 6:9], expo_value[2], gamma=self.crf_value)], axis=-1)
        # log hdr domain
        me_hdr_lfi = log_transformation_np(me_hdr_lfi, param_u=5000.)

        # Inputs (LDR + HDR)
        me_input = np.concatenate([me_lfi, me_hdr_lfi], axis=-1)     # [ah,aw,h,w,18]

        ##########################  Get input ###########################
        me_input = np.transpose(me_input, [0, 1, 4, 2, 3])           # [ah,aw,18,h,w]

        ###########################  Convert to tensor  ###########################
        me_input = torch.from_numpy(me_input)
        return me_input

    def __len__(self):
        return self.UnderLF.shape[5]