import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import os
from os.path import join
from tqdm import tqdm
import time
from datetime import datetime
from collections import defaultdict
import scipy.io as scio
import imageio
import matplotlib.pyplot as plt
from model import Build_HDRLFINet
from load_dataset import TestSetLoader, TestSetLoader_noGT
from utils import mk_dir, log_transformation, writeEXR, radiance_writer, calculate_quality, lfi2mlia, CropPatches_w, MergePatches_w, inverse_log_transformation_np


#########################################################################################################
parser = argparse.ArgumentParser(description="High dynamic range light field imaging -- test mode")
parser.add_argument("--device", type=str, default='cuda:0', help="GPU setting")
parser.add_argument("--ang_res", type=int, default=7, help="Angular resolution of light field")
parser.add_argument("--crf_gamma", type=float, default=1/0.7, help="Gamma value of camera response function")
parser.add_argument("--u_law", type=float, default=5000.0, help="u value of dynamic range compressor")
parser.add_argument("--model_path", type=str, default="checkpoints", help="Checkpoints path")
parser.add_argument("--dataset_path", type=str, default="Dataset/testLF", help="Test data path")
parser.add_argument("--with_gt", type=str, default=True, help="With GT")
parser.add_argument("--output_path", type=str, default="results/", help="Output path")
parser.add_argument("--crop", type=int, default=1, help="Crop the LF image into LF patches when out of memory")
cfg = parser.parse_args()
print(cfg)


################################################################################################
def test(opt, test_loader):

    print('==>testing')
    # Pretrained model path
    pretrained_model_path = opt.model_path + '/Trained_model.pth'
    if not os.path.exists(pretrained_model_path):
        print('Pretrained model folder is not found ')

    # Load pretrained weight
    checkpoints = torch.load(pretrained_model_path)
    ckp_dict = checkpoints['model']

    ###########################################################################################
    # Build model
    print("Building HDRLFINet")
    model_test = Build_HDRLFINet(opt).to(opt.device)
    total = sum([param.nelement() for param in model_test.parameters()])
    print((total/1e6))

    print('loaded model from ' + pretrained_model_path)
    model_test_dict = model_test.state_dict()
    ckp_dict_refine = {k: v for k, v in ckp_dict.items() if k in model_test_dict}
    model_test_dict.update(ckp_dict_refine)
    model_test.load_state_dict(model_test_dict)

    # output folder
    mk_dir(opt.output_path)

    #######################################################################################
    # Test
    model_test.eval()
    with torch.no_grad():
        for idx_iter, idx_batch in enumerate(test_loader):
            if(opt.with_gt):
                start_time = datetime.now()

                #########################  Input data  #########################
                in_melf, label_hdr = idx_batch[0].to(opt.device), idx_batch[1].to(opt.device)

                ######################  Forward inference  ######################
                if (opt.crop == 1):
                    length = 128
                    crop = 20
                    in_left, in_middle, in_right = CropPatches_w(in_melf, length, crop)   # [B,ah,aw,c,h,length+crop]
                    # left: [1,ah,aw,c,h,length+crop]
                    test_hdr_left = model_test(in_left)
                    # middle [n,ah,aw,c,h,length+crop]
                    test_hdr_middle = torch.Tensor(in_middle.shape[0], in_middle.shape[1], in_middle.shape[2], 3, in_middle.shape[4], in_middle.shape[5])
                    for mi in range(in_middle.shape[0]):
                        cur_in_middle = in_middle[mi:mi+1]
                        test_hdr_middle[mi:mi+1] = model_test(cur_in_middle)
                    # right: 1,ah,aw,c,h,length+crop]
                    test_hdr_right = model_test(in_right)

                    test_hdr = MergePatches_w(test_hdr_left, test_hdr_middle, test_hdr_right, in_melf.shape[4], in_melf.shape[5], length, crop)  # [B,ah,aw,c,h,w]

                else:
                    test_hdr = model_test(in_melf)

                elapsed_time = datetime.now() - start_time

                # ###########################  Tone mapping  ###########################
                # test_tm = log_transformation(test_hdr, param_u=opt.u_law).squeeze(0).cpu().numpy()      # [ah,aw,c,h,w]
                # label_tm = log_transformation(label_hdr, param_u=opt.u_law).squeeze(0).cpu().numpy()

                ###########################  Inverse tone mapping  ###########################
                test_hdr = test_hdr.squeeze(0).cpu().numpy()                 # [ah,aw,c,h,w]
                label_hdr = label_hdr.squeeze(0).cpu().numpy()
                test_hdr_ = inverse_log_transformation_np(test_hdr, log_param=5000.)

                ########################  Calculate metrics  ########################
                obtained_psnr, obtained_ssim = calculate_quality(test_hdr, label_hdr)
                print('Test image.%d,  PSNR: %s,  SSIM: %s,  Elapsed time: %s'
                      % (idx_iter + 1, obtained_psnr, obtained_ssim, elapsed_time))

                ###########################  Save results  ##########################
                test_hdr_ = lfi2mlia(test_hdr_)
                test_hdr = lfi2mlia(test_hdr)

                imageio.imwrite(opt.output_path + str(idx_iter + 1) + '.png', (test_hdr.clip(0, 1) * 255.0).astype(np.uint8))
                radiance_writer(opt.output_path + str(idx_iter + 1) + '.hdr', test_hdr_)
                # writeEXR(test_hdr, opt.output_path + str(idx_iter + 1) + '.exr')
                # scio.savemat(opt.output_path + str(idx_iter + 1) + '.mat', {'PredHDR': test_hdr})

                file_handle = open(opt.output_path + 'quality_score.txt', mode='a')
                file_handle.write('Img.%d,  PSNR: %s,  SSIM: %s\n' % (idx_iter + 1, obtained_psnr, obtained_ssim))
                file_handle.close()

            else:
                start_time = datetime.now()
                #########################  Input data  #########################
                in_melf = idx_batch[0].to(opt.device)

                ######################  Forward inference  ######################
                if (opt.crop == 1):
                    length = 128
                    crop = 20
                    in_left, in_middle, in_right = CropPatches_w(in_melf, length, crop)  # [B,ah,aw,c,h,length+crop]
                    # left: [1,ah,aw,c,h,length+crop]
                    test_hdr_left = model_test(in_left)
                    # middle [n,ah,aw,c,h,length+crop]
                    test_hdr_middle = torch.Tensor(in_middle.shape[0], in_middle.shape[1], in_middle.shape[2], 3, in_middle.shape[4], in_middle.shape[5])
                    for mi in range(in_middle.shape[0]):
                        cur_in_middle = in_middle[mi:mi + 1]
                        test_hdr_middle[mi:mi + 1] = model_test(cur_in_middle)
                    # right: 1,ah,aw,c,h,length+crop]
                    test_hdr_right = model_test(in_right)

                    test_hdr = MergePatches_w(test_hdr_left, test_hdr_middle, test_hdr_right, in_melf.shape[4], in_melf.shape[5], length, crop)  # [B,ah,aw,c,h,w]

                else:
                    test_hdr = model_test(in_melf)

                elapsed_time = datetime.now() - start_time
                
                ##########################  Tone mapping  ##########################
                test_tm = log_transformation(test_hdr, param_u=opt.u_law).squeeze(0).cpu().numpy()
                print('Test image.%d,  Elapsed time: %s' % (idx_iter + 1, elapsed_time))

                ###########################  Save results  ##########################
                test_tm = lfi2mlia(test_tm)
                test_hdr = lfi2mlia(test_hdr.squeeze(0).cpu().numpy())

                imageio.imwrite(opt.output_path + str(idx_iter + 1) + '.png', (test_tm.clip(0, 1) * 255.0).astype(np.uint8))
                radiance_writer(opt.output_path + str(idx_iter + 1) + '.hdr', test_hdr)
                # writeEXR(test_hdr, opt.output_path + str(idx_iter + 1) + '.exr')
                scio.savemat(opt.output_path + str(idx_iter + 1) + '.mat', {'PredHDR': test_hdr})


def main(opt):
    time1 = datetime.now()
    if(opt.with_gt):
        test_set = TestSetLoader(opt)
    else:
        test_set = TestSetLoader_noGT(opt)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    print('Loaded {} test image from {}.h5'.format(len(test_loader), opt.dataset_path))
    test(opt, test_loader)

    time2 = datetime.now() - time1
    print('testing end, and taking: ', time2)


##############################################
if __name__ == '__main__':
    main(cfg)
