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
import imageio
import matplotlib.pyplot as plt
from model import Build_HDRLFINet
from load_dataset import TrainSetLoader
from loss import get_loss
from utils import mk_dir, log_transformation, lfi2mlia


#########################################################################################################
parser = argparse.ArgumentParser(description="High dynamic range light field imaging -- train mode")
parser.add_argument("--device", type=str, default='cuda:0', help="GPU setting")
parser.add_argument("--model_dir", type=str, default="models/", help="Checkpoints path")
parser.add_argument("--dataset_path", type=str, default="Dataset/trainLF", help="Training data path")
parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
parser.add_argument("--ang_res", type=int, default=7, help="Angular resolution of light field")
parser.add_argument("--crf_gamma", type=float, default=1/0.7, help="Gamma value of camera response function")
parser.add_argument("--u_law", type=float, default=5000.0, help="u value of dynamic range compressor")
parser.add_argument("--patch_size", type=int, default=128, help="Training patch size")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--train_epoch", type=int, default=500, help="Number of epochs to train")
parser.add_argument("--n_steps", type=int, default=50, help="Number of epochs to update learning rate")
parser.add_argument("--decay_value", type=float, default=0.5, help="Learning rate decaying factor")
parser.add_argument("--l1_weight", type=float, default=1.0, help="Weight of L1 loss")
parser.add_argument("--ssim_weight", type=float, default=0, help="Weight of ssim loss")
parser.add_argument("--perceptual_weight", type=float, default=0, help="Weight of perceptual loss")
parser.add_argument("--detail_weight", type=float, default=1.0, help="Weight of detail loss")
parser.add_argument("--epi_weight", type=float, default=0, help="The weight of EPI gradient loss")
parser.add_argument("--resume_epoch", type=int, default=0, help="Resume from checkpoint epoch")
parser.add_argument("--num_save", type=int, default=1, help="Number of epochs for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epochs for saving loss figure")
parser.add_argument("--train_save", type=int, default=1, help="Save the image in training")
cfg = parser.parse_args()
print(cfg)

#####################################################################################################
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# Weight initialization
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

# Loss functions
loss_all = get_loss(cfg)
print(loss_all)


###############################################################################################
def train(opt, train_loader):

    print('==>training')
    start_time = datetime.now()

    train_results_dir = 'training_results/'
    if opt.train_save:
        mk_dir(train_results_dir)

    # model save folder
    mk_dir(opt.model_dir)

    #######################################################################################
    # Build model
    print("Building HDRLFINet")
    model_train = Build_HDRLFINet(opt).to(opt.device)
    # Initialize weight
    model_train.apply(weights_init_xavier)
    # for para_name in model_train.state_dict():  # print trained parameters
    #     print(para_name)
    total = sum([param.nelement() for param in model_train.parameters()])
    print((total/1e6))


    #######################################################################################
    # Optimizer and loss logger
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_train.parameters()), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.n_steps, gamma=opt.decay_value)
    losslogger = defaultdict(list)

    #######################################################################################
    # Reload previous parameters
    if opt.resume_epoch:
        resume_path = join(opt.model_dir, 'model_epoch_{}.pth'.format(opt.resume_epoch))
        if os.path.isfile(resume_path):
            print("==>Loading model parameters '{}'".format(resume_path))
            checkpoints = torch.load(resume_path)
            model_train.load_state_dict(checkpoints['model'])
            optimizer.load_state_dict(checkpoints['optimizer'])
            scheduler.load_state_dict(checkpoints['scheduler'])
            losslogger = checkpoints['losslogger']
        else:
            print("==> no model found at 'epoch{}'".format(opt.resume_epoch))

    epoch_state = opt.resume_epoch + 1
    for idx_epoch in range(epoch_state, opt.train_epoch + 1):   # epochs
        # Train
        model_train.train()
        print('Current epoch learning rate: %e' % (optimizer.state_dict()['param_groups'][0]['lr']))
        loss_epoch = 0.      # Total loss per epoch

        for ic in range(15):     # 15 cycles per epoch
            for idx_iter, idx_batch in enumerate(train_loader):
                in_melf, label_hdr = idx_batch[0].to(opt.device), idx_batch[1].to(opt.device)    # [b,ah,aw,c,h,w]

                ##########################  Forward inference  ##########################
                train_hdr = model_train(in_melf)     # [b,ah,aw,c,h,w]

                #############################  Tone mapping  #############################
                # train_tm = log_transformation(train_hdr, param_u=opt.u_law)
                # label_tm = log_transformation(label_hdr, param_u=opt.u_law)

                #############################  Calculate loss  #############################
                # l1_loss = opt.l1_weight * loss_all['l1_loss'](train_tm, label_tm)
                # epi_loss = opt.epi_weight * loss_all['epi_loss'](train_tm, label_tm)
                # loss = l1_loss + epi_loss
                l1_loss = opt.l1_weight * loss_all['l1_loss'](train_hdr, label_hdr)
                detail_loss = opt.detail_weight * loss_all['detail_loss'](train_hdr, label_hdr)
                loss = l1_loss + detail_loss

                # Cumulative loss
                loss_epoch += loss.item()

                #######################  Backward and optimize  #######################
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #####################  Save training results  #####################
                if opt.train_save:
                    if (ic + 1) % 5 == 0 and idx_iter == len(train_loader)-1:
                        in_name1 = '{}/epoch{}_cyc{}_input1.jpg'.format(train_results_dir, idx_epoch, ic + 1)
                        in_name2 = '{}/epoch{}_cyc{}_input2.jpg'.format(train_results_dir, idx_epoch, ic + 1)
                        in_name3 = '{}/epoch{}_cyc{}_input3.jpg'.format(train_results_dir, idx_epoch, ic + 1)
                        infer_name = '{}/epoch{}_cyc{}_infer.jpg'.format(train_results_dir, idx_epoch, ic + 1)
                        label_name = '{}/epoch{}_cyc{}_label.jpg'.format(train_results_dir, idx_epoch, ic + 1)

                        # [ah,aw,c,h,w]
                        save_in1 = (in_melf[0, :, :, 0:3, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                        save_in2 = (in_melf[0, :, :, 3:6, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                        save_in3 = (in_melf[0, :, :, 6:9, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                        save_infer = (train_hdr[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                        save_label = (label_hdr[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)

                        # [ah,aw,3,h,w] --> [h*ah,w*aw,3]
                        save_in1 = lfi2mlia(save_in1)
                        save_in2 = lfi2mlia(save_in2)
                        save_in3 = lfi2mlia(save_in3)
                        save_infer = lfi2mlia(save_infer)
                        save_label = lfi2mlia(save_label)

                        imageio.imwrite(in_name1, save_in1.astype(np.uint8))
                        imageio.imwrite(in_name2, save_in2.astype(np.uint8))
                        imageio.imwrite(in_name3, save_in3.astype(np.uint8))
                        imageio.imwrite(infer_name, save_infer.astype(np.uint8))
                        imageio.imwrite(label_name, save_label.astype(np.uint8))

        scheduler.step()

        ####################################  Print loss  ####################################
        losslogger['epoch'].append(idx_epoch)
        losslogger['loss'].append(loss_epoch/len(train_loader))
        elapsed_time = datetime.now() - start_time
        print('Training==>>Epoch: %d,  loss: %s,  elapsed time: %s'
              % (idx_epoch, loss_epoch/len(train_loader), elapsed_time))

        # write loss
        file_handle = open('loss.txt', mode='a')
        file_handle.write('epoch: %d,  loss: %s,  elapsed time: %s\n'
                          % (idx_epoch, loss_epoch/len(train_loader), elapsed_time))
        file_handle.close()

        # save trained model's parameters
        if idx_epoch % opt.num_save == 0:
            model_save_path = join(opt.model_dir, "model_epoch_{}.pth".format(idx_epoch))
            state = {'epoch': idx_epoch, 'model': model_train.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'losslogger': losslogger}
            torch.save(state, model_save_path)
            print("checkpoints saved to {}".format(model_save_path))

        # save loss figure
        if idx_epoch % opt.num_snapshot == 0:
            plt.figure()
            plt.title('loss')
            plt.plot(losslogger['epoch'], losslogger['loss'])
            plt.savefig(opt.model_dir + "loss.png")
            plt.close('all')


def main(opt):
    train_set = TrainSetLoader(opt)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
    print('Loaded {} training image from {}.h5'.format(len(train_loader), opt.dataset_path))
    train(opt, train_loader)


##############################################
if __name__ == '__main__':
    main(cfg)
