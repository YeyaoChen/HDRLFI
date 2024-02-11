clc
clear all;

h5disp('K:/HDRLF_dataset/HDRLFFNet/Dataset/trainLF.h5');

x1 = h5read('K:/HDRLF_dataset/HDRLFFNet/Dataset/trainLF.h5','/Exposure_values');
x2 = h5read('K:/HDRLF_dataset/HDRLFFNet/Dataset/trainLF.h5','/Under_Exposure_LFimgs');
x3 = h5read('K:/HDRLF_dataset/HDRLFFNet/Dataset/trainLF.h5','/Normal_Exposure_LFimgs');
x4 = h5read('K:/HDRLF_dataset/HDRLFFNet/Dataset/trainLF.h5','/Over_Exposure_LFimgs');
x5 = h5read('K:/HDRLF_dataset/HDRLFFNet/Dataset/trainLF.h5','/Label_HDR_LFimgs');

ind = 5;
an1 = 4; an2 = 4;
y1 = permute(squeeze(x2(ind,1:3,:,:,an1,an2)),[3,2,1]);
y2 = permute(squeeze(x2(ind,4:6,:,:,an1,an2)),[3,2,1]);
y3 = permute(squeeze(x2(ind,7:9,:,:,an1,an2)),[3,2,1]);
figure;
imshow(y1);
figure;
imshow(y2);
figure;
imshow(y3);
