clc
clear all;

%% path
folder = 'K:/HDRLF_dataset/Triangular weighting/Dataset/Final_DynamicSet_version/TestSet/';
scene_inf = dir(folder);
scene_inf = scene_inf(3:end);
test_number = length(scene_inf);

save_path = 'K:/HDRLF_dataset/HDRLFINet/Dataset/';
if ~exist(save_path,'dir')
    mkdir(save_path);
end
savename = [save_path,'testLF.h5'];

%% parameters
angRes = 7;
SAI_H = 400;    
SAI_W = 600;

%% generate data
count = 0;
Under_Exposure_LFimgs = zeros(1, SAI_H, SAI_W, 3, angRes, angRes, 'uint16');   % under-exposure image
Normal_Exposure_LFimgs = zeros(1, SAI_H, SAI_W, 3, angRes, angRes, 'uint16');  
Over_Exposure_LFimgs = zeros(1, SAI_H, SAI_W, 3, angRes, angRes, 'uint16'); 
Label_HDR_LFimgs = zeros(1, SAI_H, SAI_W, 3, angRes, angRes, 'single');
Exposure_values = zeros(1, 3, 'single');

%% test dataset
for ind = 1:test_number
    in_expo1 = imread([folder,scene_inf(ind).name,'/expo1.png']);
    in_expo2 = imread([folder,scene_inf(ind).name,'/expo2.png']);
    in_expo3 = imread([folder,scene_inf(ind).name,'/expo3.png']);
    
    in_expo1 = permute(reshape(in_expo1, [angRes,SAI_H,angRes,SAI_W,3]), [2,4,5,1,3]);   % [h,w,3,ah,aw]   
    in_expo2 = permute(reshape(in_expo2, [angRes,SAI_H,angRes,SAI_W,3]), [2,4,5,1,3]);   % [h,w,3,ah,aw]    
    in_expo3 = permute(reshape(in_expo3, [angRes,SAI_H,angRes,SAI_W,3]), [2,4,5,1,3]);   % [h,w,3,ah,aw]

    in_hdri = hdrread([folder,scene_inf(ind).name,'/HDRLFimg.hdr']);
    in_hdri = permute(reshape(in_hdri, [angRes,SAI_H,angRes,SAI_W,3]), [2,4,5,1,3]);    % [h,w,3,ah,aw]
    
    in_ev = single(permute(load([folder,scene_inf(ind).name,'/exposure.txt']), [2,1]));     % [3,1]--->[1,3]
    
    count = count + 1;    
    Under_Exposure_LFimgs(count, :, :, :, :, :) = in_expo1;     % [N,h,w,3,ah,aw]
    Normal_Exposure_LFimgs(count, :, :, :, :, :) = in_expo2;   
    Over_Exposure_LFimgs(count, :, :, :, :, :) = in_expo3;      
    Label_HDR_LFimgs(count, :, :, :, :, :) = in_hdri;   
    Exposure_values(count, :) = in_ev;
    fprintf('Processing on the Scene "%s"\n', num2str(ind));
end
    
%% generate final data
Under_Exposure_LFimgs = permute(Under_Exposure_LFimgs,[1,4,3,2,6,5]);    % [N,h,w,3,ah,aw] --> [N,3,w,h,aw,ah]
Normal_Exposure_LFimgs = permute(Normal_Exposure_LFimgs,[1,4,3,2,6,5]);  % [N,h,w,3,ah,aw] --> [N,3,w,h,aw,ah]
Over_Exposure_LFimgs = permute(Over_Exposure_LFimgs,[1,4,3,2,6,5]);      % [N,h,w,3,ah,aw] --> [N,3,w,h,aw,ah]
Label_HDR_LFimgs = permute(Label_HDR_LFimgs,[1,4,3,2,6,5]);              % [N,h,w,3,ah,aw] --> [N,3,w,h,aw,ah]
Exposure_values = permute(Exposure_values,[1,2]);                        % [N,3]--->[N,3]

%% save data
if exist(savename,'file')
  fprintf('Warning: replacing existing file %s \n',savename);
  delete(savename);
end 

h5create(savename,'/Under_Exposure_LFimgs',size(Under_Exposure_LFimgs),'Datatype','uint16');
h5write(savename, '/Under_Exposure_LFimgs', Under_Exposure_LFimgs);
h5create(savename,'/Normal_Exposure_LFimgs',size(Normal_Exposure_LFimgs),'Datatype','uint16');
h5write(savename, '/Normal_Exposure_LFimgs', Normal_Exposure_LFimgs);
h5create(savename,'/Over_Exposure_LFimgs',size(Over_Exposure_LFimgs),'Datatype','uint16');
h5write(savename, '/Over_Exposure_LFimgs', Over_Exposure_LFimgs);

h5create(savename,'/Label_HDR_LFimgs',size(Label_HDR_LFimgs),'Datatype','single');
h5write(savename, '/Label_HDR_LFimgs', Label_HDR_LFimgs);
h5create(savename,'/Exposure_values',size(Exposure_values),'Datatype','single');
h5write(savename, '/Exposure_values', Exposure_values);

h5disp(savename);
