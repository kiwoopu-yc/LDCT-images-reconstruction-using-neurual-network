clc;
clear;

filepath =  'C:\Users\pyc\Desktop\head-ct-hemorrhage\head_ct\head_ct\';
outputpath = 'C:\Users\pyc\Desktop\DATA\head_ct_sinogram\';
namelist = dir([filepath,'*.png']);
l = length(namelist);
for i = 1:l
    filename{i} = [filepath,namelist(i).name];%通过字符串拼接获得的就是绝对路径了
    
    % 图像预处理 灰度 double型
    %chr = [filepath 'pic' num2str(i) '.png'];
    P= imread([filename{i}]);
    P = rgb2gray(P);
    P = imresize(P,[512 512]);
    P = im2double(P);

    % Create a basic 512*512 square volume geometry
    vol_geom = astra_create_vol_geom(512,512);

    % function astra_create_proj_geom .
    %平行束 探测器大小1.0 探测器个数736 1160个view
    proj_geom = astra_create_proj_geom('parallel', 1.0, 736, linspace2(0,pi,1160));

    % Create a sinogram using the GPU.
    % Note that the first time the GPU is accessed, there may be a delay
    % of up to 10 seconds for initialization.
    [sinogram_id, sinogram] = astra_create_sino_gpu(P, proj_geom, vol_geom);
    
    chr1 = [outputpath 'pic' num2str(i) '_sino.mat'];

%     figure; imshow(P, []);
%     figure; imshow(sinogram, []);
    save(chr1,'sinogram');
%    测试数据是否保存正确    
%     a = load(chr1);
%     b = a.sinogram;
%     
%     figure; imshow(b,[]);
    
    % Free memory
    astra_mex_data2d('delete', sinogram_id);
end

