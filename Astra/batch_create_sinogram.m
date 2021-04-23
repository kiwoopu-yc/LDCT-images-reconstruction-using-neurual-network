clc;
clear;

filepath =  'C:\Users\pyc\Desktop\head-ct-hemorrhage\head_ct\head_ct\';
outputpath = 'C:\Users\pyc\Desktop\DATA\head_ct_sinogram\';
namelist = dir([filepath,'*.png']);
l = length(namelist);
for i = 1:l
    filename{i} = [filepath,namelist(i).name];%ͨ���ַ���ƴ�ӻ�õľ��Ǿ���·����
    
    % ͼ��Ԥ���� �Ҷ� double��
    %chr = [filepath 'pic' num2str(i) '.png'];
    P= imread([filename{i}]);
    P = rgb2gray(P);
    P = imresize(P,[512 512]);
    P = im2double(P);

    % Create a basic 512*512 square volume geometry
    vol_geom = astra_create_vol_geom(512,512);

    % function astra_create_proj_geom .
    %ƽ���� ̽������С1.0 ̽��������736 1160��view
    proj_geom = astra_create_proj_geom('parallel', 1.0, 736, linspace2(0,pi,1160));

    % Create a sinogram using the GPU.
    % Note that the first time the GPU is accessed, there may be a delay
    % of up to 10 seconds for initialization.
    [sinogram_id, sinogram] = astra_create_sino_gpu(P, proj_geom, vol_geom);
    
    chr1 = [outputpath 'pic' num2str(i) '_sino.mat'];

%     figure; imshow(P, []);
%     figure; imshow(sinogram, []);
    save(chr1,'sinogram');
%    ���������Ƿ񱣴���ȷ    
%     a = load(chr1);
%     b = a.sinogram;
%     
%     figure; imshow(b,[]);
    
    % Free memory
    astra_mex_data2d('delete', sinogram_id);
end

