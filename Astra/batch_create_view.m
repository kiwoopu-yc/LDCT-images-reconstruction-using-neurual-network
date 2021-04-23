clear;
clc;


dataset_path = 'C:\Users\pyc\Desktop\output_path1\';
output_path = 'C:\Users\pyc\Desktop\output_path2\';

for i = 1:99
    if i<10
        str = '00'
    else
        str = '0'
    end
    P = imread([dataset_path  'pic' num2str(i) '.png']);
    %P = rgb2gray(P);
    P = imresize(P,[512 512]);
    P = im2double(P);
    
    vol_geom = astra_create_vol_geom(512,512);
    proj_geom = astra_create_proj_geom('parallel', 1.0, 736, linspace2(0,pi,72));
    [sinogram_id, sinogram] = astra_create_sino_gpu(P, proj_geom, vol_geom);
    for j=1:72
        chr = [output_path 'pic' num2str(i) '-' num2str(j),'-views.mat']
        b = sinogram(j,:)
        save(chr,'b');
    end
    astra_mex_data2d('delete', sinogram);
    astra_mex_data2d('delete', sinogram_id);
end