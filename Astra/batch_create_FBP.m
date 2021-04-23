clear;
clc;

%dataset_path = 'C:\Users\pyc\Desktop\head-ct-hemorrhage\head_ct\head_ct\';
dataset_path = 'C:\Users\pyc\Desktop\数据集\n13044778\';
output_path = 'C:\Users\pyc\Desktop\72_290_1160\mat\other_fbp_72\';
filepath = 'C:\Users\pyc\Desktop\iiii.jpg';
namelist = dir([dataset_path,'*.jpeg']);
l = length(namelist);
for i = 1:l
    filename = [dataset_path,namelist(i).name];%通过字符串拼接获得的就是绝对路径了
    P = imread(filename);
    P = rgb2gray(P);
    P = imresize(P,[512 512]);
    P = im2double(P);
    
    vol_geom = astra_create_vol_geom(512,512);
    proj_geom = astra_create_proj_geom('parallel', 1.0, 736, linspace2(0,pi,72));
    [sinogram_id, sinogram] = astra_create_sino_gpu(P, proj_geom, vol_geom);
    % Create a data object for the reconstruction
    rec_id = astra_mex_data2d('create', '-vol', vol_geom);

    % create configuration 
    cfg = astra_struct('FBP_CUDA');
    cfg.ReconstructionDataId = rec_id;
    cfg.ProjectionDataId = sinogram_id;
    cfg.option.FilterType = 'ram-lak';

    % possible values for FilterType:
    % none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
    % triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
    % blackman-nuttall, flat-top, kaiser, parzen
    
    % Create and run the algorithm object from the configuration structure
    alg_id = astra_mex_algorithm('create', cfg);
    astra_mex_algorithm('run', alg_id);

    % Get the result
    rec = astra_mex_data2d('get', rec_id);
    chr = [output_path 'picii' num2str(i) '.png'];
    chr1 = [output_path 'pic' num2str(i) '_fbp.mat'];
    imwrite(rec, chr);
    save(chr1,'rec');

    % Clean up. Note that GPU memory is tied up in the algorithm object,
    % and main RAM in the data objects.
    astra_mex_algorithm('delete', alg_id);
    astra_mex_data2d('delete', rec_id);
    astra_mex_data2d('delete', sinogram_id);

end