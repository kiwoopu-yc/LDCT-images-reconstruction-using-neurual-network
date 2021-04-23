clear;
clc;

dataset_path = 'C:\Users\pyc\Desktop\���ݼ�\n13044778\';
output_path = 'C:\Users\pyc\Desktop\72_290_1160\mat\other_fbp_reference\';
namelist = dir([dataset_path,'*.jpeg']);
l = length(namelist);
for i = 1:l
    filename = [dataset_path,namelist(i).name];%ͨ���ַ���ƴ�ӻ�õľ��Ǿ���·����
    P = imread(filename);
    P = rgb2gray(P);
    P = imresize(P,[512 512]);
    P = im2double(P);
    chr1 = [output_path 'pic' num2str(i) '_reference.mat'];
    save(chr1,'P');
end