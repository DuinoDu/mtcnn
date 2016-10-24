addpath(genpath('/home/duino/project/matlab/toolbox/channels'));

%img = imread('/home/duino/project/mtcnn/images/test1.jpg');
%hs = 207;
%ws = 270;
im_data = (imResample(single(img), [hs ws], 'bilinear')-127.5)*0.0078125;
