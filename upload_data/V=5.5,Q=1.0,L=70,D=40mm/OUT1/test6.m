%% 合并9张图像，采用第2种合并方式。
%把 9 张图片按你指定的相对位移放到同一个结果图里，尽量保留原始像素值；
%对于没有重叠的地方，直接保留原图像素；
%对于重叠的地方，用 max 方式融合，也就是谁更亮就保留谁。

clc; 
clear; 
close all;

%%1) 文件列表
files = {
    
    '1_crop.tif'
    '2_crop.tif'
    '3_crop.tif'
    '4_crop.tif'
    '5_crop.tif'
    '6_crop.tif'
    '7_crop.tif'
    '8_crop.tif'
    '9_crop.tif'
    '10_crop.tif'
    '11_crop.tif'
    '12_crop.tif'
    '13_crop.tif'
    '14_crop.tif'
    '15_crop.tif'
    % '16_crop.tif'
    % '17_crop.tif'
    % '18_crop.tif'
    % '19_crop.tif'
    % '20_crop.tif'
};

N = numel(files);
%%2) 读取第一张，确定数据类型
img0 = imread(files{1});
if ndims(img0) == 3
    img0 = rgb2gray(img0);
end
imgClass = class(img0);

%%3) 读取全部图像
imgs = cell(N,1);
h = zeros(N,1);
w = zeros(N,1);

for k = 1:N
    img = imread(files{k});
    if ndims(img) == 3
        img = rgb2gray(img);
    end
    
    if ~strcmp(class(img), imgClass)
        img = cast(img, imgClass);
    end
    
    imgs{k} = img;
    [h(k), w(k)] = size(img);
end

%%4) 设置每张图相对第一张图的位置
% 第一张图作为参考，位置设为 (0,0)
% 下面举例：每张图在 y 方向依次下移 20 像素
dx = zeros(1, N);
dy = 0:17:(N-1)*17;
% dx = [0 0 0 0 0 0 0 0 0];
% dy = [0 17 40 60 80 100 120 140 160];

% 如果你的 9 张图不是这种规律，也可以手动写：
% dx = [0 3 -2 1 0 4 -1 2 0];
% dy = [0 20 41 60 79 100 121 140 159];

%%5) 计算最终画布大小
xStart = zeros(N,1);
yStart = zeros(N,1);
xEnd   = zeros(N,1);
yEnd   = zeros(N,1);

for k = 1:N
    xStart(k) = 1 + dx(k);
    yStart(k) = 1 + dy(k);
    xEnd(k)   = xStart(k) + w(k) - 1;
    yEnd(k)   = yStart(k) + h(k) - 1;
end

xmin = min(xStart);
ymin = min(yStart);
xmax = max(xEnd);
ymax = max(yEnd);

newW = xmax - xmin + 1;
newH = ymax - ymin + 1;

% 转成新画布中的坐标
xStartNew = xStart - xmin + 1;
yStartNew = yStart - ymin + 1;

%%6) 初始化结果图和有效掩膜
merged = zeros(newH, newW, imgClass);
hasData = false(newH, newW);

%%7) 逐张融合
for k = 1:N
    img = imgs{k};
    [hk, wk] = size(img);
    
    xs = xStartNew(k);
    ys = yStartNew(k);
    xe = xs + wk - 1;
    ye = ys + hk - 1;
    
    temp = zeros(newH, newW, imgClass);
    temp(ys:ye, xs:xe) = img;
    
    % 只把非零像素当作有效区域
    valid = temp > 0;
    
    onlyNew = valid & ~hasData;
    overlap = valid & hasData;
    
    % 非重叠区：直接保留原像素
    merged(onlyNew) = temp(onlyNew);
    
    % 重叠区：取较亮值
    merged(overlap) = cast( ...
        max(double(merged(overlap)), double(temp(overlap))), ...
        imgClass);
    
    % 更新已有数据区域
    hasData = hasData | valid;
end

%%8) 显示与保存
figure;
imshow(merged, []);
title('15张图合并结果');

imwrite(merged, 'merged_15_images.tif');
