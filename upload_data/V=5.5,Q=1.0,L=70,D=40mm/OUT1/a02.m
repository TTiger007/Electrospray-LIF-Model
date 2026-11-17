%% 这节代码功能：实现图片的纵向拼接功能
clc; clear; close all;

% --- 1. 参数设置 ---
folder_path = '.';     % 图片存放的文件夹
file_ext = '*.tif';       % 图片格式
mode = 'vertical';        % 模式选择：'vertical' (纵向长图) 或 'grid' (网格 2x5)

% --- 2. 读取所有图片 ---
files = dir(fullfile(folder_path, file_ext));
num_imgs = length(files);

if num_imgs == 0
    error('未找到图片，请检查文件夹路径！');
end

fprintf('共找到 %d 张图片，正在读取...\n', num_imgs);

% 使用 Cell 数组存储图片，比矩阵更灵活
img_cell = cell(1, num_imgs);

for i = 1:num_imgs
    img_path = fullfile(folder_path, files(i).name);
    img_cell{i} = imread(img_path);
end

% --- 3. 开始拼接 ---

if strcmp(mode, 'vertical')
    % === 模式 A: 纵向拼接 (1列 10行) ===
    % cat(1, A, B) 表示在维度1（高度）上拼接
    % img_cell{:} 把所有图片展开作为参数传进去
    result_img = cat(1, img_cell{:}); 
    
    disp('已完成纵向拼接（垂直堆叠）。');

elseif strcmp(mode, 'grid')
    % === 模式 B: 网格拼接 (2行 5列) ===
    % 这种适合 10 张图。如果是其他数量，请修改下面的 [2, 5]
    % montgage 主要是用来显示的，若要生成高清大图数据，建议用 imtile (R2018b及以上)
    
    % GridSize = [行数, 列数]
    % BorderSize = 像素间距 (0表示无缝)
    result_img = imtile(folder_path, 'GridSize', [2, 5], 'BorderSize', 0);
    
    disp('已完成网格拼接 (2行 x 5列)。');
else
    error('请在 mode 参数中填入 vertical 或 grid');
end

% --- 4. 显示与保存 ---
figure; 
imshow(result_img); 
title(['拼接完成 - 模式: ', mode]);

% 保存结果
save_name = ['Merged_Result_', mode, '.tif'];
imwrite(result_img, save_name);
fprintf('图片已保存为: %s\n', save_name);