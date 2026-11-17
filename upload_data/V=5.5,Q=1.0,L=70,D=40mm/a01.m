%% 确定截取的坐标范围
% --- 坐标获取专用工具 (修正版) ---
clc; clear; close all;

% 1. 选择一张最典型的照片
[filename, pathname] = uigetfile({'*.jpg;*.png;*.tif'}, '选择一张用于定位的照片');
if isequal(filename, 0), return; end

img = imread(fullfile(pathname, filename));

% 2. 画框
figure;
imshow(img);
title('【重要】请画出最终确定的裁剪框，双击确认', 'Color', 'r', 'FontSize', 12);

% --- 兼容旧版本的最大化窗口代码 ---
% 将窗口设置铺满屏幕，替代 maximize(gcf)
set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]); 
% ---------------------------------

% 3. 获取坐标 (rect 包含了 [x, y, 宽, 高])
% 程序会在这里暂停，等待您画框并双击
[~, rect] = imcrop; 

% 4. 关闭窗口并输出结果
close(gcf);

% 5. 打印结果供您复制
fprintf('\n================================================\n');
fprintf('请复制下面这行代码到您的批量处理程序中：\n\n');
fprintf('fixed_rect = [%.1f, %.1f, %.1f, %.1f];\n', rect(1), rect(2), rect(3), rect(4));
fprintf('\n================================================\n');

% 6. (可选) 验证一下效果
figure; imshow(img);
hold on;
rectangle('Position', rect, 'EdgeColor', 'g', 'LineWidth', 2);
title('绿框是您锁定的区域，确认无误即可');

%% 该代码功能：根据确定的坐标实现filename中图片的裁剪，并且将裁剪后的图片存入OUT1文件夹中。
clc; 
clear; % 清空工作区，防止变量干扰

% --- 1. 设置文件和参数 ---
filename = '20.tif';       % 原始文件名
output_folder = 'OUT1';   % 输出文件夹名称

% 检查文件夹是否存在，如果不存在则自动创建
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
    disp(['已创建文件夹: ', output_folder]);
end

% --- 2. 读取与裁剪 (您提供的逻辑) ---
img = imread(filename);

% 定义裁剪区域 [x, y, width, height]
% 这里填入您之前确定的那个 fixed_rect 数值
rect = [0.5, 181.5, 1024.0, 63.0]; 

% 裁剪
cropped_img = imcrop(img, rect);

% --- 3. 保存到 OUT1 文件夹 ---
% 构造新的保存路径。使用 fullfile 可以自动处理斜杠，不用担心路径报错
% 这里的命名规则是：在 OUT1 文件夹下，保存为 "1_crop.tif"
[~, name, ext] = fileparts(filename); % 提取文件名 "1" 和后缀 ".tif"
new_filename = [name, '_crop', ext];  % 拼成 "1_crop.tif"
save_path = fullfile(output_folder, new_filename); % 拼成 "OUT1\1_crop.tif"

% 保存图片
% 如果是科研数据，TIFF 格式默认支持无损保存
imwrite(cropped_img, save_path);

fprintf('处理完成！图片已保存至: %s\n', save_path);