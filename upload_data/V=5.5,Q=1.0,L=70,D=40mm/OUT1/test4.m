clc; clear; close all;

%% =========================
%  批量处理 1_crop.tif ~ 9_crop.tif
%  输出：
%   1) 原图+中心线图：*_center.png
%   2) 灰度分布图：  *_profile.png
%   3) 灰度数据：    *_profile.csv
%% =========================

% ---------- 可调参数 ----------
halfLineWidth   = 2;    % 亮线半宽，信号窗口高度 = 2*halfLineWidth+1
bgOffset        = 6;    % 背景窗口相对亮线中心的偏移
bgHalfWidth     = 2;    % 背景窗口半宽
sigmaSmooth     = 1;    % 高斯平滑参数
centerSmoothWin = 31;   % 中心轨迹平滑窗口
profileSmoothWin = 9;   % 灰度曲线平滑窗口

% 16位图像最大灰度
max_gray_value = 65535;

% ---------- 批量处理 ----------
for n = 1:20
    
    % 输入文件名
    inputFile = sprintf('%d_crop.tif', n);
    
    % 检查文件是否存在
    if ~isfile(inputFile)
        fprintf('文件不存在，跳过：%s\n', inputFile);
        continue;
    end
    
    fprintf('正在处理：%s\n', inputFile);
    
    % 文件名
    [~, baseName, ~] = fileparts(inputFile);
    
    % 输出文件名
    centerFigFile  = [baseName, '_center.png'];
    profileFigFile = [baseName, '_profile.png'];
    csvFile        = [baseName, '_profile.csv'];
    
    %% 1) 读图
    I = imread(inputFile);
    I = uint16(I);          % 显式保持为16位
    I_double = double(I);   % 后续计算用 double
    
    [H, W] = size(I);
    
    %% 2) 高斯平滑（用于找中心）
    I_smooth = imgaussfilt(I_double, sigmaSmooth);
    
    %% 3) 逐列寻找亮线中心
    y_peak = zeros(1, W);
    for x = 1:W
        col = I_smooth(:, x);
        [~, y_peak(x)] = max(col);
    end
    
    % 中心轨迹平滑
    y_center = round(smoothdata(y_peak, 'movmean', centerSmoothWin));
    
    % 防越界
    y_center(y_center < 1) = 1;
    y_center(y_center > H) = H;
    
    %% 4) 逐列提取亮线信号及背景
    signal_raw = zeros(1, W);     % 信号窗口原始平均灰度（绝对灰度值）
    background = zeros(1, W);     % 背景窗口平均灰度
    
    for x = 1:W
        yc = y_center(x);
        
        % ---- 信号窗口 ----
        y1 = max(1, yc - halfLineWidth);
        y2 = min(H, yc + halfLineWidth);
        signal_raw(x) = mean(I_double(y1:y2, x));
        
        % ---- 上背景窗口 ----
        yb1_u = max(1, yc - bgOffset - bgHalfWidth);
        yb2_u = min(H, yc - bgOffset + bgHalfWidth);
        
        % ---- 下背景窗口 ----
        yb1_d = max(1, yc + bgOffset - bgHalfWidth);
        yb2_d = min(H, yc + bgOffset + bgHalfWidth);
        
        bg_vals = [];
        
        if yb1_u <= yb2_u
            bg_vals = [bg_vals; I_double(yb1_u:yb2_u, x)];
        end
        
        if yb1_d <= yb2_d
            bg_vals = [bg_vals; I_double(yb1_d:yb2_d, x)];
        end
        
        if ~isempty(bg_vals)
            background(x) = mean(bg_vals);
        else
            background(x) = 0;
        end
    end
    
    %% 5) 背景扣除与平滑
    signal_corrected = signal_raw - background;
    signal_corrected_smooth = smoothdata(signal_corrected, 'movmean', profileSmoothWin);
    
    %% 6) 归一化（用于画图和对比）
    signal_corrected_abs = signal_corrected / max_gray_value;
    signal_corrected_smooth_abs = signal_corrected_smooth / max_gray_value;
    
    %% 7) 保存“原图 + 中心线”图
    fig1 = figure('Visible', 'off');
    imagesc(I_double);
    colormap(gray);
    colorbar;
    axis image;
    hold on;
    plot(1:W, y_center, 'r-', 'LineWidth', 1.5);
    title(['原图及提取到的亮线中心 - ', baseName], 'Interpreter', 'none');
    xlabel('x (pixel)');
    ylabel('y (pixel)');
    
    exportgraphics(fig1, centerFigFile, 'Resolution', 300);
    close(fig1);
    
    %% 8) 保存“灰度分布”图
    fig2 = figure('Visible', 'off');
    plot(1:W, signal_corrected_abs, 'b-', 'LineWidth', 1.0); hold on;
    plot(1:W, signal_corrected_smooth_abs, 'r-', 'LineWidth', 1.5);
    legend('背景扣除后', '平滑后', 'Location', 'best');
    xlabel('x (pixel)');
    ylabel('Normalized intensity');
    title(['亮线沿水平方向的灰度分布 - ', baseName], 'Interpreter', 'none');
    grid on;
    
    exportgraphics(fig2, profileFigFile, 'Resolution', 300);
    close(fig2);
    
    %% 9) 保存 csv
    % 第1列：x坐标
    % 第2列：背景扣除后的归一化灰度
    % 第3列：平滑后的归一化灰度
    % 第4列：信号窗口原始平均灰度（绝对灰度值，0~65535）
    result = [(1:W).', ...
              signal_corrected_abs(:), ...
              signal_corrected_smooth_abs(:), ...
              signal_raw(:)];
    
    fid = fopen(csvFile, 'w');
    fprintf(fid, 'x_pixel,signal_corrected_abs,signal_corrected_smooth_abs,signal_raw_abs_gray\n');
    fclose(fid);
    
    dlmwrite(csvFile, result, '-append');
    
    fprintf('已保存：%s, %s, %s\n', centerFigFile, profileFigFile, csvFile);
end

fprintf('\n全部处理完成。\n');