clc;
clear;
close all;

%% =========================
% 批量高斯拟合（含背景项 B）
% 模型：y = A * exp(-(x-mu)^2/(2*sigma^2)) + B
% 数据：1_crop_profile.csv ~ 20_crop_profile.csv
% 第4列为灰度值
%% =========================

% 数据文件夹
dataFolder = pwd;   % 可改成你的实际路径

% 输出文件夹
resultFolder = fullfile(dataFolder, 'Gaussian_Fit_Results_WithBackground');
if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end

plotFolder = fullfile(resultFolder, 'Fit_Plots');
if ~exist(plotFolder, 'dir')
    mkdir(plotFolder);
end

fitDataFolder = fullfile(resultFolder, 'Fit_Curve_Data');
if ~exist(fitDataFolder, 'dir')
    mkdir(fitDataFolder);
end

% 文件数量
numFiles = 20;

% 结果存储
results = cell(numFiles, 9);

%% 高斯模型（含背景项）
gaussFun = @(p, x) p(1) * exp(-((x - p(2)).^2) ./ (2 * p(3)^2)) + p(4);
% p(1)=A, p(2)=mu, p(3)=sigma, p(4)=B

%% 优化选项
opts = optimoptions('lsqcurvefit', ...
    'Display', 'off', ...
    'MaxFunctionEvaluations', 10000, ...
    'MaxIterations', 2000, ...
    'FunctionTolerance', 1e-12, ...
    'StepTolerance', 1e-12);

%% 循环处理每个csv文件
for i = 1:numFiles
    
    fileName = sprintf('%d_crop_profile.csv', i);
    filePath = fullfile(dataFolder, fileName);
    
    if ~exist(filePath, 'file')
        warning('文件不存在：%s，跳过。', fileName);
        continue;
    end
    
    %% 读取数据
    data = readmatrix(filePath);
    
    if size(data, 2) < 4
        warning('文件 %s 列数不足4列，跳过。', fileName);
        continue;
    end
    
    y = double(data(:, 4));
    y = y(:);
    x = (1:length(y))';
    
    % 去除 NaN
    validIdx = ~isnan(x) & ~isnan(y);
    x = x(validIdx);
    y = y(validIdx);
    
    if isempty(y) || all(y == 0)
        warning('文件 %s 数据为空或全零，跳过。', fileName);
        continue;
    end
    
    %% 初值估计
    yMin = min(y);
    yMax = max(y);
    
    % 背景项初值：取两端平均更稳
    nEdge = max(10, round(0.05 * length(y)));
    B0 = mean([y(1:nEdge); y(end-nEdge+1:end)]);
    
    % 峰值振幅初值
    A0 = yMax - B0;
    if A0 <= 0
        A0 = yMax;
    end
    
    % 峰位置初值
    [~, idxMax] = max(y);
    mu0 = x(idxMax);
    
    % sigma 初值：用半高宽粗估
    halfLevel = B0 + A0 / 2;
    idxHalf = find(y >= halfLevel);
    if numel(idxHalf) >= 2
        FWHM0 = x(idxHalf(end)) - x(idxHalf(1));
        sigma0 = FWHM0 / 2.355;
    else
        sigma0 = 50;
    end
    
    if isnan(sigma0) || sigma0 <= 0
        sigma0 = 50;
    end
    
    p0 = [A0, mu0, sigma0, B0];
    
    %% 参数边界
    % A >= 0
    % mu 在像素范围内
    % sigma > 0
    % B >= 0
    lb = [0, min(x), 1e-6, 0];
    ub = [Inf, max(x), Inf, Inf];
    
    %% 拟合
    try
        pFit = lsqcurvefit(gaussFun, p0, x, y, lb, ub, opts);
    catch ME
        warning('文件 %s 拟合失败：%s', fileName, ME.message);
        continue;
    end
    
    %% 拟合结果
    A_fit = pFit(1);
    mu_fit = pFit(2);
    sigma_fit = pFit(3);
    B_fit = pFit(4);
    
    y_fit = gaussFun(pFit, x);
    
    %% 计算 R^2
    SS_res = sum((y - y_fit).^2);
    SS_tot = sum((y - mean(y)).^2);
    
    if SS_tot == 0
        R2 = NaN;
    else
        R2 = 1 - SS_res / SS_tot;
    end
    
    %% 计算误差指标
    RMSE = sqrt(mean((y - y_fit).^2));
    MAE = mean(abs(y - y_fit));
    
    %% 保存拟合曲线数据
    % 第1列：x
    % 第2列：原始 y
    % 第3列：拟合 y_fit
    % 第4列：残差 y-y_fit
    fitCurveData = [x, y, y_fit, (y - y_fit)];
    fitCurveFile = fullfile(fitDataFolder, sprintf('%d_fit_curve.csv', i));
    writematrix(fitCurveData, fitCurveFile);
    
    %% 绘图保存
    fig = figure('Visible', 'off', 'Color', 'w');
    plot(x, y, 'b-', 'LineWidth', 1.5); hold on;
    plot(x, y_fit, 'r--', 'LineWidth', 2);
    grid on;
    xlabel('Horizontal Pixel');
    ylabel('Gray Intensity');
    title(sprintf('Gaussian Fit with Background of %s', fileName), 'Interpreter', 'none');
    legend('Original Profile', 'Gaussian Fit', 'Location', 'best');
    
    txt = sprintf(['A = %.4f\n' ...
                   '\\mu = %.4f\n' ...
                   '\\sigma = %.4f\n' ...
                   'B = %.4f\n' ...
                   'R^2 = %.6f\n' ...
                   'RMSE = %.4f\n' ...
                   'MAE = %.4f'], ...
                   A_fit, mu_fit, sigma_fit, B_fit, R2, RMSE, MAE);
    
    xText = 0.62 * max(x);
    yText = 0.68 * max(y);
    text(xText, yText, txt, 'FontSize', 10, ...
        'BackgroundColor', 'white', 'EdgeColor', 'black');
    
    saveas(fig, fullfile(plotFolder, sprintf('%d_gaussian_fit.png', i)));
    close(fig);
    
    %% 保存结果
    distance_mm = (i - 1) * 1;   % 若第1张对应 0 mm
    
    results{i, 1} = fileName;
    results{i, 2} = i;
    results{i, 3} = distance_mm;
    results{i, 4} = A_fit;
    results{i, 5} = mu_fit;
    results{i, 6} = sigma_fit;
    results{i, 7} = B_fit;
    results{i, 8} = R2;
    results{i, 9} = RMSE;
    
    fprintf('已完成拟合：%s | A=%.3f, mu=%.3f, sigma=%.3f, B=%.3f, R2=%.5f\n', ...
        fileName, A_fit, mu_fit, sigma_fit, B_fit, R2);
end

%% 汇总结果保存
resultsTable = cell2table(results, ...
    'VariableNames', {'FileName', 'Index', 'Distance_mm', 'A', 'mu', 'sigma', 'B', 'R2', 'RMSE'});

summaryFile = fullfile(resultFolder, 'Gaussian_Fit_Results.csv');
writetable(resultsTable, summaryFile);

disp('全部处理完成。');
disp(['结果汇总已保存到：', summaryFile]);