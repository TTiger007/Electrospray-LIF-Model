clc;
clear;
close all;

%% =========================================================
%  基于参数函数生成 2~20 mm 区间的“荧光面”
%  并与未参与拟合的 10_crop 进行比较
%
%  已知：
%    1 mm 对应 17 个像素（纵向）
%    水平像素数 = 1024
%
%  参数模型：
%    A(x)     = a * exp(b*x)
%    mu(x)    = a*x + b
%    sigma(x) = a*(1-exp(b*x)) + c
%    B(x)     = a*x + b
%
%  需要输入：
%    1) Parameter_vs_X_Fit_Results/Parameter_vs_X_Fit_Summary.csv
%    2) 10_crop_profile.csv
%% =========================================================

%% ---------- 路径设置 ----------
baseFolder = pwd;

paramSummaryFile = fullfile(baseFolder, ...
    'Parameter_vs_X_Fit_Results', ...
    'Parameter_vs_x_Fit_Summary.csv');

testProfileFile = fullfile(baseFolder, '10_crop_profile.csv');

resultFolder = fullfile(baseFolder, 'Generated_Fluorescence_Plane');
if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end

plotFolder = fullfile(resultFolder, 'Plots');
if ~exist(plotFolder, 'dir')
    mkdir(plotFolder);
end

dataFolder = fullfile(resultFolder, 'Data');
if ~exist(dataFolder, 'dir')
    mkdir(dataFolder);
end

%% ---------- 基本参数 ----------
x_start_mm = 2;
x_end_mm   = 20;
pix_per_mm = 17;       % 1 mm = 17 px
imgWidth   = 1024;     % 水平像素数

% 纵向行数（包含起止端点）
numRows = round((x_end_mm - x_start_mm) * pix_per_mm) + 1;

% 每一行对应的物理坐标
x_mm_vec = x_start_mm + (0:numRows-1)' / pix_per_mm;

% 水平像素坐标
u = (1:imgWidth);

%% ---------- 读取参数拟合结果 ----------
if ~exist(paramSummaryFile, 'file')
    error('找不到参数拟合汇总表：%s', paramSummaryFile);
end

T = readtable(paramSummaryFile);

requiredParams = {'A','mu','sigma','B'};
for k = 1:numel(requiredParams)
    if ~any(strcmp(T.Parameter, requiredParams{k}))
        error('参数汇总表中缺少参数：%s', requiredParams{k});
    end
end

% 读取 A 参数
rowA = T(strcmp(T.Parameter, 'A'), :);
aA = rowA.a;
bA = rowA.b;

% 读取 mu 参数
rowMu = T(strcmp(T.Parameter, 'mu'), :);
aMu = rowMu.a;
bMu = rowMu.b;

% 读取 sigma 参数
rowSigma = T(strcmp(T.Parameter, 'sigma'), :);
aSigma = rowSigma.a;
bSigma = rowSigma.b;
cSigma = rowSigma.c;

% 读取 B 参数
rowB = T(strcmp(T.Parameter, 'B'), :);
aB = rowB.a;
bB = rowB.b;

%% ---------- 定义参数函数 ----------
A_fun     = @(x) aA * exp(bA * x);
mu_fun    = @(x) aMu * x + bMu;
sigma_fun = @(x) aSigma * (1 - exp(bSigma * x)) + cSigma;
B_fun     = @(x) aB * x + bB;

%% ---------- 生成二维荧光面 ----------
fluorPlane = zeros(numRows, imgWidth);

A_rows = zeros(numRows, 1);
mu_rows = zeros(numRows, 1);
sigma_rows = zeros(numRows, 1);
B_rows = zeros(numRows, 1);

for r = 1:numRows
    x_now = x_mm_vec(r);

    A_now = A_fun(x_now);
    mu_now = mu_fun(x_now);
    sigma_now = sigma_fun(x_now);
    B_now = B_fun(x_now);

    % 防止数值异常
    if sigma_now <= 0 || isnan(sigma_now) || isinf(sigma_now)
        sigma_now = 1e-6;
    end

    line_now = A_now * exp(-((u - mu_now).^2) ./ (2 * sigma_now^2)) + B_now;

    % 防止负值
    line_now(line_now < 0) = 0;

    fluorPlane(r, :) = line_now;

    A_rows(r) = A_now;
    mu_rows(r) = mu_now;
    sigma_rows(r) = sigma_now;
    B_rows(r) = B_now;
end

%% ---------- 保存生成的荧光面数据 ----------
% 保存原始矩阵
writematrix(fluorPlane, fullfile(dataFolder, 'generated_fluorescence_plane.csv'));

% 保存每一行对应物理坐标及高斯参数
paramPerRowTable = table(x_mm_vec, A_rows, mu_rows, sigma_rows, B_rows, ...
    'VariableNames', {'x_mm', 'A', 'mu', 'sigma', 'B'});
writetable(paramPerRowTable, fullfile(dataFolder, 'generated_parameters_per_row.csv'));

%% ---------- 保存图像 ----------
% 为了保存成16位图像，先裁剪到 uint16 范围
fluorPlane_uint16 = uint16(min(max(round(fluorPlane), 0), 65535));

imwrite(fluorPlane_uint16, fullfile(resultFolder, 'generated_fluorescence_plane.tif'));

% 可视化 png
fig = figure('Visible', 'off', 'Color', 'w');
imagesc(fluorPlane);
axis image;
colormap(gray);
colorbar;
xlabel('Horizontal Pixel');
ylabel('Vertical Pixel');
title('Generated Fluorescence Plane (2 mm to 20 mm)');
saveas(fig, fullfile(plotFolder, 'generated_fluorescence_plane.png'));
close(fig);

%% ---------- 与 10_crop 比较 ----------
if ~exist(testProfileFile, 'file')
    error('找不到测试文件：%s', testProfileFile);
end

data10 = readmatrix(testProfileFile);
if size(data10,2) < 4
    error('10_crop_profile.csv 的列数不足4列。');
end

y_true = double(data10(:,4));
y_true = y_true(:);

% 若长度不是1024，则按实际长度修正
x_true = (1:length(y_true))';

% 生成 x = 10 mm 处的预测亮线
x_test = 10;

A_test = A_fun(x_test);
mu_test = mu_fun(x_test);
sigma_test = sigma_fun(x_test);
B_test = B_fun(x_test);

if sigma_test <= 0 || isnan(sigma_test) || isinf(sigma_test)
    sigma_test = 1e-6;
end

u_test = (1:length(y_true))';
y_pred = A_test * exp(-((u_test - mu_test).^2) ./ (2 * sigma_test^2)) + B_test;
y_pred(y_pred < 0) = 0;

%% ---------- 计算误差指标 ----------
residual = y_true - y_pred;

SS_res = sum((y_true - y_pred).^2);
SS_tot = sum((y_true - mean(y_true)).^2);
if SS_tot == 0
    R2_test = NaN;
else
    R2_test = 1 - SS_res / SS_tot;
end

RMSE_test = sqrt(mean((y_true - y_pred).^2));
MAE_test = mean(abs(y_true - y_pred));

eps_rel = max(1, 0.01 * max(y_true));
relErr = (y_true - y_pred) ./ max(y_true, eps_rel);
MeanAbsRelErr_test = mean(abs(relErr));
RelRMSE_test = sqrt(mean(relErr.^2));

%% ---------- 保存 10_crop 对比数据 ----------
compareData = [u_test, y_true, y_pred, residual, relErr];
writematrix(compareData, fullfile(dataFolder, 'compare_with_10_crop.csv'));

compareTable = table(x_test, A_test, mu_test, sigma_test, B_test, ...
    R2_test, RMSE_test, MAE_test, MeanAbsRelErr_test, RelRMSE_test, ...
    'VariableNames', {'x_mm', 'A_pred', 'mu_pred', 'sigma_pred', 'B_pred', ...
                      'R2', 'RMSE', 'MAE', 'MeanAbsRelErr', 'RelRMSE'});
writetable(compareTable, fullfile(resultFolder, 'comparison_metrics_10_crop.csv'));

%% ---------- 保存 10_crop 对比图 ----------
fig = figure('Visible', 'off', 'Color', 'w');
plot(u_test, y_true, 'b-', 'LineWidth', 1.5); hold on;
plot(u_test, y_pred, 'r--', 'LineWidth', 2);
grid on;
xlabel('Horizontal Pixel');
ylabel('Gray Intensity');
title('Comparison at x = 10 mm (10\_crop)');
legend('True 10\_crop', 'Predicted from fitted parameters', 'Location', 'best');

txt = sprintf(['x = %.2f mm\n' ...
               'A = %.4f\n' ...
               '\\mu = %.4f\n' ...
               '\\sigma = %.4f\n' ...
               'B = %.4f\n' ...
               'R^2 = %.6f\n' ...
               'RMSE = %.4f\n' ...
               'MAE = %.4f\n' ...
               'Mean|RelErr| = %.6f\n' ...
               'RelRMSE = %.6f'], ...
               x_test, A_test, mu_test, sigma_test, B_test, ...
               R2_test, RMSE_test, MAE_test, MeanAbsRelErr_test, RelRMSE_test);

xText = 0.60 * max(u_test);
yText = min(y_true) + 0.60 * (max(y_true) - min(y_true));
text(xText, yText, txt, 'FontSize', 10, ...
    'BackgroundColor', 'white', 'EdgeColor', 'black');

saveas(fig, fullfile(plotFolder, 'compare_with_10_crop.png'));
close(fig);

%% ---------- 在荧光面中标出 x=10 mm 对应行 ----------
row_test = round((x_test - x_start_mm) * pix_per_mm) + 1;

fig = figure('Visible', 'off', 'Color', 'w');
imagesc(fluorPlane);
axis image;
colormap(gray);
colorbar;
hold on;
plot([1 imgWidth], [row_test row_test], 'r--', 'LineWidth', 1.5);
xlabel('Horizontal Pixel');
ylabel('Vertical Pixel');
title('Generated Fluorescence Plane with x = 10 mm Row Marked');
saveas(fig, fullfile(plotFolder, 'generated_plane_with_10mm_marked.png'));
close(fig);

%% ---------- 输出提示 ----------
disp('========================================');
disp('荧光面生成完成，并已与 10_crop 比较。');
disp(['生成图像保存到: ', resultFolder]);
disp(['绘图保存到: ', plotFolder]);
disp(['数据保存到: ', dataFolder]);
disp('10_crop 对比指标：');
fprintf('R2 = %.6f\n', R2_test);
fprintf('RMSE = %.6f\n', RMSE_test);
fprintf('MAE = %.6f\n', MAE_test);
fprintf('Mean|RelErr| = %.6f\n', MeanAbsRelErr_test);
fprintf('RelRMSE = %.6f\n', RelRMSE_test);
disp('========================================');