clc;
clear;
close all;

%% =========================
% 对高斯参数 A, mu, sigma, B 关于物理坐标 x 进行二次拟合
%
% 使用数据：
%   2_crop ~ 20_crop
%   但排除 1_crop 和 10_crop
%
% 坐标定义：
%   2_crop -> x = 2 mm
%   3_crop -> x = 3 mm
%   ...
%   20_crop -> x = 20 mm
%
% 拟合模型：
%   A     = a * exp(b*x)
%   mu    = a*x + b
%   sigma = a*(1 - exp(b*x)) + c
%   B     = a*x + b
%
% 输出：
%   1. 每个参数的拟合图
%   2. 每个参数的拟合数据 csv
%   3. 拟合系数汇总表
%   4. 各自 R^2
%% =========================

%% ---------- 路径设置 ----------
baseFolder = pwd;

% 前一步四参数高斯拟合结果汇总表
inputCSV = fullfile(baseFolder, ...
    'Gaussian_Fit_Results_WithBackground', ...
    'Gaussian_Fit_Results.csv');

% 输出文件夹
resultFolder = fullfile(baseFolder, 'Parameter_vs_X_Fit_Results');
if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end

plotFolder = fullfile(resultFolder, 'Fit_Plots');
if ~exist(plotFolder, 'dir')
    mkdir(plotFolder);
end

dataFolder = fullfile(resultFolder, 'Fit_Data');
if ~exist(dataFolder, 'dir')
    mkdir(dataFolder);
end

%% ---------- 读取参数表 ----------
if ~exist(inputCSV, 'file')
    error('找不到输入文件：%s', inputCSV);
end

T = readtable(inputCSV);

% 检查所需列
requiredVars = {'Index', 'A', 'mu', 'sigma', 'B'};
for k = 1:numel(requiredVars)
    if ~ismember(requiredVars{k}, T.Properties.VariableNames)
        error('输入表中缺少列：%s', requiredVars{k});
    end
end

%% ---------- 选取样本 ----------
% 去掉 1_crop 和 10_crop
excludeIdx = [1, 10];
useMask = ~ismember(T.Index, excludeIdx);

Tfit = T(useMask, :);

% 再按 Index 排序，保证 x 坐标顺序正确
Tfit = sortrows(Tfit, 'Index');

% 物理坐标 x（mm），这里直接令 x = Index
x = double(Tfit.Index);

% 四个待拟合参数
A_data     = double(Tfit.A);
mu_data    = double(Tfit.mu);
sigma_data = double(Tfit.sigma);
B_data     = double(Tfit.B);

%% ---------- 模型定义 ----------
% 1) A = a * exp(b*x)
modelA = @(p, x) p(1) * exp(p(2) * x);   % p = [a, b]

% 2) mu = a*x + b
modelMu = @(p, x) p(1) * x + p(2);       % p = [a, b]

% 3) sigma = a*(1 - exp(b*x)) + c
modelSigma = @(p, x) p(1) * (1 - exp(p(2) * x)) + p(3);   % p = [a, b, c]

% 4) B = a*x + b
modelB = @(p, x) p(1) * x + p(2);        % p = [a, b]

%% ---------- 优化选项 ----------
opts_curve = optimoptions('lsqcurvefit', ...
    'Display', 'off', ...
    'MaxFunctionEvaluations', 10000, ...
    'MaxIterations', 2000, ...
    'FunctionTolerance', 1e-12, ...
    'StepTolerance', 1e-12);

%% =========================================================
% 1. 拟合 A(x) = a * exp(b*x)
%% =========================================================
% 初值
p0_A = [max(A_data), -0.05];

% 边界
lb_A = [0, -Inf];
ub_A = [Inf, Inf];

try
    pFit_A = lsqcurvefit(modelA, p0_A, x, A_data, lb_A, ub_A, opts_curve);
catch ME
    error('A(x) 拟合失败：%s', ME.message);
end

A_fit = modelA(pFit_A, x);

% R^2
SS_res_A = sum((A_data - A_fit).^2);
SS_tot_A = sum((A_data - mean(A_data)).^2);
R2_A = 1 - SS_res_A / SS_tot_A;

% 稠密绘图曲线
x_dense = linspace(min(x), max(x), 300)';
A_dense = modelA(pFit_A, x_dense);

% 保存数据
A_out = [x, A_data, A_fit];
writematrix(A_out, fullfile(dataFolder, 'A_vs_x_fit.csv'));

% 绘图
fig = figure('Visible', 'off', 'Color', 'w');
plot(x, A_data, 'bo', 'MarkerSize', 7, 'LineWidth', 1.5); hold on;
plot(x_dense, A_dense, 'r-', 'LineWidth', 2);
grid on;
xlabel('x (mm)');
ylabel('A');
title('Fit of A(x) = a e^{bx}');
legend('Data', 'Fit', 'Location', 'best');

txt = sprintf(['A(x) = a e^{bx}\n' ...
               'a = %.6g\n' ...
               'b = %.6g\n' ...
               'R^2 = %.6f'], ...
               pFit_A(1), pFit_A(2), R2_A);
text(0.62*max(x), 0.80*max(A_data), txt, ...
    'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');

saveas(fig, fullfile(plotFolder, 'A_vs_x_fit.png'));
close(fig);

%% =========================================================
% 2. 拟合 mu(x) = a*x + b
%% =========================================================
% 线性模型可直接 polyfit，也可统一用 lsqcurvefit
p0_mu = polyfit(x, mu_data, 1);   % [a, b]

try
    pFit_mu = lsqcurvefit(modelMu, p0_mu, x, mu_data, [], [], opts_curve);
catch ME
    error('mu(x) 拟合失败：%s', ME.message);
end

mu_fit = modelMu(pFit_mu, x);

SS_res_mu = sum((mu_data - mu_fit).^2);
SS_tot_mu = sum((mu_data - mean(mu_data)).^2);
R2_mu = 1 - SS_res_mu / SS_tot_mu;

mu_dense = modelMu(pFit_mu, x_dense);

mu_out = [x, mu_data, mu_fit];
writematrix(mu_out, fullfile(dataFolder, 'mu_vs_x_fit.csv'));

fig = figure('Visible', 'off', 'Color', 'w');
plot(x, mu_data, 'bo', 'MarkerSize', 7, 'LineWidth', 1.5); hold on;
plot(x_dense, mu_dense, 'r-', 'LineWidth', 2);
grid on;
xlabel('x (mm)');
ylabel('\mu');
title('Fit of \mu(x) = a x + b');
legend('Data', 'Fit', 'Location', 'best');

txt = sprintf(['\\mu(x) = a x + b\n' ...
               'a = %.6g\n' ...
               'b = %.6g\n' ...
               'R^2 = %.6f'], ...
               pFit_mu(1), pFit_mu(2), R2_mu);
text(0.62*max(x), min(mu_data) + 0.75*(max(mu_data)-min(mu_data)), txt, ...
    'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');

saveas(fig, fullfile(plotFolder, 'mu_vs_x_fit.png'));
close(fig);

%% =========================================================
% 3. 拟合 sigma(x) = a*(1 - exp(b*x)) + c
%% =========================================================
% 初值建议：
% c 初值可取 sigma 最小值
% a 初值取 sigma 最大值 - sigma 最小值
% b 初值取负值，通常更合理
sigma_min = min(sigma_data);
sigma_max = max(sigma_data);

p0_sigma = [sigma_max - sigma_min, -0.05, sigma_min];

% 边界：可按需要调整
lb_sigma = [-Inf, -Inf, -Inf];
ub_sigma = [ Inf,  Inf,  Inf];

try
    pFit_sigma = lsqcurvefit(modelSigma, p0_sigma, x, sigma_data, lb_sigma, ub_sigma, opts_curve);
catch ME
    error('sigma(x) 拟合失败：%s', ME.message);
end

sigma_fit = modelSigma(pFit_sigma, x);

SS_res_sigma = sum((sigma_data - sigma_fit).^2);
SS_tot_sigma = sum((sigma_data - mean(sigma_data)).^2);
R2_sigma = 1 - SS_res_sigma / SS_tot_sigma;

sigma_dense = modelSigma(pFit_sigma, x_dense);

sigma_out = [x, sigma_data, sigma_fit];
writematrix(sigma_out, fullfile(dataFolder, 'sigma_vs_x_fit.csv'));

fig = figure('Visible', 'off', 'Color', 'w');
plot(x, sigma_data, 'bo', 'MarkerSize', 7, 'LineWidth', 1.5); hold on;
plot(x_dense, sigma_dense, 'r-', 'LineWidth', 2);
grid on;
xlabel('x (mm)');
ylabel('\sigma');
title('Fit of \sigma(x) = a(1 - e^{bx}) + c');
legend('Data', 'Fit', 'Location', 'best');

txt = sprintf(['\\sigma(x) = a(1 - e^{bx}) + c\n' ...
               'a = %.6g\n' ...
               'b = %.6g\n' ...
               'c = %.6g\n' ...
               'R^2 = %.6f'], ...
               pFit_sigma(1), pFit_sigma(2), pFit_sigma(3), R2_sigma);
text(0.55*max(x), min(sigma_data) + 0.70*(max(sigma_data)-min(sigma_data)), txt, ...
    'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');

saveas(fig, fullfile(plotFolder, 'sigma_vs_x_fit.png'));
close(fig);

%% =========================================================
% 4. 拟合 B(x) = a*x + b
%% =========================================================
p0_B = polyfit(x, B_data, 1);

try
    pFit_B = lsqcurvefit(modelB, p0_B, x, B_data, [], [], opts_curve);
catch ME
    error('B(x) 拟合失败：%s', ME.message);
end

B_fit = modelB(pFit_B, x);

SS_res_B = sum((B_data - B_fit).^2);
SS_tot_B = sum((B_data - mean(B_data)).^2);
R2_B = 1 - SS_res_B / SS_tot_B;

B_dense = modelB(pFit_B, x_dense);

B_out = [x, B_data, B_fit];
writematrix(B_out, fullfile(dataFolder, 'B_vs_x_fit.csv'));

fig = figure('Visible', 'off', 'Color', 'w');
plot(x, B_data, 'bo', 'MarkerSize', 7, 'LineWidth', 1.5); hold on;
plot(x_dense, B_dense, 'r-', 'LineWidth', 2);
grid on;
xlabel('x (mm)');
ylabel('B');
title('Fit of B(x) = a x + b');
legend('Data', 'Fit', 'Location', 'best');

txt = sprintf(['B(x) = a x + b\n' ...
               'a = %.6g\n' ...
               'b = %.6g\n' ...
               'R^2 = %.6f'], ...
               pFit_B(1), pFit_B(2), R2_B);
text(0.62*max(x), min(B_data) + 0.75*(max(B_data)-min(B_data)), txt, ...
    'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');

saveas(fig, fullfile(plotFolder, 'B_vs_x_fit.png'));
close(fig);

%% ---------- 保存拟合参数汇总 ----------
% 为了统一保存成表，这里把不同模型参数按 a,b,c 三列存
% 对于只有 a,b 的模型，c 填 NaN
summaryNames = {'A', 'mu', 'sigma', 'B'}';
coef_a = [pFit_A(1);     pFit_mu(1);     pFit_sigma(1);     pFit_B(1)];
coef_b = [pFit_A(2);     pFit_mu(2);     pFit_sigma(2);     pFit_B(2)];
coef_c = [NaN;           NaN;            pFit_sigma(3);     NaN];
R2_all = [R2_A;          R2_mu;          R2_sigma;          R2_B];

modelExpr = { ...
    'A = a*exp(b*x)'; ...
    'mu = a*x + b'; ...
    'sigma = a*(1-exp(b*x)) + c'; ...
    'B = a*x + b'};

summaryTable = table(summaryNames, modelExpr, coef_a, coef_b, coef_c, R2_all, ...
    'VariableNames', {'Parameter', 'Model', 'a', 'b', 'c', 'R2'});

summaryFile = fullfile(resultFolder, 'Parameter_vs_x_Fit_Summary.csv');
writetable(summaryTable, summaryFile);

%% ---------- 额外保存用于复现的原始输入数据 ----------
rawDataTable = table(x, A_data, mu_data, sigma_data, B_data, ...
    'VariableNames', {'x_mm', 'A', 'mu', 'sigma', 'B'});
writetable(rawDataTable, fullfile(resultFolder, 'Input_Parameter_Data_Used.csv'));

disp('参数关于 x 的拟合全部完成。');
disp(['拟合参数汇总表已保存到：', summaryFile]);
disp(['图像保存在：', plotFolder]);
disp(['拟合数据保存在：', dataFolder]);