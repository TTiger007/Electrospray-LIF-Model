# -*- coding: utf-8 -*-
"""
Test a trained line-wise MLP surrogate model and generate paper-ready outputs.

The script keeps the original output style and additionally creates:
    paper_results_mlp_linewise/figures
    paper_results_mlp_linewise/tables
    paper_results_mlp_linewise/data

All Gaussian curves and fluorescence surfaces use the fixed pixel range 0-1024.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


GROUP_COL_CANDIDATES = ["GroupID", "Group ID", "GroupId", "groupID", "group_id", "ExpNo"]
INPUT_COLS = ["V", "Q", "l", "d", "x"]
TARGET_COLS = ["A", "mu", "sigma", "B"]
DEFAULT_DATA_PATH = "NN_dataset_valid.xlsx"
MODEL_DIR = Path("train_outputs_mlp_linewise")
OUTPUT_DIR = Path("test_outputs_mlp_linewise")
PAPER_DIR = Path("paper_results_mlp_linewise")
TEST_GROUPS = tuple(range(10, 81, 10))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG_DPI = 300
CURVE_PLOT_DPI = 300
SAVE_VECTOR_FORMATS = False  # set True manually if PDF/SVG copies are required
SAVE_GAUSSIAN_CURVES = True
PIXEL_MIN = 0.0
PIXEL_MAX = 1024.0
GAUSSIAN_CURVE_POINTS = 1025


class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Sequence[int] = (16, 16), dropout: float = 0.05):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, int(h)), nn.ReLU(), nn.Dropout(dropout)])
            prev = int(h)
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def read_table(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError("数据文件不存在: {}".format(path))
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_excel(path)
    except ImportError as e:
        raise ImportError("读取 xlsx 需要 openpyxl，请先安装 openpyxl，或将数据另存为 csv。") from e


def find_group_col(df: pd.DataFrame) -> str:
    for col in GROUP_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError("未找到实验组列。请确认表格中存在以下列名之一: {}".format(GROUP_COL_CANDIDATES))


def check_required_columns(df: pd.DataFrame, group_col: str, input_cols: Sequence[str], target_cols: Sequence[str]) -> None:
    required = [group_col] + list(input_cols) + list(target_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError("数据表缺少必要列: {}。当前列名为: {}".format(missing, list(df.columns)))


def prepare_dataframe(df: pd.DataFrame, input_cols: Sequence[str], target_cols: Sequence[str]) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    group_col = find_group_col(df)
    check_required_columns(df, group_col, input_cols, target_cols)
    numeric_cols = [group_col] + list(input_cols) + list(target_cols)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_rows = df[numeric_cols].isna().any(axis=1)
    if bad_rows.any():
        bad_idx = df.index[bad_rows].tolist()[:10]
        raise ValueError("必要列中存在无法转换为数值的单元格，示例行索引: {}".format(bad_idx))
    df[group_col] = df[group_col].astype(int)
    df = df.sort_values([group_col, "x"]).reset_index(drop=True)
    return df, group_col


def inverse_transform_matrix(Y: np.ndarray, transform_names: Sequence[str]) -> np.ndarray:
    Y_inv = Y.copy().astype(float)
    for j, t in enumerate(transform_names):
        if t == "log1p":
            Y_inv[:, j] = np.expm1(Y_inv[:, j])
    return Y_inv


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2 or np.allclose(y_true, y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = (y_pred - y_true) / denom
        mape = float(np.nanmean(np.abs(rel_err)) * 100.0)
        rmspe = float(np.sqrt(np.nanmean(rel_err ** 2)) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape_percent": mape, "rmspe_percent": rmspe, "r2": safe_r2(y_true, y_pred)}


def save_figure(fig: plt.Figure, save_path: Path, dpi: int = FIG_DPI, vector: bool = True) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if vector and SAVE_VECTOR_FORMATS:
        fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")


def ensure_paper_dirs(paper_dir: Path) -> None:
    for sub in ["figures", "tables", "data"]:
        (paper_dir / sub).mkdir(parents=True, exist_ok=True)


def make_true_pred_plot(y_true: pd.DataFrame, y_pred: pd.DataFrame, save_path: Path, title: str) -> None:
    target_cols = list(y_true.columns)
    n_targets = len(target_cols)
    ncols = 2
    nrows = int(math.ceil(n_targets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.6 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(target_cols):
        ax = axes[i]
        true_vals = y_true[col].to_numpy(dtype=float)
        pred_vals = y_pred[col].to_numpy(dtype=float)
        ax.scatter(true_vals, pred_vals, s=18, alpha=0.78, edgecolors="none")
        lo = float(np.nanmin([np.nanmin(true_vals), np.nanmin(pred_vals)]))
        hi = float(np.nanmax([np.nanmax(true_vals), np.nanmax(pred_vals)]))
        if np.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5
        ax.plot([lo, hi], [lo, hi], linewidth=1.2)
        ax.set_title(col)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.25)
        r2_val = safe_r2(true_vals, pred_vals)
        ax.text(
            0.06, 0.94, "$R^2$ = {:.4f}".format(r2_val),
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.7", alpha=0.9),
        )
    for j in range(n_targets, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def load_checkpoint(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def gaussian_curve(pixel: np.ndarray, A: float, mu: float, sigma: float, B: float) -> np.ndarray:
    sigma_safe = max(abs(float(sigma)), 1e-12)
    return float(A) * np.exp(-((pixel - float(mu)) ** 2) / (2.0 * sigma_safe ** 2)) + float(B)


def make_gaussian_comparison_data(
    row_info: pd.Series,
    true_params: pd.Series,
    pred_params: pd.Series,
    group_col: str,
    n_points: int,
) -> pd.DataFrame:
    pixel = np.linspace(PIXEL_MIN, PIXEL_MAX, n_points)
    true_intensity = gaussian_curve(pixel, true_params["A"], true_params["mu"], true_params["sigma"], true_params["B"])
    pred_intensity = gaussian_curve(pixel, pred_params["A"], pred_params["mu"], pred_params["sigma"], pred_params["B"])
    return pd.DataFrame({
        group_col: int(row_info[group_col]),
        "line_x": float(row_info["x"]),
        "pixel": pixel,
        "true_intensity": true_intensity,
        "pred_intensity": pred_intensity,
        "abs_intensity_error": np.abs(pred_intensity - true_intensity),
    })


def curve_metrics_from_df(curve_df: pd.DataFrame) -> Dict[str, float]:
    y_true = curve_df["true_intensity"].to_numpy(dtype=float)
    y_pred = curve_df["pred_intensity"].to_numpy(dtype=float)
    abs_err = np.abs(y_pred - y_true)
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.nanmean(abs_err / denom) * 100.0)
    return {
        "curve_mae": float(np.mean(abs_err)),
        "curve_rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "curve_mape_percent": mape,
        "curve_max_abs_error": float(np.max(abs_err)),
        "curve_r2": safe_r2(y_true, y_pred),
    }


def save_gaussian_curve_plot(curve_df: pd.DataFrame, save_path: Path, group_col: str, title_prefix: str = "", dpi: int = CURVE_PLOT_DPI) -> None:
    group_id = int(curve_df[group_col].iloc[0])
    x_val = curve_df["line_x"].iloc[0]
    metrics = curve_metrics_from_df(curve_df)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(curve_df["pixel"], curve_df["true_intensity"], label="True Gaussian fit", linewidth=1.7)
    ax.plot(curve_df["pixel"], curve_df["pred_intensity"], label="Predicted Gaussian fit", linewidth=1.7, linestyle="--")
    ax.set_xlim(PIXEL_MIN, PIXEL_MAX)
    ax.set_xlabel("Pixel coordinate")
    ax.set_ylabel("Fluorescence intensity")
    title = "{}Group {}, line x={}".format(title_prefix, group_id, int(x_val) if float(x_val).is_integer() else x_val)
    ax.set_title(title)
    label = "$R^2$ = {:.4f}\nRMSE = {:.3g}".format(metrics["curve_r2"], metrics["curve_rmse"])
    ax.text(0.98, 0.96, label, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.92))
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True, framealpha=0.9, facecolor="white", edgecolor="0.8")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, save_path, dpi=dpi, vector=False)
    plt.close(fig)


def save_table_figure(df: pd.DataFrame, save_path: Path, title: str, max_rows: int = 12) -> None:
    display_df = df.copy().head(max_rows)
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda v: "" if pd.isna(v) else "{:.4g}".format(float(v)))
    nrows, ncols = display_df.shape
    fig_w = max(7.0, min(14.0, 1.25 * ncols))
    fig_h = max(2.4, 0.42 * nrows + 1.25)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.28)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.82")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.92")
        else:
            cell.set_facecolor("white" if row % 2 else "0.98")
    fig.tight_layout()
    save_figure(fig, save_path, vector=False)
    plt.close(fig)


def build_surface_arrays(group_curve_df: pd.DataFrame, group_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    line_values = np.array(sorted(group_curve_df["line_x"].unique()), dtype=float)
    pixel_values = np.array(sorted(group_curve_df["pixel"].unique()), dtype=float)
    true_surface = []
    pred_surface = []
    for line_x in line_values:
        sub = group_curve_df[np.isclose(group_curve_df["line_x"].to_numpy(dtype=float), line_x)].sort_values("pixel")
        true_surface.append(sub["true_intensity"].to_numpy(dtype=float))
        pred_surface.append(sub["pred_intensity"].to_numpy(dtype=float))
    return pixel_values, line_values, np.vstack(true_surface), np.vstack(pred_surface)


def save_surface_figure(group_curve_df: pd.DataFrame, save_path: Path, group_col: str) -> None:
    group_id = int(group_curve_df[group_col].iloc[0])
    pixels, lines, true_surface, pred_surface = build_surface_arrays(group_curve_df, group_col)
    vmin = float(np.nanmin([true_surface.min(), pred_surface.min()]))
    vmax = float(np.nanmax([true_surface.max(), pred_surface.max()]))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    extent = [PIXEL_MIN, PIXEL_MAX, float(np.min(lines)), float(np.max(lines))]
    im0 = axes[0].imshow(true_surface, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
    axes[0].set_title("True fluorescence surface")
    axes[0].set_xlabel("Pixel coordinate")
    axes[0].set_ylabel("Line position x")
    im1 = axes[1].imshow(pred_surface, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted fluorescence surface")
    axes[1].set_xlabel("Pixel coordinate")
    axes[1].set_ylabel("Line position x")
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.86, label="Fluorescence intensity")
    fig.suptitle("MLP fluorescence surface reconstruction, Group {}".format(group_id), y=1.04)
    save_figure(fig, save_path, vector=False)
    plt.close(fig)


def surface_metrics(group_curve_df: pd.DataFrame) -> Dict[str, float]:
    y_true = group_curve_df["true_intensity"].to_numpy(dtype=float)
    y_pred = group_curve_df["pred_intensity"].to_numpy(dtype=float)
    abs_err = np.abs(y_pred - y_true)
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = (y_pred - y_true) / denom
        mape = float(np.nanmean(np.abs(rel_err)) * 100.0)
        rmspe = float(np.sqrt(np.nanmean(rel_err ** 2)) * 100.0)
    return {
        "n_lines": int(group_curve_df["line_x"].nunique()),
        "n_pixels_per_line": int(group_curve_df["pixel"].nunique()),
        "surface_mae": float(np.mean(abs_err)),
        "surface_mape_percent": mape,
        "surface_rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "surface_rmspe_percent": rmspe,
        "surface_max_abs_error": float(np.max(abs_err)),
        "surface_r2": safe_r2(y_true, y_pred),
    }


def choose_typical_group(surface_metrics_df: pd.DataFrame, group_col: str, requested_group: Optional[int]) -> int:
    if requested_group is not None:
        if requested_group not in set(surface_metrics_df[group_col].astype(int).tolist()):
            raise ValueError("指定的 paper_group={} 不在测试组结果中。".format(requested_group))
        return int(requested_group)
    med = float(surface_metrics_df["surface_rmse"].median())
    idx = (surface_metrics_df["surface_rmse"] - med).abs().idxmin()
    return int(surface_metrics_df.loc[idx, group_col])


def choose_typical_line(curve_metrics_df: pd.DataFrame, group_col: str, group_id: int, requested_line_x: Optional[float]) -> float:
    sub = curve_metrics_df[curve_metrics_df[group_col].astype(int) == int(group_id)].copy()
    if requested_line_x is not None:
        if not np.any(np.isclose(sub["line_x"].to_numpy(dtype=float), requested_line_x)):
            raise ValueError("指定的 paper_line_x={} 不在 Group {} 的测试结果中。".format(requested_line_x, group_id))
        return float(requested_line_x)
    med = float(sub["curve_rmse"].median())
    idx = (sub["curve_rmse"] - med).abs().idxmin()
    return float(sub.loc[idx, "line_x"])


def run_test(
    data_path: str,
    model_dir: str,
    output_dir: str,
    paper_dir: str,
    save_curves: bool = SAVE_GAUSSIAN_CURVES,
    paper_group: Optional[int] = None,
    paper_line_x: Optional[float] = None,
) -> None:
    out_dir = Path(output_dir)
    paper_path = Path(paper_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prediction_plots").mkdir(parents=True, exist_ok=True)
    ensure_paper_dirs(paper_path)
    if save_curves:
        (out_dir / "gaussian_curve_plots").mkdir(parents=True, exist_ok=True)
        (out_dir / "curve_data").mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(model_dir) / "mlp_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError("未找到模型文件: {}".format(ckpt_path))
    ckpt = load_checkpoint(ckpt_path)

    input_cols = list(ckpt.get("input_cols", INPUT_COLS))
    target_cols = list(ckpt.get("target_cols", TARGET_COLS))
    if input_cols != INPUT_COLS or target_cols != TARGET_COLS:
        print("Warning: checkpoint 中的列名与脚本默认值不同，将使用 checkpoint 中的列名。")

    df_raw = read_table(data_path)
    df, group_col = prepare_dataframe(df_raw, input_cols, target_cols)

    split_info = ckpt.get("split_info", {})
    test_groups = split_info.get("test_groups", list(TEST_GROUPS)) if isinstance(split_info, dict) else list(TEST_GROUPS)
    test_groups = [int(g) for g in test_groups]
    test_df = df[df[group_col].isin(test_groups)].copy().reset_index(drop=True)
    if test_df.empty:
        raise RuntimeError("测试集为空。请确认数据表中存在测试组: {}".format(test_groups))

    active_mask = np.array(ckpt["active_mask"], dtype=bool)
    hidden_dims = tuple(int(x) for x in ckpt["hidden_dims"])
    dropout = float(ckpt["dropout"])
    transform_names = list(ckpt["target_transforms"])

    model = SmallMLP(int(active_mask.sum()), len(target_cols), hidden_dims, dropout).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_test = test_df[input_cols].to_numpy(dtype=float)[:, active_mask]
    X_test_scaled = (X_test - np.array(ckpt["x_scaler_mean"])) / np.array(ckpt["x_scaler_scale"])
    x_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred_scaled = model(x_test_tensor).cpu().numpy()

    pred_trans = pred_scaled * np.array(ckpt["y_scaler_scale"]) + np.array(ckpt["y_scaler_mean"])
    pred_raw = inverse_transform_matrix(pred_trans, transform_names)

    y_true_df = test_df[target_cols].reset_index(drop=True)
    y_pred_df = pd.DataFrame(pred_raw, columns=target_cols)

    sample_results = test_df[[group_col] + input_cols].copy().reset_index(drop=True)
    for col in target_cols:
        sample_results["true_{}".format(col)] = y_true_df[col]
        sample_results["pred_{}".format(col)] = y_pred_df[col]
        sample_results["abs_err_{}".format(col)] = (y_pred_df[col] - y_true_df[col]).abs()
        denom = y_true_df[col].abs().replace(0, np.nan)
        sample_results["rel_err_percent_{}".format(col)] = (sample_results["abs_err_{}".format(col)] / denom) * 100.0

    per_output_metrics = []
    for col in target_cols:
        metrics = metric_dict(y_true_df[col].to_numpy(dtype=float), y_pred_df[col].to_numpy(dtype=float))
        per_output_metrics.append({"target": col, **metrics})
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_true_df[col], y_pred_df[col], s=20, alpha=0.78, edgecolors="none")
        lo = float(np.nanmin([y_true_df[col].min(), y_pred_df[col].min()]))
        hi = float(np.nanmax([y_true_df[col].max(), y_pred_df[col].max()]))
        if np.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5
        ax.plot([lo, hi], [lo, hi], linewidth=1.2)
        ax.set_title("MLP: {}".format(col))
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.06, 0.94, "$R^2$ = {:.4f}".format(metrics["r2"]),
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.7", alpha=0.9),
        )
        fig.tight_layout()
        save_figure(fig, out_dir / "prediction_plots" / "true_vs_pred_{}.png".format(col))
        plt.close(fig)

    y_true_all = y_true_df.to_numpy(dtype=float)
    y_pred_all = y_pred_df.to_numpy(dtype=float)
    summary = {
        "overall_mae_flattened": float(mean_absolute_error(y_true_all.ravel(), y_pred_all.ravel())),
        "overall_rmse_flattened": float(np.sqrt(mean_squared_error(y_true_all.ravel(), y_pred_all.ravel()))),
        "overall_r2_flattened": safe_r2(y_true_all.ravel(), y_pred_all.ravel()),
        "test_groups": sorted(test_df[group_col].unique().astype(int).tolist()),
        "n_test_samples": int(len(test_df)),
        "input_cols": input_cols,
        "target_cols": target_cols,
        "device": DEVICE,
        "pixel_range": [PIXEL_MIN, PIXEL_MAX],
    }

    copy_table = pd.concat([
        test_df[[group_col] + input_cols].reset_index(drop=True),
        y_pred_df.add_prefix("pred_"),
        y_true_df.add_prefix("true_"),
    ], axis=1)

    curve_metric_rows: List[Dict[str, float]] = []
    all_curve_parts: List[pd.DataFrame] = []
    if save_curves:
        for i in range(len(test_df)):
            row_info = test_df.loc[i, [group_col] + input_cols]
            true_params = y_true_df.loc[i, target_cols]
            pred_params = y_pred_df.loc[i, target_cols]
            curve_df = make_gaussian_comparison_data(
                row_info=row_info,
                true_params=true_params,
                pred_params=pred_params,
                group_col=group_col,
                n_points=GAUSSIAN_CURVE_POINTS,
            )
            curve_df.insert(0, "sample_index", int(i))
            all_curve_parts.append(curve_df)
            cm = curve_metrics_from_df(curve_df)
            row_metrics = {
                "sample_index": int(i),
                group_col: int(row_info[group_col]),
                "line_x": float(row_info["x"]),
            }
            for col in target_cols:
                row_metrics["true_{}".format(col)] = float(true_params[col])
                row_metrics["pred_{}".format(col)] = float(pred_params[col])
                row_metrics["abs_err_{}".format(col)] = abs(float(pred_params[col]) - float(true_params[col]))
            row_metrics.update(cm)
            curve_metric_rows.append(row_metrics)
            safe_x = str(row_info["x"]).replace(".", "p")
            save_gaussian_curve_plot(
                curve_df,
                out_dir / "gaussian_curve_plots" / "group_{:02d}_x_{}.png".format(int(row_info[group_col]), safe_x),
                group_col=group_col,
            )
        curve_metrics_df = pd.DataFrame(curve_metric_rows)
        all_curves_df = pd.concat(all_curve_parts, axis=0, ignore_index=True)
        curve_metrics_df.to_csv(out_dir / "curve_data" / "gaussian_curve_metrics_per_line.csv", index=False)
        all_curves_df.to_csv(out_dir / "curve_data" / "gaussian_curve_comparison_long.csv", index=False)
        curve_metrics_df.to_csv(paper_path / "data" / "gaussian_curve_metrics_per_line.csv", index=False)
        all_curves_df.to_csv(paper_path / "data" / "gaussian_curve_comparison_long.csv", index=False)
        summary.update({
            "gaussian_curve_points_per_line": int(GAUSSIAN_CURVE_POINTS),
            "curve_mean_mae": float(curve_metrics_df["curve_mae"].mean()),
            "curve_mean_rmse": float(curve_metrics_df["curve_rmse"].mean()),
            "curve_mean_r2": float(curve_metrics_df["curve_r2"].mean()),
        })

        surface_metric_rows: List[Dict[str, float]] = []
        for gid, group_curve_df in all_curves_df.groupby(group_col):
            sm = surface_metrics(group_curve_df)
            sm_row = {group_col: int(gid)}
            sm_row.update(sm)
            surface_metric_rows.append(sm_row)
        surface_metrics_df = pd.DataFrame(surface_metric_rows).sort_values(group_col).reset_index(drop=True)
        overall_sm = surface_metrics(all_curves_df)
        overall_row = {group_col: "All test groups"}
        overall_row.update(overall_sm)
        table2_df = pd.concat([pd.DataFrame([overall_row]), surface_metrics_df], axis=0, ignore_index=True)
        surface_metrics_df.to_csv(paper_path / "data" / "surface_metrics_by_group.csv", index=False)
        all_curves_df.rename(columns={group_col: "GroupID_for_surface" if group_col != "GroupID" else group_col}).to_csv(
            paper_path / "data" / "surface_data_long.csv", index=False
        )

        typical_group = choose_typical_group(surface_metrics_df, group_col, paper_group)
        typical_line_x = choose_typical_line(curve_metrics_df, group_col, typical_group, paper_line_x)
        group_curve_df = all_curves_df[all_curves_df[group_col].astype(int) == int(typical_group)].copy()
        save_surface_figure(
            group_curve_df,
            paper_path / "figures" / "Fig3_typical_group_{:02d}_true_pred_fluorescence_surfaces.png".format(typical_group),
            group_col=group_col,
        )
        line_curve_df = group_curve_df[np.isclose(group_curve_df["line_x"].to_numpy(dtype=float), typical_line_x)].copy()
        safe_line = str(int(typical_line_x) if float(typical_line_x).is_integer() else typical_line_x).replace(".", "p")
        save_gaussian_curve_plot(
            line_curve_df,
            paper_path / "figures" / "Fig4_typical_group_{:02d}_line_x_{}_gaussian_curve_comparison.png".format(typical_group, safe_line),
            group_col=group_col,
            title_prefix="MLP Gaussian curve comparison, ",
        )
        summary["paper_typical_group"] = int(typical_group)
        summary["paper_typical_line_x"] = float(typical_line_x)
    else:
        curve_metrics_df = pd.DataFrame()
        table2_df = pd.DataFrame()

    per_output_metrics_df = pd.DataFrame(per_output_metrics)
    table1_df = per_output_metrics_df.rename(columns={
        "target": "Parameter",
        "mae": "MAE",
        "mape_percent": "MAPE (%)",
        "rmse": "RMSE",
        "rmspe_percent": "RMSPE (%)",
        "r2": "R2",
    })[["Parameter", "MAE", "MAPE (%)", "RMSE", "RMSPE (%)", "R2"]]

    sample_results.to_csv(out_dir / "test_predictions_and_errors.csv", index=False)
    per_output_metrics_df.to_csv(out_dir / "per_output_metrics.csv", index=False)
    copy_table.to_csv(out_dir / "test_copy_ready_table.csv", index=False)
    sample_results.to_csv(paper_path / "data" / "test_predictions_and_errors.csv", index=False)

    with open(out_dir / "test_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    make_true_pred_plot(y_true_df, y_pred_df, out_dir / "true_vs_pred_all_outputs.png", "MLP test: true vs predicted")
    make_true_pred_plot(
        y_true_df,
        y_pred_df,
        paper_path / "figures" / "Fig2_test_true_vs_pred_gaussian_parameters.png",
        "MLP test: true vs predicted Gaussian parameters",
    )

    # Paper tables.
    table1_df.to_csv(paper_path / "tables" / "Table1_gaussian_parameter_test_errors.csv", index=False)
    save_table_figure(table1_df, paper_path / "tables" / "Table1_gaussian_parameter_test_errors.png", "Table 1. Test errors of Gaussian parameters")
    if save_curves and not table2_df.empty:
        table2_save = table2_df.rename(columns={
            group_col: "GroupID",
            "n_lines": "Lines",
            "surface_mae": "MAE",
            "surface_mape_percent": "MAPE (%)",
            "surface_rmse": "RMSE",
            "surface_rmspe_percent": "RMSPE (%)",
            "surface_r2": "R2",
        })[["GroupID", "Lines", "MAE", "MAPE (%)", "RMSE", "RMSPE (%)", "R2"]]
        table2_save.to_csv(paper_path / "tables" / "Table2_fluorescence_surface_error_statistics.csv", index=False)
        save_table_figure(table2_save, paper_path / "tables" / "Table2_fluorescence_surface_error_statistics.png", "Table 2. Fluorescence surface error statistics", max_rows=10)
    else:
        table2_save = pd.DataFrame()

    try:
        with pd.ExcelWriter(out_dir / "test_results.xlsx") as writer:
            sample_results.to_excel(writer, sheet_name="sample_results", index=False)
            per_output_metrics_df.to_excel(writer, sheet_name="per_output_metrics", index=False)
            pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
            copy_table.to_excel(writer, sheet_name="copy_ready_table", index=False)
            if save_curves and not curve_metrics_df.empty:
                curve_metrics_df.to_excel(writer, sheet_name="curve_metrics", index=False)
        with pd.ExcelWriter(paper_path / "paper_tables_and_data.xlsx") as writer:
            table1_df.to_excel(writer, sheet_name="Table1_param_errors", index=False)
            if not table2_save.empty:
                table2_save.to_excel(writer, sheet_name="Table2_surface_errors", index=False)
            sample_results.to_excel(writer, sheet_name="test_predictions", index=False)
            if save_curves and not curve_metrics_df.empty:
                curve_metrics_df.to_excel(writer, sheet_name="curve_metrics", index=False)
                surface_metrics_df.to_excel(writer, sheet_name="surface_metrics", index=False)
    except ImportError:
        print("Warning: openpyxl 未安装，已跳过 xlsx 文件。CSV 文件已正常保存。")

    readme = {
        "script": Path(__file__).name,
        "paper_dir": str(paper_path),
        "test_output_dir": str(out_dir),
        "generated_figures": [
            "figures/Fig2_test_true_vs_pred_gaussian_parameters.png",
            "figures/Fig3_typical_group_XX_true_pred_fluorescence_surfaces.png",
            "figures/Fig4_typical_group_XX_line_x_XX_gaussian_curve_comparison.png",
        ],
        "generated_tables": [
            "tables/Table1_gaussian_parameter_test_errors.csv/png",
            "tables/Table2_fluorescence_surface_error_statistics.csv/png",
        ],
        "pixel_range": [PIXEL_MIN, PIXEL_MAX],
        "summary": summary,
    }
    with open(paper_path / "README_paper_outputs.json", "w", encoding="utf-8") as f:
        json.dump(readme, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Paper outputs saved to: {}".format(paper_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test line-wise MLP surrogate model and generate paper outputs.")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH, help="Path to the new line-wise dataset (.xlsx or .csv).")
    parser.add_argument("--model_dir", default=str(MODEL_DIR), help="Directory containing mlp_model.pth from training.")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), help="Output directory for test results.")
    parser.add_argument("--paper_dir", default=str(PAPER_DIR), help="Output directory for publication-ready paper figures/tables.")
    parser.add_argument("--no_curves", action="store_true", help="Do not save reconstructed Gaussian-curve comparison plots/data.")
    parser.add_argument("--paper_group", type=int, default=10, help="GroupID used for Fig.3 and Fig.4. Default: 10.")
    parser.add_argument("--paper_line_x", type=float, default=10.0, help="Line x used for Fig.4. Default: 10.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test(
        data_path=args.data_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        paper_dir=args.paper_dir,
        save_curves=not args.no_curves,
        paper_group=args.paper_group,
        paper_line_x=args.paper_line_x,
    )
