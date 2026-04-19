#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fiber2star_v7: K-mode fiber tracking from NN prediction volumes.

输入：NN 纤维概率 MRC（0–255）。束内方向高度一致、较直；全局可能有 1 或 2 个模式
方向（5% 左右的断层图有两组近正交微管束搭在一起）。K-mode 让双向场景不再把方向
平均成中间向量。
相比于v6，对18002处理时间从53.1s加速到21.4s。

管线
────────────────────────────────────────────────────────────────────────────
  1. Load & normalize
  2. Z-axis Gaussian smooth           (GPU)
  3. K-mode direction detection       (GPU)  # 高阈值 CC 的 PCA + 加权外积 eigh
  4. Binary mask (thr + ball open)    (GPU)
  5. Multi-mode elongation filter     (GPU)  # 对齐任一模式即保留
  6. Skeletonize → branch → mode 指派 (CPU skel, 其余 GPU)
  7. Per-mode greedy B-spline + erase (CPU)  # segments 沿 d 排序的滑窗搜索
  8. Sample + write .star             (所有模式合并)

GPU 驻留策略
────────────────────────────────────────────────────────────────────────────
  skeletonize（skimage 无 GPU 版）前后各 D2H/H2D 一次，其余常驻 cp.ndarray。
  filter 的 length/perp_span 和 branch 方向 PCA 都走 cupyx.scipy.ndimage 的
  segment-reduce + batched eigh，避免 Python for-CC。
"""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    action="ignore",
    message=r".*cupyx\.jit\.rawkernel is experimental.*",
    category=FutureWarning,
    module="cupyx.jit._interface",
)

import argparse
import math
import re
import time
import traceback
from glob import glob, has_magic
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import mrcfile
import starfile
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian1d
from cupyx.scipy.ndimage import convolve as cp_convolve
from cupyx.scipy.ndimage import sum_labels as cp_sum_labels
from cupyx.scipy.ndimage import maximum as cp_max_labels
from cupyx.scipy.ndimage import minimum as cp_min_labels
from cupyx.scipy.spatial import KDTree as cp_KDTree
from cucim import skimage as cskimage

from skimage.morphology import skeletonize  # cucim 尚无 3D 骨架化


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────


def load_mrc(path: Path) -> Tuple[np.ndarray, float]:
    with mrcfile.open(path, permissive=True, mode="r") as m:
        vol = np.asarray(m.data, dtype=np.float32)
        try:
            vs = float(m.voxel_size.x)
        except Exception:
            vs = 1.0
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    return vol, vs


def save_mrc(path: Path, data, voxel_size: float = 1.0) -> None:
    arr = cp.asnumpy(data) if isinstance(data, cp.ndarray) else np.asarray(data)
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(arr.astype(np.float32))
        try:
            m.voxel_size = float(voxel_size)
        except Exception:
            pass


def write_star(
    out_path: Path,
    coords_xyz: np.ndarray,
    angles: np.ndarray,
    shift_z: float = 0,
    cc_ids: Optional[np.ndarray] = None,
) -> None:
    c = np.asarray(coords_xyz, dtype=np.float32)
    a = np.asarray(angles, dtype=np.float32)
    data = {
        "rlnCoordinateX": c[:, 0],
        "rlnCoordinateY": c[:, 1],
        "rlnCoordinateZ": c[:, 2] + shift_z,
        "rlnAngleRot":    a[:, 0],
        "rlnAngleTilt":   a[:, 1],
        "rlnAnglePsi":    a[:, 2],
    }
    if cc_ids is not None:
        data["aisCcId"] = np.asarray(cc_ids, dtype=np.int32)
    df = pd.DataFrame(data)
    starfile.write({"particles": df}, out_path, overwrite=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: GPU-resident batched label-wise PCA
# ─────────────────────────────────────────────────────────────────────────────


def _label_sorted_coords_gpu(labels_gpu: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, int]:
    """
    前景体素按 label 排序返回（后续 cupyx segment-reduce 需要同 label 点连续存放）。

    存储用 float32 / int32：坐标上界远小于 int32，精度对几何量足够；大幅减内存峰值
    （11.5M 前景点位时 float64 vs float32 差 138 MB）。PCA 做二阶矩时再局部 upcast。
    """
    n_lab = int(labels_gpu.max().get())
    if n_lab == 0:
        return (cp.empty((0, 3), cp.float32), cp.empty((0,), cp.int32), 0)
    fg_zyx = cp.argwhere(labels_gpu > 0).astype(cp.int32)
    if len(fg_zyx) == 0:
        return (cp.empty((0, 3), cp.float32), cp.empty((0,), cp.int32), n_lab)
    fg_labs = labels_gpu[fg_zyx[:, 0], fg_zyx[:, 1], fg_zyx[:, 2]].astype(cp.int32)
    fg_xyz = fg_zyx[:, [2, 1, 0]].astype(cp.float32)
    del fg_zyx
    sort_idx = cp.argsort(fg_labs)
    return fg_xyz[sort_idx], fg_labs[sort_idx], n_lab


def _batched_label_pca(
    fg_xyz: cp.ndarray,
    fg_labs: cp.ndarray,
    n_lab: int,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    每 label 的 (count, mean_xyz, evals, evecs)。两遍法：先求 mean，再以 centered
    二阶矩估协方差，避免 E[xx]−E[x]² 在大均值小方差下的 catastrophic 消减。
    每 (a,b) pair 只 upcast 到 float64 做 (N,) 乘法 + sum_labels，峰值仅 (N,) f64。
    """
    if n_lab == 0 or len(fg_xyz) == 0:
        return (cp.zeros((0,), cp.float64),
                cp.zeros((0, 3), cp.float64),
                cp.zeros((0, 3), cp.float64),
                cp.zeros((0, 3, 3), cp.float64))
    indices = cp.arange(1, n_lab + 1, dtype=cp.int32)

    ones = cp.ones(len(fg_xyz), dtype=cp.float64)
    count = cp_sum_labels(ones, fg_labs, indices)
    count_safe = cp.maximum(count, 1.0)

    # Pass 1: mean (upcast per-axis; sum stays float64 for大 N 求和精度)
    sum_xyz = cp.stack([
        cp_sum_labels(fg_xyz[:, a].astype(cp.float64), fg_labs, indices) for a in range(3)
    ], axis=1)
    mean_xyz = sum_xyz / count_safe[:, None]               # (K, 3) float64

    # Pass 2: centered second moment — 无消减
    cov = cp.zeros((n_lab, 3, 3), dtype=cp.float64)
    for a in range(3):
        xa = fg_xyz[:, a].astype(cp.float64) - mean_xyz[fg_labs - 1, a]  # (N,) f64
        for b in range(a, 3):
            if a == b:
                v = cp_sum_labels(xa * xa, fg_labs, indices) / count_safe
            else:
                xb = fg_xyz[:, b].astype(cp.float64) - mean_xyz[fg_labs - 1, b]
                v = cp_sum_labels(xa * xb, fg_labs, indices) / count_safe
            cov[:, a, b] = v
            if a != b:
                cov[:, b, a] = v

    evals, evecs = cp.linalg.eigh(cov)   # ascending
    return count, mean_xyz, evals, evecs


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Z-axis smoothing (GPU in/out)
# ─────────────────────────────────────────────────────────────────────────────


def smooth_z_axis_gpu(vol_gpu: cp.ndarray, sigma_z: float, sigma_xy: float = 0.0) -> cp.ndarray:
    """沿 Z 抹平 NN 逐层跳变；xy 方向可选轻量平滑以补内部洞。"""
    v = vol_gpu
    if sigma_z > 0:
        v = cp_gaussian1d(v, sigma=sigma_z, axis=0)
    if sigma_xy > 0:
        v = cp_gaussian1d(v, sigma=sigma_xy, axis=1)
        v = cp_gaussian1d(v, sigma=sigma_xy, axis=2)
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: K-mode direction detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_direction_modes(
    high_mask_gpu: cp.ndarray,
    mode2_ratio_thr: float,
    min_pts_per_cc: int,
    voxel_size: float = 1.0,
    debug_prefix: Optional[Path] = None,
) -> List[np.ndarray]:
    """
    返回 1 或 2 个单位向量（XYZ）。

    每个高阈值 CC 做 3×3 PCA 取主轴 dᵢ 与 lᵢ (~sqrt(λ_top·n))，以 wᵢ = lᵢ² 加权
    外积得 W = Σ wᵢ·dᵢdᵢᵀ。W 是"方向分布的协方差"：
      - 单峰：λ_top1 压倒性大 → 取 top-1 eigenvector
      - 双峰（如正交）：λ_top2/λ_top1 接近 1 → 取 top-1 + top-2

    mode2_ratio_thr ∈ (0,1)：越小越敏感（易触发双模式）。
    """
    labels_gpu = cskimage.measure.label(high_mask_gpu.astype(cp.uint8), connectivity=2)
    fg_xyz, fg_labs, n_lab = _label_sorted_coords_gpu(labels_gpu)
    if n_lab == 0 or len(fg_xyz) == 0:
        return []

    count, _, evals, evecs = _batched_label_pca(fg_xyz, fg_labs, n_lab)
    principal = evecs[:, :, 2]                                # (K, 3) 主轴
    lengths = cp.sqrt(cp.abs(evals[:, 2]) * count)            # (K,)

    keep = count >= float(min_pts_per_cc)
    if not bool(keep.any().get()):
        return []
    dirs = principal[keep]
    w = lengths[keep] ** 2                                    # (K',)

    W = (dirs[:, :, None] * dirs[:, None, :] * w[:, None, None]).sum(axis=0)
    W_evals, W_evecs = cp.linalg.eigh(W)                      # ascending
    W_evals_np = cp.asnumpy(W_evals)
    W_evecs_np = cp.asnumpy(W_evecs)

    lam1 = float(W_evals_np[2])
    lam2 = float(W_evals_np[1])
    ratio = lam2 / max(lam1, 1e-12)
    top1 = W_evecs_np[:, 2].astype(np.float32)
    top1 /= max(np.linalg.norm(top1), 1e-12)

    modes = [top1]
    if ratio >= mode2_ratio_thr:
        top2 = W_evecs_np[:, 1].astype(np.float32)
        top2 /= max(np.linalg.norm(top2), 1e-12)
        modes.append(top2)

    print(f"  K-mode: λ2/λ1 = {ratio:.3f} (thr={mode2_ratio_thr}) → K={len(modes)}")
    for i, m in enumerate(modes):
        print(f"    mode {i}: [{m[0]:+.3f}, {m[1]:+.3f}, {m[2]:+.3f}]")

    if debug_prefix is not None:
        for i, m in enumerate(modes):
            save_mrc(
                debug_prefix.parent / (debug_prefix.name + f"_2_dir_mode{i}.mrc"),
                _draw_direction_volume(high_mask_gpu.shape, m), voxel_size,
            )
    return modes


def _draw_direction_volume(
    shape_zyx: Tuple[int, ...],
    dominant_dir_xyz: np.ndarray,
    ball_radius: int = 8,
    line_length: int = 80,
) -> np.ndarray:
    """中心放小球 + 从中心沿 d 画一条线，供肉眼核对估计方向。"""
    vol = np.zeros(shape_zyx, dtype=np.float32)
    cz, cy, cx = shape_zyx[0] // 2, shape_zyx[1] // 2, shape_zyx[2] // 2
    r = int(ball_radius)
    zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    in_ball = (zz ** 2 + yy ** 2 + xx ** 2) <= r ** 2
    bz = np.clip(cz + zz[in_ball], 0, shape_zyx[0] - 1)
    by = np.clip(cy + yy[in_ball], 0, shape_zyx[1] - 1)
    bx = np.clip(cx + xx[in_ball], 0, shape_zyx[2] - 1)
    vol[bz, by, bx] = 2.0
    d = np.asarray(dominant_dir_xyz, np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    for t in np.linspace(0, float(line_length), line_length * 4 + 1):
        lx = int(round(cx + t * d[0]))
        ly = int(round(cy + t * d[1]))
        lz = int(round(cz + t * d[2]))
        if 0 <= lz < shape_zyx[0] and 0 <= ly < shape_zyx[1] and 0 <= lx < shape_zyx[2]:
            if vol[lz, ly, lx] < 1.5:
                vol[lz, ly, lx] = 1.0
    return vol


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Binary mask (GPU in/out)
# ─────────────────────────────────────────────────────────────────────────────


def extract_binary_mask_gpu(
    vol_gpu: cp.ndarray,
    threshold: float,
    opening_radius: int,
    voxel_size: float = 1.0,
    debug_prefix: Optional[Path] = None,
) -> Optional[cp.ndarray]:
    mask = vol_gpu >= threshold
    n_on = int(mask.sum().get())
    if n_on == 0:
        print("  mask empty after threshold")
        return None
    print(f"  {n_on} voxels ({100.*n_on/vol_gpu.size:.2f}%)")
    if opening_radius > 0:
        mask = cskimage.morphology.binary_opening(mask, cskimage.morphology.ball(opening_radius))
        if not bool(mask.any().get()):
            print("  mask empty after ball opening")
            return None
        if debug_prefix is not None:
            save_mrc(debug_prefix.parent / (debug_prefix.name + "_3_opened.mrc"), mask, voxel_size)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Multi-mode elongation filter (GPU segment-reduce)
# ─────────────────────────────────────────────────────────────────────────────


def filter_ccs_by_elongation_multi(
    mask_gpu: cp.ndarray,
    modes: List[np.ndarray],
    min_aspect_ratio: float,
    max_angle_deg: float,
    min_pts: int = 10,
    voxel_size: float = 1.0,
    debug_prefix: Optional[Path] = None,
) -> cp.ndarray:
    """
    保留"至少对齐任一 mode"的 CC：
      CC 主轴与某 mode |cos| ≥ cos(max_angle_deg)，且沿该 mode 投影的
      length/(2·perp_span) ≥ min_aspect_ratio。

    所有 per-CC 量（mean/cov/proj 极值/perp 极值）走 cupyx segment-reduce，
    没有 Python for-CC。
    """
    cos_thr = math.cos(math.radians(max_angle_deg))

    labels_gpu = cskimage.measure.label(mask_gpu.astype(cp.uint8), connectivity=2)
    fg_xyz, fg_labs, n_lab = _label_sorted_coords_gpu(labels_gpu)
    if n_lab == 0 or len(fg_xyz) == 0:
        print("  no CC")
        return mask_gpu

    count, mean_xyz, _, evecs = _batched_label_pca(fg_xyz, fg_labs, n_lab)
    cc_main = evecs[:, :, 2]                                 # (K, 3) 主轴 float64
    indices = cp.arange(1, n_lab + 1, dtype=cp.int32)

    # centered 用 float32 存储（坐标差值精度够）；|centered|² 外提一次，后面多模式复用
    mean_f32 = mean_xyz.astype(cp.float32)
    centered = fg_xyz - mean_f32[fg_labs - 1]                # (N, 3) f32
    centered_sq_sum = (centered * centered).sum(axis=1)      # (N,)   f32

    pass_any = cp.zeros(n_lab, dtype=bool)
    size_ok = count >= int(min_pts)
    for md in modes:
        d = cp.asarray(md, dtype=cp.float32)
        proj = centered @ d                                  # (N,)   f32
        proj_max = cp_max_labels(proj, fg_labs, indices)
        proj_min = cp_min_labels(proj, fg_labs, indices)
        length = proj_max - proj_min
        # |r_perp|² = |centered|² − proj²  (d 单位向量)。省掉 (N,3) r_perp 分配
        r_perp_norm = cp.sqrt(cp.maximum(centered_sq_sum - proj * proj, 0.0))
        perp = cp_max_labels(r_perp_norm, fg_labs, indices)
        aspect = length / (2.0 * perp + 1e-3)
        align = cp.abs(cc_main @ d.astype(cp.float64))       # (K,)
        pass_any |= (align >= cos_thr) & (aspect >= float(min_aspect_ratio)) & size_ok
    del centered, centered_sq_sum

    keep = cp.concatenate([cp.zeros(1, dtype=bool), pass_any])   # 1-based
    result = keep[labels_gpu]

    n_kept = int(pass_any.sum().get())
    print(f"  {n_lab} CCs checked → kept {n_kept}")

    if debug_prefix is not None:
        save_mrc(debug_prefix.parent / (debug_prefix.name + "_4_elongated.mrc"), result, voxel_size)
        rej = cp.where(keep[labels_gpu], cp.uint16(0), labels_gpu.astype(cp.uint16))
        save_mrc(debug_prefix.parent / (debug_prefix.name + "_4_rejected.mrc"), rej, voxel_size)
        del rej

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Skeletonize → per-mode segments
# ─────────────────────────────────────────────────────────────────────────────


def skeleton_to_segments_per_mode(
    mask_gpu: cp.ndarray,
    modes: List[np.ndarray],
    max_angle_deg: float = 30.0,
    bridge_radius: int = 3,
    junction_clearance: int = 2,
    min_branch_pts: int = 5,
    voxel_size: float = 1.0,
    debug_prefix: Optional[Path] = None,
) -> List[List[np.ndarray]]:
    """
    骨架 + bridge + junction 拆支 → branch PCA → 按最近模式分配。融合 CC 里混进的
    横向 cross-link 会被 junction split 打断，且方向不对齐任一模式的 branch 直接丢弃。
    """
    # ---- skeletonize (唯一 D2H/H2D) ----
    mask_cpu = cp.asnumpy(mask_gpu)
    skel_cpu = skeletonize(mask_cpu.astype(bool))
    if not skel_cpu.any():
        print("  skeleton empty")
        return [[] for _ in modes]
    if debug_prefix is not None:
        save_mrc(debug_prefix.parent / (debug_prefix.name + "_5_skel_raw.mrc"), skel_cpu.astype(np.float32), voxel_size)

    # ---- bridge 微缺口 → 再骨架化 (bridge 本身 GPU，skel 必 CPU) ----
    skel_gpu = cp.asarray(skel_cpu)
    if bridge_radius > 0:
        dilated = cskimage.morphology.binary_dilation(skel_gpu, cskimage.morphology.ball(bridge_radius))
        del skel_gpu
        skel_cpu = skeletonize(cp.asnumpy(dilated).astype(bool))
        del dilated
        if not skel_cpu.any():
            return [[] for _ in modes]
        skel_gpu = cp.asarray(skel_cpu)
        if debug_prefix is not None:
            save_mrc(debug_prefix.parent / (debug_prefix.name + "_6_skel_bridged.mrc"), skel_cpu, voxel_size)

    # ---- junction split ----
    # n_nbr 上限 = 27 邻居，int8 足够；避免 int32 (1.26 GB on 300×1024² vol)
    kernel = cp.ones((3, 3, 3), dtype=cp.int8)
    kernel[1, 1, 1] = 0
    skel_i8 = skel_gpu.astype(cp.int8)
    n_nbr = cp_convolve(skel_i8, kernel, mode="constant", cval=0)
    del skel_i8
    junc = skel_gpu & (n_nbr >= 3)
    del n_nbr
    if int(junc.sum().get()) > 0 and junction_clearance > 0:
        junc = cskimage.morphology.binary_dilation(junc, cskimage.morphology.ball(junction_clearance))
    branches = skel_gpu & (~junc)
    del junc, skel_gpu

    # ---- branch batched PCA → 最近模式指派 ----
    # label 上界 ≈ 数千，uint16 足够（0..65535）
    lab_gpu = cskimage.measure.label(branches.astype(cp.uint8), connectivity=2).astype(cp.uint16)
    del branches
    fg_xyz, fg_labs, n_lab = _label_sorted_coords_gpu(lab_gpu)
    if n_lab == 0 or len(fg_xyz) == 0:
        return [[] for _ in modes]

    count, _, _, evecs = _batched_label_pca(fg_xyz, fg_labs, n_lab)
    b_dir = evecs[:, :, 2]                                    # (K, 3)

    cos_thr = math.cos(math.radians(max_angle_deg))
    modes_gpu = cp.asarray(np.stack(modes, axis=0), dtype=cp.float64)   # (M, 3)
    align = cp.abs(b_dir @ modes_gpu.T)                       # (K, M)
    best_mode = cp.argmax(align, axis=1)                      # (K,)
    best_align = cp.take_along_axis(align, best_mode[:, None], axis=1).ravel()
    valid = (best_align >= cos_thr) & (count >= int(min_branch_pts))

    branch_mode = cp.where(valid, best_mode + 1, 0).astype(cp.uint8)    # (K,) 1-based, 模式数 ≤ 2
    lut = cp.concatenate([cp.zeros(1, cp.uint8), branch_mode])
    mode_vol_branch = lut[lab_gpu]                            # (Z,Y,X) uint8 0=bg, 1..M
    del lab_gpu

    n_kept = int(valid.sum().get())
    print(f"  branches: {n_lab} → kept {n_kept}")
    if debug_prefix is not None:
        save_mrc(debug_prefix.parent / (debug_prefix.name + "_7_branches_mode.mrc"), mode_vol_branch, voxel_size)

    # ---- per-mode regroup → 有序骨架点集 ----
    segments_per_mode: List[List[np.ndarray]] = []
    vol_shape = mode_vol_branch.shape
    for mi in range(len(modes)):
        thin_gpu = (mode_vol_branch == (mi + 1))
        if not bool(thin_gpu.any().get()):
            segments_per_mode.append([])
            del thin_gpu
            continue
        if bridge_radius > 0:
            grouped = cskimage.morphology.binary_dilation(thin_gpu, cskimage.morphology.ball(bridge_radius))
        else:
            grouped = thin_gpu
        final_lab = cskimage.measure.label(grouped.astype(cp.uint8), connectivity=2).astype(cp.uint16)
        del grouped
        n_final = int(final_lab.max().get())
        if n_final == 0:
            segments_per_mode.append([])
            del thin_gpu, final_lab
            continue
        thin_lab = cp.where(thin_gpu, final_lab, 0)
        del thin_gpu, final_lab
        segs = _extract_ordered_segments(thin_lab, n_final, modes[mi])
        del thin_lab
        segments_per_mode.append(segs)
        if debug_prefix is not None and segs:
            save_mrc(
                debug_prefix.parent / (debug_prefix.name + f"_8_segments_mode{mi}.mrc"),
                _draw_segments_to_volume(segs, vol_shape, radius=2), voxel_size,
            )
    del mode_vol_branch

    print(f"  segments per mode: {[len(ss) for ss in segments_per_mode]}")
    return segments_per_mode


def _extract_ordered_segments(
    thin_lab_gpu: cp.ndarray,
    n_lab: int,
    mode_dir: np.ndarray,
) -> List[np.ndarray]:
    """
    每个 label 的骨架体素：沿 mode 投影排序，稀疏去重 (间隔 ≥ 1 像素)。
    label 数和每个 label 点数都不大 → 一次 D2H 后 CPU 循环即可。
    """
    fg_xyz, fg_labs, _ = _label_sorted_coords_gpu(thin_lab_gpu)
    if len(fg_xyz) == 0:
        return []
    xyz_np = cp.asnumpy(fg_xyz).astype(np.float32)
    labs_np = cp.asnumpy(fg_labs).astype(np.int64)
    bounds = np.searchsorted(labs_np, np.arange(1, n_lab + 2))
    d = np.asarray(mode_dir, dtype=np.float64)

    segs: List[np.ndarray] = []
    for lab in range(1, n_lab + 1):
        s, e = bounds[lab - 1], bounds[lab]
        if e - s < 2:
            continue
        pts = xyz_np[s:e]
        pts = pts[np.argsort(pts @ d)]
        pts = _dedup_sequential(pts, 1.0)
        if len(pts) >= 2:
            segs.append(pts.astype(np.float32))
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Per-mode greedy B-spline + erase (sliding-window candidate search)
# ─────────────────────────────────────────────────────────────────────────────


def _safe_unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    nv = float(np.linalg.norm(v))
    if not np.isfinite(nv) or nv < 1e-8:
        return fallback.copy()
    return (v / nv).astype(np.float64)


def _polyline_arc_length(pts_xyz: np.ndarray) -> float:
    if len(pts_xyz) < 2:
        return 0.0
    d = np.diff(pts_xyz.astype(np.float64), axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))


def _segment_dir(seg_xyz: np.ndarray, dominant_dir: np.ndarray) -> np.ndarray:
    seg = seg_xyz.astype(np.float64)
    if len(seg) >= 3:
        try:
            _, _, vh = np.linalg.svd(seg - seg.mean(0), full_matrices=False)
            d = _safe_unit(vh[0], dominant_dir)
        except Exception:
            d = _safe_unit(seg[-1] - seg[0], dominant_dir)
    else:
        d = _safe_unit(seg[-1] - seg[0], dominant_dir)
    if float(np.dot(d, dominant_dir)) < 0.0:
        d = -d
    return d


def _orient_segment_along(seg_xyz: np.ndarray, dominant_dir: np.ndarray) -> np.ndarray:
    seg = seg_xyz.astype(np.float64)
    if len(seg) < 2:
        return seg
    p = seg @ dominant_dir
    if p[-1] < p[0]:
        seg = seg[::-1]
    return seg


def _dedup_sequential(pts: np.ndarray, thr: float) -> np.ndarray:
    """
    保留：每个点距"上一个保留点"的 3D 距离 ≥ thr。由于依赖 last-kept 参考点，
    无法 numpy 向量化；用标量三分量算术避免 np.linalg.norm 的 slice 分配开销。
    末点强制保留。
    """
    n = len(pts)
    if n <= 1:
        return pts
    thr2 = float(thr) ** 2
    keep_idx = [0]
    lx, ly, lz = float(pts[0, 0]), float(pts[0, 1]), float(pts[0, 2])
    for i in range(1, n):
        dx = float(pts[i, 0]) - lx
        dy = float(pts[i, 1]) - ly
        dz = float(pts[i, 2]) - lz
        if dx * dx + dy * dy + dz * dz >= thr2:
            keep_idx.append(i)
            lx, ly, lz = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
    if keep_idx[-1] != n - 1:
        keep_idx.append(n - 1)
    return pts[keep_idx]


def _stitch_chain_points(chain: List[int], segments: List[np.ndarray], dominant_dir: np.ndarray) -> np.ndarray:
    if not chain:
        return np.empty((0, 3), dtype=np.float64)
    pts_list = [_orient_segment_along(segments[i], dominant_dir) for i in chain]
    pts = np.vstack(pts_list)
    pts = pts[np.argsort(pts @ dominant_dir)]
    return _dedup_sequential(pts, 0.7).astype(np.float64)


def _sample_volume_nn(vol_zyx: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    if len(pts_xyz) == 0:
        return np.empty((0,), dtype=np.float64)
    z = np.clip(np.round(pts_xyz[:, 2]).astype(np.int64), 0, vol_zyx.shape[0] - 1)
    y = np.clip(np.round(pts_xyz[:, 1]).astype(np.int64), 0, vol_zyx.shape[1] - 1)
    x = np.clip(np.round(pts_xyz[:, 0]).astype(np.int64), 0, vol_zyx.shape[2] - 1)
    return vol_zyx[z, y, x].astype(np.float64)


def _fit_bspline_curve(
    pts_xyz: np.ndarray,
    smoothness: float,
    min_eval: int = 40,
) -> Tuple[np.ndarray, float, float]:
    """拟合 3D B-spline；返回 (曲线点, 弯曲能量, 曲率变异系数)。"""
    pts = np.asarray(pts_xyz, dtype=np.float64)
    if len(pts) < 4:
        return pts.astype(np.float32), 0.0, 0.0
    try:
        k = min(3, len(pts) - 1)
        s = max(float(smoothness), 0.0) * float(len(pts))
        tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=s, k=k)
        n_eval = max(min_eval, len(pts) * 3)
        u = np.linspace(0.0, 1.0, n_eval)
        x, y, z = splev(u, tck)
        curve = np.stack([x, y, z], axis=1).astype(np.float64)
        d1 = np.stack(splev(u, tck, der=1), axis=1)
        d2 = np.stack(splev(u, tck, der=2), axis=1)
        sp = np.linalg.norm(d1, axis=1)
        cross = np.cross(d1, d2)
        curv = np.linalg.norm(cross, axis=1) / np.maximum(sp ** 3, 1e-8)
        du = 1.0 / max(n_eval - 1, 1)
        ds = sp * du
        total_len = max(float(np.sum(ds)), 1e-8)
        bending = float(np.sum((curv ** 2) * ds) / total_len)
        curv_mean = float(np.mean(curv))
        curv_cv = float(np.std(curv) / max(curv_mean, 1e-6))
        return curve.astype(np.float32), bending, curv_cv
    except Exception:
        return pts.astype(np.float32), 0.0, 0.0


def _grow_chain_sliding(
    seed_idx: int,
    active_mask: np.ndarray,
    t_sorted_idxs: np.ndarray,
    t_sorted_vals: np.ndarray,
    centroids: np.ndarray,
    seg_dirs: np.ndarray,
    seg_signals: np.ndarray,
    dominant_dir: np.ndarray,
    fiber_radius_px: float,
    max_gap_along: float,
    cos_thr: float,
) -> List[int]:
    """
    沿 d 前后贪心扩展。候选限制在 |Δt| ≤ max_gap_along 的滑窗内（np.searchsorted）；
    每步对窗内所有候选做 numpy 批量过滤 + 评分，避免 Python 标量循环。
    """
    n = len(active_mask)
    used_mask = np.zeros(n, dtype=bool)
    used_mask[seed_idx] = True
    chain = [seed_idx]

    def _best_next(cur_idx: int, forward: bool) -> Optional[int]:
        c0 = centroids[cur_idx]
        d0 = seg_dirs[cur_idx]
        t0 = float(c0 @ dominant_dir)
        lo = int(np.searchsorted(t_sorted_vals, t0 - max_gap_along, side="left"))
        hi = int(np.searchsorted(t_sorted_vals, t0 + max_gap_along, side="right"))
        if hi <= lo:
            return None
        cand = t_sorted_idxs[lo:hi]
        valid = active_mask[cand] & (~used_mask[cand])
        if not valid.any():
            return None
        cand = cand[valid]
        dv = centroids[cand] - c0                                        # (C, 3)
        gap_along = dv @ dominant_dir                                    # (C,)
        if forward:
            m = (gap_along > 1e-6) & (gap_along <= max_gap_along)
        else:
            m = (gap_along < -1e-6) & (-gap_along <= max_gap_along)
        if not m.any():
            return None
        cand, dv, gap_along = cand[m], dv[m], gap_along[m]

        lateral = np.linalg.norm(dv - gap_along[:, None] * dominant_dir[None, :], axis=1)
        m = lateral <= fiber_radius_px
        if not m.any():
            return None
        cand, dv, gap_along, lateral = cand[m], dv[m], gap_along[m], lateral[m]

        gap_norm = np.linalg.norm(dv, axis=1)
        m = gap_norm >= 1e-6
        if not m.any():
            return None
        cand = cand[m]; gap_along = gap_along[m]; lateral = lateral[m]; gap_norm = gap_norm[m]

        cos_gap = np.abs(gap_along) / gap_norm
        m = cos_gap >= cos_thr
        if not m.any():
            return None
        cand = cand[m]; gap_along = gap_along[m]; lateral = lateral[m]; cos_gap = cos_gap[m]

        sd = seg_dirs[cand]                                              # (C, 3)
        m = np.abs(sd @ dominant_dir) >= cos_thr
        if not m.any():
            return None
        cand = cand[m]; gap_along = gap_along[m]; lateral = lateral[m]; cos_gap = cos_gap[m]; sd = sd[m]

        turn_align = np.abs(sd @ d0)
        score = (
            2.0 * seg_signals[cand]
            + 50.0 * turn_align
            + 30.0 * cos_gap
            - 1.5 * lateral
            - 0.4 * np.abs(gap_along)
        )
        return int(cand[int(np.argmax(score))])

    cur = seed_idx
    while True:
        nxt = _best_next(cur, forward=True)
        if nxt is None:
            break
        chain.append(nxt)
        used_mask[nxt] = True
        cur = nxt

    cur = seed_idx
    head: List[int] = []
    while True:
        prv = _best_next(cur, forward=False)
        if prv is None:
            break
        head.append(prv)
        used_mask[prv] = True
        cur = prv
    head.reverse()
    return head + chain


def extract_fibers_by_greedy_bspline(
    segments: List[np.ndarray],
    vol_zyx: np.ndarray,
    dominant_dir: np.ndarray,
    fiber_radius_px: float,
    erase_radius_px: float,
    min_len_px: float,
    max_angle_deg: float,
    smoothness: float,
    signal_weight: float,
    bending_weight: float,
    curvature_cv_weight: float,
    min_candidate_score: float,
) -> List[np.ndarray]:
    """每次选全局最优候选链 → 拟样条 → 抹除半径内所有段 → 迭代至耗尽。"""
    if not segments:
        return []

    d = np.asarray(dominant_dir, dtype=np.float64)
    d /= max(float(np.linalg.norm(d)), 1e-8)
    cos_thr = math.cos(math.radians(max_angle_deg))
    max_gap_along = max(8.0 * float(fiber_radius_px), 12.0)

    n = len(segments)
    centroids = np.array([s.mean(0) for s in segments], dtype=np.float64)
    seg_dirs = np.array([_segment_dir(s, d) for s in segments], dtype=np.float64)
    seg_lens = np.array([_polyline_arc_length(s) for s in segments], dtype=np.float64)
    seg_signals = np.array([float(_sample_volume_nn(vol_zyx, s).mean()) for s in segments], dtype=np.float64)

    t_vals = centroids @ d
    t_sorted_idxs = np.argsort(t_vals)
    t_sorted_vals = t_vals[t_sorted_idxs]

    # erase 端：所有 segment 点平铺到 GPU，每 iter 一次 KDTree 批量最近距离 + per-seg min-reduce
    seg_lens_each = np.array([len(s) for s in segments], dtype=np.int32)
    all_pts_gpu = cp.asarray(np.concatenate(segments, axis=0), dtype=cp.float32)    # (Σlen, 3)
    pt_to_seg_gpu = cp.asarray(np.repeat(np.arange(n, dtype=np.int32), seg_lens_each)) + 1
    seg_indices_gpu = cp.arange(1, n + 1, dtype=cp.int32)

    active = np.ones(n, dtype=bool)
    fibers: List[np.ndarray] = []
    iter_idx = 0

    while active.any():
        iter_idx += 1
        best_score = -1e18
        best_curve = None
        best_chain: List[int] = []
        best_arc = 0.0

        for seed in np.where(active)[0]:
            chain = _grow_chain_sliding(
                int(seed), active, t_sorted_idxs, t_sorted_vals,
                centroids, seg_dirs, seg_signals, d,
                fiber_radius_px, max_gap_along, cos_thr,
            )
            if not chain:
                continue
            chain_pts = _stitch_chain_points(chain, segments, d)
            if len(chain_pts) < 4:
                continue
            arc = _polyline_arc_length(chain_pts)
            if arc < float(min_len_px):
                continue
            curve, bending, curv_cv = _fit_bspline_curve(chain_pts, smoothness=smoothness)
            if len(curve) < 2:
                continue
            curve_sig = float(_sample_volume_nn(vol_zyx, curve).mean())
            chain_arr = np.asarray(chain, dtype=np.int64)
            chain_strength = float(seg_signals[chain_arr].mean())
            chain_len = float(seg_lens[chain_arr].sum())
            score = (
                signal_weight * (0.55 * curve_sig + 0.45 * chain_strength)
                + 0.05 * chain_len
                + 0.25 * len(chain)
                - bending_weight * bending
                - curvature_cv_weight * curv_cv
            )
            if score > best_score:
                best_score = score
                best_curve = curve.astype(np.float32)
                best_chain = chain
                best_arc = arc

        if best_curve is None or best_score < float(min_candidate_score):
            break

        fibers.append(best_curve)
        # KDTree 批量最近距离 → per-seg min → 半径内整段抹除
        curve_gpu = cp.asarray(best_curve, dtype=cp.float32)
        tree = cp_KDTree(curve_gpu)
        dists, _ = tree.query(all_pts_gpu)                               # (Σlen,)
        seg_min = cp_min_labels(dists, pt_to_seg_gpu, seg_indices_gpu)   # (n,)
        seg_min_np = cp.asnumpy(seg_min)
        chain_set = set(best_chain)
        erased = 0
        for i in np.where(active)[0]:
            ii = int(i)
            if ii in chain_set or seg_min_np[ii] < erase_radius_px:
                active[ii] = False
                erased += 1
        print(
            f"  greedy iter {iter_idx:02d}: chain={len(best_chain)} seg, "
            f"arc={best_arc:.1f}px, score={best_score:.3f}, "
            f"erased={erased}, remain={int(active.sum())}"
        )
    del all_pts_gpu, pt_to_seg_gpu, seg_indices_gpu
    return fibers


# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Sampling + Euler
# ─────────────────────────────────────────────────────────────────────────────


def fit_line_and_sample(
    segment_xyz: np.ndarray,
    dominant_dir: np.ndarray,
    spacing_px: float,
    pre_smoothed: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在局部 (d, u, v) 坐标系下以 t² 弱项拟合轻弯曲中心线，再按弧长重采样。
    直接多段折线插值容易被骨架抖动带偏；二次项加正则是故意的欠拟合。
    """
    def _unit(v: np.ndarray) -> np.ndarray:
        nv = float(np.linalg.norm(v))
        if nv < 1e-8 or not np.isfinite(nv):
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return v / nv

    def _resample_polyline(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(poly) == 0:
            d0 = _unit(np.asarray(dominant_dir, dtype=np.float64))
            return np.empty((0, 3), dtype=np.float32), d0.astype(np.float32)
        if len(poly) == 1:
            d0 = _unit(np.asarray(dominant_dir, dtype=np.float64))
            return poly.astype(np.float32), d0.astype(np.float32)
        seg_vec = np.diff(poly, axis=0)
        seg_len = np.linalg.norm(seg_vec, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = float(cum[-1])
        if total_len < max(spacing_px * 0.5, 1e-6):
            mid = poly.mean(0, keepdims=True)
            dd = _unit(poly[-1] - poly[0])
            if float(np.dot(dd, dominant_dir)) < 0:
                dd = -dd
            return mid.astype(np.float32), dd.astype(np.float32)
        n_pts = max(2, int(round(total_len / spacing_px)) + 1)
        ts = np.linspace(0.0, total_len, n_pts)
        out = np.empty((n_pts, 3), dtype=np.float64)
        j = 0
        for k, t in enumerate(ts):
            while j + 1 < len(cum) and cum[j + 1] < t:
                j += 1
            if j + 1 >= len(cum):
                out[k] = poly[-1]
                continue
            den = max(cum[j + 1] - cum[j], 1e-8)
            a = (t - cum[j]) / den
            out[k] = (1.0 - a) * poly[j] + a * poly[j + 1]
        gd = _unit(out[-1] - out[0])
        if float(np.dot(gd, dominant_dir)) < 0:
            gd = -gd
        return out.astype(np.float32), gd.astype(np.float32)

    pts = np.asarray(segment_xyz, dtype=np.float64)
    if len(pts) == 0:
        d0 = _unit(np.asarray(dominant_dir, dtype=np.float64))
        return np.empty((0, 3), dtype=np.float32), d0.astype(np.float32)

    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= 1e-6:
            keep.append(i)
    pts = pts[keep]
    if len(pts) < 4:
        return _resample_polyline(pts)

    if pre_smoothed:
        pts_smooth = pts
    else:
        win = min(7, len(pts) if len(pts) % 2 == 1 else len(pts) - 1)
        if win >= 3:
            r = win // 2
            sm = np.zeros_like(pts)
            for i in range(len(pts)):
                s = max(0, i - r)
                e = min(len(pts), i + r + 1)
                sm[i] = pts[s:e].mean(0)
            pts_smooth = sm
        else:
            pts_smooth = pts

    d = _unit(np.asarray(dominant_dir, dtype=np.float64))
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(ref, d))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = _unit(ref - float(np.dot(ref, d)) * d)
    v = _unit(np.cross(d, u))

    origin = pts_smooth.mean(axis=0)
    rel = pts_smooth - origin[None, :]
    t = rel @ d
    tu = rel @ u
    tv = rel @ v
    span = float(np.max(t) - np.min(t))
    if span < 2.0:
        return _resample_polyline(pts_smooth)

    n_bins = min(24, max(6, len(t) // 8))
    edges = np.linspace(float(np.min(t)), float(np.max(t)), n_bins + 1)
    tb, ub, vb = [], [], []
    for bi in range(n_bins):
        if bi < n_bins - 1:
            m = (t >= edges[bi]) & (t < edges[bi + 1])
        else:
            m = (t >= edges[bi]) & (t <= edges[bi + 1])
        if not np.any(m):
            continue
        tb.append(float(np.median(t[m])))
        ub.append(float(np.median(tu[m])))
        vb.append(float(np.median(tv[m])))

    if len(tb) < 4:
        return _resample_polyline(pts_smooth)

    tb = np.asarray(tb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    vb = np.asarray(vb, dtype=np.float64)

    t0 = float(np.mean(tb))
    t_scale = max(float(np.std(tb)), 1.0)
    ts = (tb - t0) / t_scale
    A = np.stack([np.ones_like(ts), ts, ts * ts], axis=1)

    lam2 = 1.5 if pre_smoothed else 2.0
    reg = np.diag([1e-6, 1e-4, lam2])

    try:
        w = np.ones(len(tb), dtype=np.float64)
        coef_u = np.zeros(3, dtype=np.float64)
        coef_v = np.zeros(3, dtype=np.float64)
        for _ in range(2):
            W = np.sqrt(w)[:, None]
            Aw = A * W
            uw = ub * np.sqrt(w)
            vw = vb * np.sqrt(w)
            coef_u = np.linalg.solve(Aw.T @ Aw + reg, Aw.T @ uw)
            coef_v = np.linalg.solve(Aw.T @ Aw + reg, Aw.T @ vw)
            ru = ub - A @ coef_u
            rv = vb - A @ coef_v
            rr = np.sqrt(ru * ru + rv * rv)
            s = max(float(np.median(rr) * 1.4826), 1e-3)
            w = 1.0 / (1.0 + (rr / (2.5 * s)) ** 2)
    except Exception:
        return _resample_polyline(pts_smooth)

    t_min = float(np.percentile(t, 1))
    t_max = float(np.percentile(t, 99))
    n_dense = max(60, int(round((t_max - t_min) * 2.0)))
    t_grid = np.linspace(t_min, t_max, n_dense)
    ts_grid = (t_grid - t0) / t_scale
    G = np.stack([np.ones_like(ts_grid), ts_grid, ts_grid * ts_grid], axis=1)
    gu = G @ coef_u
    gv = G @ coef_v
    curve = origin[None, :] + t_grid[:, None] * d[None, :] + gu[:, None] * u[None, :] + gv[:, None] * v[None, :]
    return _resample_polyline(curve.astype(np.float64))


def vector_to_euler_zyz(vec_xyz: np.ndarray) -> Tuple[float, float, float]:
    """XYZ 向量 → RELION ZYZ Euler (deg)。"""
    v = np.asarray(vec_xyz, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-6:
        return 0.0, 0.0, 0.0
    v /= n
    z_ref = np.array([0., 0., 1.], dtype=np.float64)
    try:
        rot_obj, _ = R.align_vectors(v[None, :], z_ref[None, :])
    except Exception:
        axis = np.cross(z_ref, v)
        s = float(np.linalg.norm(axis))
        c = float(np.clip(np.dot(z_ref, v), -1., 1.))
        if s < 1e-8:
            rot_obj = R.identity() if c > 0 else R.from_rotvec(np.pi * np.array([1., 0., 0.]))
        else:
            axis /= s
            kx, ky, kz = axis
            K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
            rot_obj = R.from_matrix(np.eye(3) + K * s + (K @ K) * (1 - c))
    ang = rot_obj.as_euler("ZYZ", degrees=True)
    return float(ang[0]), float(ang[1]), float(ang[2])


def _draw_segments_to_volume(
    segments: List[np.ndarray],
    shape_zyx: Tuple[int, ...],
    radius: int = 0,
    bridge_gaps: bool = False,
) -> np.ndarray:
    """按 segment 索引染色写入 float32 体积，debug 可视化用。"""
    vol = np.zeros(shape_zyx, dtype=np.float32)

    def _draw_point(cx: int, cy: int, cz: int, val: float) -> None:
        if radius <= 0:
            if 0 <= cz < shape_zyx[0] and 0 <= cy < shape_zyx[1] and 0 <= cx < shape_zyx[2]:
                vol[cz, cy, cx] = val
        else:
            offsets = _draw_point._offsets  # type: ignore[attr-defined]
            zs = np.clip(cz + offsets[:, 0], 0, shape_zyx[0] - 1)
            ys = np.clip(cy + offsets[:, 1], 0, shape_zyx[1] - 1)
            xs = np.clip(cx + offsets[:, 2], 0, shape_zyx[2] - 1)
            vol[zs, ys, xs] = val

    if radius > 0:
        r = int(radius)
        _draw_point._offsets = (  # type: ignore[attr-defined]
            np.argwhere(sum((np.mgrid[-r:r+1, -r:r+1, -r:r+1][i] ** 2 for i in range(3))) <= r * r) - r)

    for idx, seg in enumerate(segments, 1):
        val = float(idx)
        for pt in seg:
            _draw_point(int(round(float(pt[0]))), int(round(float(pt[1]))), int(round(float(pt[2]))), val)
        if bridge_gaps and len(seg) >= 2:
            for k in range(len(seg) - 1):
                p1, p2 = seg[k].astype(np.float64), seg[k + 1].astype(np.float64)
                gap = float(np.linalg.norm(p2 - p1))
                if gap > 1.5:
                    n_steps = max(int(gap), 2)
                    for t in np.linspace(0.0, 1.0, n_steps + 1)[1:-1]:
                        pt_b = p1 + t * (p2 - p1)
                        _draw_point(int(round(pt_b[0])), int(round(pt_b[1])), int(round(pt_b[2])), val)
    return vol


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline per tomogram
# ─────────────────────────────────────────────────────────────────────────────


def _process_single_tomogram(args: argparse.Namespace, tomo_path: Path, out_prefix: Path) -> None:
    t0 = time.time()
    max_step = int(getattr(args, "max_step", 99))

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f"loading: {tomo_path}")
    vol_np, mrc_voxel = load_mrc(tomo_path)
    voxel = float(args.voxel_size) if float(args.voxel_size) > 0 else mrc_voxel
    print(f"  shape (z,y,x): {vol_np.shape}, voxel: {voxel:.3f} Å, value range: [{vol_np.min():.0f}, {vol_np.max():.0f}]")

    vol_gpu = cp.asarray(vol_np, dtype=cp.float32)
    del vol_np

    bin_factor = max(1, int(args.bin))
    if bin_factor > 1:
        print(f"binning by factor {bin_factor}")
        vol_gpu = cskimage.transform.downscale_local_mean(vol_gpu, (bin_factor, bin_factor, bin_factor))
        voxel *= bin_factor
        print(f"  binned shape: {tuple(vol_gpu.shape)}, effective voxel: {voxel:.3f} Å")

    # ── 2. Smooth ───────────────────────────────────────────────────────────
    if max_step < 2:
        return
    if args.sigma_z > 0 or args.sigma_xy > 0:
        print(f"smoothing: sigma_z={args.sigma_z}, sigma_xy={args.sigma_xy}")
        vol_gpu = smooth_z_axis_gpu(vol_gpu, sigma_z=args.sigma_z, sigma_xy=args.sigma_xy)
        if args.debug:
            save_mrc(out_prefix.parent / (out_prefix.name + "_1_smoothed.mrc"), vol_gpu, voxel)

    # ── 3. K-mode direction detection ───────────────────────────────────────
    if max_step < 3:
        return
    high_th = float(args.high_threshold)
    print(f"K-mode direction: high_threshold={high_th}, mode2_ratio_thr={args.mode2_ratio_thr}")
    high_mask_gpu = vol_gpu >= high_th
    n_high = int(high_mask_gpu.sum().get())
    if n_high < 10:
        print("not enough high-conf voxels; exiting")
        return
    if args.debug:
        save_mrc(out_prefix.parent / (out_prefix.name + "_2_high_mask.mrc"), high_mask_gpu, voxel)

    modes = detect_direction_modes(
        high_mask_gpu,
        mode2_ratio_thr=args.mode2_ratio_thr,
        min_pts_per_cc=20,
        voxel_size=voxel,
        debug_prefix=out_prefix if args.debug else None,
    )
    if not modes:
        print("no direction mode found; exiting")
        return
    del high_mask_gpu

    # ── 4. Binary mask (shared) ─────────────────────────────────────────────
    if max_step < 4:
        return
    main_th = float(args.threshold)
    print(f"binary mask: threshold={main_th}, opening_radius={args.opening_radius}")
    mask_gpu = extract_binary_mask_gpu(
        vol_gpu, main_th,
        opening_radius=args.opening_radius,
        voxel_size=voxel,
        debug_prefix=out_prefix if args.debug else None,
    )
    if mask_gpu is None:
        print("no mask extracted; exiting")
        return
    vol_shape = mask_gpu.shape

    # ── 5. Multi-mode elongation filter ─────────────────────────────────────
    if max_step < 5:
        return
    print(f"elongation filter: min_aspect={args.min_aspect_ratio}, max_angle={args.direction_filter_angle}°")
    mask_gpu = filter_ccs_by_elongation_multi(
        mask_gpu, modes,
        min_aspect_ratio=args.min_aspect_ratio,
        max_angle_deg=args.direction_filter_angle,
        voxel_size=voxel,
        debug_prefix=out_prefix if args.debug else None,
    )
    if not bool(mask_gpu.any().get()):
        print("  all CCs removed; exiting")
        return

    # vol 后续只给 greedy 采样用，提前 D2H 让 skel 阶段有足够显存放 label 体积
    vol_np_smooth = cp.asnumpy(vol_gpu)
    del vol_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # ── 6. Skeletonize → per-mode segments ──────────────────────────────────
    if max_step < 6:
        return
    print(f"skeletonizing: bridge_radius={args.bridge_radius}")
    segments_per_mode = skeleton_to_segments_per_mode(
        mask_gpu, modes,
        max_angle_deg=args.direction_filter_angle,
        bridge_radius=args.bridge_radius,
        junction_clearance=2,
        min_branch_pts=5,
        voxel_size=voxel,
        debug_prefix=out_prefix if args.debug else None,
    )
    del mask_gpu
    cp.get_default_memory_pool().free_all_blocks()
    if not any(segments_per_mode):
        print("no segments; exiting")
        return

    # ── 7. Per-mode greedy spline extraction ────────────────────────────────
    if max_step < 7:
        return
    fiber_radius_px = float(args.fiber_diameter) / 2.0 / voxel
    erase_radius_px = (
        float(args.erase_radius) / voxel
        if float(args.erase_radius) > 0
        else max(fiber_radius_px * float(args.erase_radius_scale), fiber_radius_px)
    )
    min_len_px = max(0.0, float(args.min_fiber_length) / voxel)

    all_fibers: List[Tuple[np.ndarray, np.ndarray]] = []   # (fiber, mode_dir) pairs
    for mi, (segs, md) in enumerate(zip(segments_per_mode, modes)):
        if not segs:
            continue
        print(
            f"[mode {mi}] greedy spline extraction: "
            f"fiber_radius={fiber_radius_px:.1f}px, erase_radius={erase_radius_px:.1f}px, "
            f"angle<={args.direction_filter_angle}°, min_len={min_len_px:.1f}px"
        )
        fibers = extract_fibers_by_greedy_bspline(
            segs,
            vol_zyx=vol_np_smooth,
            dominant_dir=md,
            fiber_radius_px=fiber_radius_px,
            erase_radius_px=erase_radius_px,
            min_len_px=min_len_px,
            max_angle_deg=args.direction_filter_angle,
            smoothness=args.spline_smoothness,
            signal_weight=args.signal_weight,
            bending_weight=args.bending_weight,
            curvature_cv_weight=args.curvature_consistency_weight,
            min_candidate_score=args.min_candidate_score,
        )
        print(f"[mode {mi}] {len(segs)} segments → {len(fibers)} fibers")
        for f in fibers:
            all_fibers.append((f, md))
        if args.debug and fibers:
            save_mrc(
                out_prefix.parent / (out_prefix.name + f"_9_fibers_mode{mi}.mrc"),
                _draw_segments_to_volume(fibers, vol_shape, radius=2, bridge_gaps=True), voxel,
            )

    if not all_fibers:
        print("no fibers remain; exiting")
        return

    # ── 8. Sample + write STAR (merged across modes) ────────────────────────
    if max_step < 8:
        return
    spacing_px = float(args.spacing) / voxel
    all_coords: List[np.ndarray] = []
    all_angles: List[np.ndarray] = []
    all_cc_ids: List[np.ndarray] = []
    for fiber_idx, (fiber, md) in enumerate(all_fibers, start=1):
        pts, line_dir = fit_line_and_sample(fiber, md, spacing_px, pre_smoothed=True)
        if len(pts) == 0:
            continue
        rot, tilt, psi = vector_to_euler_zyz(line_dir)
        angs = np.tile([rot, tilt, psi], (len(pts), 1)).astype(np.float32)
        all_coords.append(pts)
        all_angles.append(angs)
        all_cc_ids.append(np.full(len(pts), fiber_idx, dtype=np.int32))

    if not all_coords:
        print("no particles sampled; STAR not written")
        return

    coords = np.vstack(all_coords).astype(np.float32)
    angles = np.vstack(all_angles).astype(np.float32)
    cc_ids = np.concatenate(all_cc_ids).astype(np.int32)
    star_path = out_prefix.parent / (out_prefix.name + "_particles.star")
    write_star(star_path, coords, angles, shift_z=args.shift_z, cc_ids=cc_ids)
    print(f"saved {len(coords)} particles → {star_path}  ({time.time()-t0:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────


def _looks_like_regex(s: str) -> bool:
    return bool(re.search(r"(\\.|\\d|\\D|\\w|\\W|\\s|\\S|[\^\$\+\?\|\(\)\[\]\{\}])", s))


def _resolve_paths(tomo_arg: str, recursive: bool) -> List[Path]:
    raw = Path(tomo_arg).expanduser()
    if not recursive:
        p = (raw if raw.is_absolute() else Path.cwd() / raw).resolve()
        if not p.exists():
            raise SystemExit(f"error: not found: {p}")
        return [p]

    if has_magic(tomo_arg):
        pattern = str(raw if raw.is_absolute() else Path.cwd() / raw)
        matched = [Path(p).resolve() for p in glob(pattern)]
    elif _looks_like_regex(tomo_arg):
        parent = (raw.parent if raw.is_absolute() else (Path.cwd() / raw.parent)).resolve()
        try:
            name_re = re.compile(raw.name)
        except re.error as e:
            raise SystemExit(f"error: bad regex: {e}")
        matched = [p.resolve() for p in parent.rglob("*.mrc") if name_re.fullmatch(p.name)]
    else:
        target = (raw if raw.is_absolute() else Path.cwd() / raw).resolve()
        matched = list(target.rglob("*.mrc")) if target.is_dir() else ([target] if target.exists() else [])

    paths = sorted(matched)
    if not paths:
        raise SystemExit(f"error: no files matched for --tomo={tomo_arg}")
    return paths


def _resolve_out_prefix(args: argparse.Namespace, tomo_path: Path, multi: bool) -> Path:
    if args.out_prefix is not None:
        p = Path(args.out_prefix)
        if not p.is_absolute():
            p = tomo_path.parent / p
        if multi:
            p = p.with_name(f"{p.name}_{tomo_path.stem}")
        return p
    return tomo_path.parent / tomo_path.stem


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="fiber2star_v2: K-mode fiber tracking from AIS prediction volumes to particles .star")
    p.add_argument("--tomo", required=True, help="input AIS prediction .mrc (0-255); supports glob/regex with --recursive")
    p.add_argument("--voxel-size", type=float, default=-1.0, help="voxel size in Å (default: read from MRC header)")
    p.add_argument("--bin", type=int, default=1, help="integer binning factor applied after load; subsequent steps and STAR coordinates live in binned space")
    p.add_argument("--sigma-z", type=float, default=1.5, help="Gaussian sigma along Z to bridge slice discontinuities (default 1.5)")
    p.add_argument("--sigma-xy", type=float, default=0.0, help="optional mild in-plane Gaussian sigma (default 0 = disabled)")
    p.add_argument("--high-threshold", type=float, default=254.0, help="threshold for K-mode direction estimation (default 254)")
    p.add_argument("--mode2-ratio-thr", type=float, default=0.25, help="trigger K=2 when λ2/λ1 of direction outer-product ≥ this (smaller = more sensitive, default 0.25)")
    p.add_argument("--threshold", type=float, default=252.0, help="threshold for main binary mask (default 252)")
    p.add_argument("--opening-radius", type=int, default=3, help="ball opening radius to remove thin bridges (default 3)")
    p.add_argument("--min-aspect-ratio", type=float, default=2.0, help="min length/cross-span aspect ratio for elongation filter (default 2)")
    p.add_argument("--direction-filter-angle", type=float, default=20.0, help="max angle from mode direction for CC/branch filtering (default 20°)")
    p.add_argument("--bridge-radius", type=int, default=3, help="dilation radius to bridge skeleton micro-gaps (default 3)")
    p.add_argument("--fiber-diameter", type=float, default=500.0, help="fiber diameter in Å used for local neighborhood radius (default 500)")
    p.add_argument("--erase-radius", type=float, default=-1.0, help="absolute erase radius in Å (default: auto from --erase-radius-scale)")
    p.add_argument("--erase-radius-scale", type=float, default=1.1, help="erase radius scale relative to fiber radius when --erase-radius<=0")
    p.add_argument("--min-fiber-length", type=float, default=1000.0, help="min arc length in Å for each greedy spline candidate (default 1000)")
    p.add_argument("--spline-smoothness", type=float, default=1.5, help="B-spline smoothing strength (larger = smoother, default 1.5)")
    p.add_argument("--signal-weight", type=float, default=15.0, help="weight of signal-confidence term in greedy score (default 15.0)")
    p.add_argument("--bending-weight", type=float, default=0.05, help="weight of B-spline bending-energy penalty (default 0.05)")
    p.add_argument("--curvature-consistency-weight", type=float, default=0.2, help="weight of curvature-consistency penalty (default 0.2)")
    p.add_argument("--min-candidate-score", type=float, default=0.0, help="stop extraction when best candidate score drops below this value")
    p.add_argument("--spacing", type=float, default=40, help="particle sampling spacing along fiber (Å, default 40)")
    p.add_argument("--shift-z", type=float, default=0, help="shift along z-axis in the output particles (voxel units, default 0)")
    p.add_argument("--out-prefix", default=None, help="output filename prefix (default: tomogram stem in same directory)")
    p.add_argument("--debug", action="store_true", help="save intermediate MRC files for each pipeline step")
    p.add_argument("--recursive", action="store_true", help="process multiple tomograms (glob/directory)")
    p.add_argument("--max-step", type=int, default=99, help="stop after this pipeline step (1..8; default 99 = run all)")
    args = p.parse_args()

    tomo_paths = _resolve_paths(args.tomo, recursive=args.recursive)
    multi = len(tomo_paths) > 1
    if multi:
        print(f"matched {len(tomo_paths)} tomograms")

    n_ok = n_fail = 0
    t_start = time.time()
    for idx, tomo_path in enumerate(tomo_paths, 1):
        print("=" * 88)
        print(f"[{idx}/{len(tomo_paths)}] {tomo_path}")
        out_prefix = _resolve_out_prefix(args, tomo_path, multi)
        try:
            _process_single_tomogram(args, tomo_path, out_prefix)
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            print(f"ERROR: {exc}")
            traceback.print_exc()

    print("=" * 88)
    print(f"done: {n_ok} ok, {n_fail} failed, total {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
