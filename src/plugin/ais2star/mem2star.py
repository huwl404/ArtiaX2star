#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mem2star_v2: AIS 膜预测 → RELION 颗粒（ridge-based 直接提取）。

与 v1 的关键差异
────────────────────────────────────────────────────────────────────────────
  v1 走 threshold -> mask -> fill -> EDT midband -> junction_cut;
  v2 跳过二值 mask 通路, 直接在概率图上提取脊 (prob 沿法向的局部极大)
     作为 midband。因此:
       - 不需要强阈值 (避免孔洞)
       - 不需要 tangent_plane_fill (ridge 自然覆盖有信号的区域)
       - Hessian 只做一次
       - sheet saliency (Hessian eigvals 各向异性) 过滤非膜结构 (线状/点状)

流程
────────────────────────────────────────────────────────────────────────────
  1. 读取 + bin + 强度平滑
  2. extract_midband_ridge:
       - ROI = prob > prob_min  (宽松, 只为限定 Hessian 计算域)
       - 逐 ROI-CC bbox 做 Hessian -> eigvals + normal (张量平滑)
       - sheet saliency S = max(0, 1 - |lam2|/|lam1|) in [0, 1]  (lam 升序, lam1 < 0)
       - ridge test: prob(p) 沿 +/-n(p) 三线性采样, p 是局部极大
       - midband = ROI 且 {S > tau_s} 且 {ridge}
  3. midband_junction_cut (切桥 + hole_fill + 再去小 CC)
  4. 定向: BFS 在 CC 内传播符号 + CC 级 divergence 符号决定整片朝向
  5. fps_gpu: GPU 欧几里得 FPS
  6. -> STAR (ZYZ 欧拉角)

Debug MRC（--debug 时，按流水线顺序保存）:
  _01_smoothed                    平滑后体积（sigma_z/xy）
  _02_roi_thresholded             prob_min 阈值后的 ROI
  _03a_saliency                   sheet saliency 场 (float)
  _03b_saliency_passed            saliency > 阈值的 mask
  _03c_ridge_passed               prob 沿 +/-n 是局部极大的 mask
  _04a_midband_intersected        saliency 且 ridge 的交集（去小 CC 前）
  _04b_midband_dropsmall          去小 CC 后的 midband
  _04c_midband_labels             对应 labels
  _04d_normals_ballstick          法向场 ball-stick 可视化
  _05_junction_field              桥检测场 J
  _06a_cut_thresholded            J > 阈值的 raw cut
  _06b_cut_separating_only        过滤掉非分离性假阳性后的 cut
  _07a_midband_cut                切桥后 midband（hole_fill 前）
  _07b_midband_holefilled         per-CC binary_closing 填洞后
  _07c_midband_dropsmall          hole_fill 后再次去小 CC
  _07d_midband_labels             drop_small 后 labels
  _07e_midband_smoothed           删毛刺后最终 midband（进 FPS 的输入）
  _07f_normals_oriented_bs        outward-flip 后的法向可视化
  _08_sample_points               FPS 采样后颗粒位置
────────────────────────────────────────────────────────────────────────────

数据驻留约定
  - vol / n_field / midband / labels 全程在 GPU（cp.ndarray）
  - H2D/D2H 只发生在 I/O（load/save MRC、STAR、debug 存盘）
  - 符号传播 BFS 依赖 Python dict，保留在 CPU，其余全部 GPU
"""
from __future__ import annotations

import argparse
import math
import re
import time
import traceback
import warnings
from dataclasses import dataclass
from glob import glob, has_magic
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cupy as cp
import mrcfile
import numpy as np
import pandas as pd
import starfile
from cucim import skimage as cskimage
from cupyx.scipy import ndimage as cp_ndimage
from scipy.ndimage import find_objects as _cpu_find_objects

warnings.filterwarnings(
    action="ignore",
    message=r".*cupyx\.jit\.rawkernel is experimental.*",
    category=FutureWarning,
    module="cupyx.jit._interface",
)


@dataclass(frozen=True)
class NormalEstimationResult:
    """法向估计结果（CPU numpy，供下游采样使用），统一 ZYX 索引顺序。"""
    coords_zyx: np.ndarray
    normals_zyx: np.ndarray
    confidence: np.ndarray
    labels: np.ndarray  # int32, per-candidate CC label (0 = background)


# ─────────────────────────────────────────────────────────────────────────────
# I/O —— 唯一的 np <-> cp 转换边界
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


def save_mrc(path: Path, data: np.ndarray, voxel_size: float = 1.0) -> None:
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(np.asarray(data, dtype=np.float32))
        try:
            m.voxel_size = float(voxel_size)
        except Exception:
            pass


def save_mrc_gpu(path: Path, data_gpu: cp.ndarray, voxel_size: float = 1.0) -> None:
    """GPU 数据存盘（仅 debug 通道；D2H 开销可接受）。"""
    save_mrc(path, cp.asnumpy(data_gpu).astype(np.float32), voxel_size)


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
        "rlnAngleRot": a[:, 0],
        "rlnAngleTilt": a[:, 1],
        "rlnAnglePsi": a[:, 2],
    }
    if cc_ids is not None:
        data["aisCcId"] = np.asarray(cc_ids, dtype=np.int32)
    df = pd.DataFrame(data)
    starfile.write({"particles": df}, out_path, overwrite=True)


def vector_to_euler_zyz(vec_xyz: np.ndarray) -> Tuple[float, float, float]:
    """
    单位向量 (颗粒 z 轴在 lab 的方向) -> RELION ZYZ 欧拉角 (Rot, Tilt, Psi)。

    RELION 约定 (见 src/euler.cpp::Euler_angles2matrix): 矩阵 A 把 reference 旋到
    lab, 第三列 A·(0,0,1) = (−cos(psi)·sin(tilt), sin(psi)·sin(tilt), cos(tilt))。
    注意方位角由 PSI 决定, 不是 ROT; 这是 ZYZ 内 / 外旋分解造成的 "反直觉" 项。

    解 A·(0,0,1) = v:
        tilt = acos(vz)
        psi  = atan2(vy, −vx)        # 注意 −vx
        rot  = 0                     # 自由参数 (绕粒子自身 Z 的 in-plane), 膜蛋白 C_n 对称无约束

    过去把方位角放进 rot (psi=0) 会让粒子 Z 恒落在 lab 的 XZ 平面, 法向有 Y
    分量时就旋成"切平面方向", 看起来像颗粒 Z 在膜上逐渐倒伏。
    """
    v = np.asarray(vec_xyz, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-6:
        return 0.0, 0.0, 0.0
    v /= n
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, vz))))
    if abs(vx) < 1e-9 and abs(vy) < 1e-9:
        psi = 0.0
    else:
        psi = math.degrees(math.atan2(vy, -vx))
    return 0.0, tilt, psi


# ─────────────────────────────────────────────────────────────────────────────
# 路径解析（与 fiber2star.py 对齐）
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
# GPU 通用工具
# ─────────────────────────────────────────────────────────────────────────────


def _label_bboxes(labels_gpu: cp.ndarray, pad: int = 0) -> List[Optional[Tuple[slice, slice, slice]]]:
    """一次性求所有 label 的 bbox。

    cucim 的 find_objects 不总是稳定；这里 D2H 一次 labels（int32）后用 scipy
    计算，比"每个 label 一次 argwhere"快得多。

    返回列表 bboxes，bboxes[lab] 即 label=lab 的 bbox（lab>=1）；
    bboxes[0] 占位为 None。缺失 label 对应位置也为 None。
    """
    labels_np = cp.asnumpy(labels_gpu)
    raw = _cpu_find_objects(labels_np)
    shape = labels_np.shape
    result: List[Optional[Tuple[slice, slice, slice]]] = [None]
    for s in raw:
        if s is None:
            result.append(None)
            continue
        if pad <= 0:
            result.append(s)
        else:
            p = int(pad)
            result.append(tuple(
                slice(max(ax.start - p, 0), min(ax.stop + p, shape[i]))
                for i, ax in enumerate(s)
            ))
    return result


def _densify_labels(labels_gpu: cp.ndarray) -> cp.ndarray:
    """重排 label 为连续 1..n_lab（便于后续定长数组操作）。"""
    unique_labs = cp.unique(labels_gpu[labels_gpu > 0])
    if int(unique_labs.size) == 0:
        return labels_gpu
    n_lab = int(unique_labs.size)
    max_old = int(unique_labs.max().get())
    lut = cp.zeros(max_old + 1, dtype=cp.int32)
    lut[unique_labs] = cp.arange(1, n_lab + 1, dtype=cp.int32)
    out = lut[labels_gpu]
    del lut, unique_labs
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 预处理（全 GPU）
# ─────────────────────────────────────────────────────────────────────────────


def smooth_intensity(
    vol_gpu: cp.ndarray,
    sigma_z: float,
    sigma_xy: float = 0.0,
    debug_prefix: Optional[Path] = None,
    voxel: float = 1.0,
) -> cp.ndarray:
    """沿 Z / XY 做可分离 1D 高斯平滑（等价于可分离 3D 高斯）。"""
    v = vol_gpu
    if sigma_z > 0:
        v = cp_ndimage.gaussian_filter1d(v, sigma=sigma_z, axis=0)
    if sigma_xy > 0:
        v = cp_ndimage.gaussian_filter1d(v, sigma=sigma_xy, axis=1)
        v = cp_ndimage.gaussian_filter1d(v, sigma=sigma_xy, axis=2)
    if debug_prefix is not None:
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_01_smoothed.mrc"), v, voxel)
    return v


# ─────────────────────────────────────────────────────────────────────────────
# 法向场（Hessian → 原始法向 → 张量平滑）
# ─────────────────────────────────────────────────────────────────────────────


def _tensor_smooth_normals(
    iz_p: cp.ndarray,
    iy_p: cp.ndarray,
    ix_p: cp.ndarray,
    normals: cp.ndarray,
    local_shape: Tuple[int, int, int],
    smooth_sigma: float,
) -> cp.ndarray:
    """
    sign-agnostic 张量平滑。

    原理：法向 n 与 -n 等价 → 不能直接高斯平均。改为平滑外积张量 n⊗n
         （6 个独立分量），再对平滑后张量做特征分解：
         最大特征向量 = 去噪后的法向。

    返回 normals_clean (N, 3)。
    """
    nss = max(float(smooth_sigma), 0.1)
    Z_l, Y_l, X_l = local_shape
    N_pl = int(iz_p.shape[0])

    nn_field = cp.zeros((6, Z_l, Y_l, X_l), dtype=cp.float32)
    weight_field = cp.zeros((Z_l, Y_l, X_l), dtype=cp.float32)
    nz, ny, nx = normals[:, 0], normals[:, 1], normals[:, 2]
    nn_field[0, iz_p, iy_p, ix_p] = nz * nz
    nn_field[1, iz_p, iy_p, ix_p] = nz * ny
    nn_field[2, iz_p, iy_p, ix_p] = nz * nx
    nn_field[3, iz_p, iy_p, ix_p] = ny * ny
    nn_field[4, iz_p, iy_p, ix_p] = ny * nx
    nn_field[5, iz_p, iy_p, ix_p] = nx * nx
    weight_field[iz_p, iy_p, ix_p] = 1.0
    del nz, ny, nx

    for i in range(6):
        nn_field[i] = cp_ndimage.gaussian_filter(nn_field[i], sigma=nss, mode="constant", cval=0.0)
    weight_field = cp_ndimage.gaussian_filter(weight_field, sigma=nss, mode="constant", cval=0.0)
    weight_field = cp.maximum(weight_field, 1e-6)
    for i in range(6):
        nn_field[i] /= weight_field
    del weight_field

    mats = cp.empty((N_pl, 3, 3), dtype=cp.float32)
    mats[:, 0, 0] = nn_field[0, iz_p, iy_p, ix_p]
    a01 = nn_field[1, iz_p, iy_p, ix_p]; mats[:, 0, 1] = a01; mats[:, 1, 0] = a01
    a02 = nn_field[2, iz_p, iy_p, ix_p]; mats[:, 0, 2] = a02; mats[:, 2, 0] = a02
    mats[:, 1, 1] = nn_field[3, iz_p, iy_p, ix_p]
    a12 = nn_field[4, iz_p, iy_p, ix_p]; mats[:, 1, 2] = a12; mats[:, 2, 1] = a12
    mats[:, 2, 2] = nn_field[5, iz_p, iy_p, ix_p]
    del nn_field, a01, a02, a12
    cp.get_default_memory_pool().free_all_blocks()

    # eigh 按升序返回特征值 → 最大特征向量 = evecs[:, :, 2]
    _, evecs = cp.linalg.eigh(mats)
    del mats
    normals_clean = evecs[:, :, 2]
    normals_clean /= cp.maximum(cp.linalg.norm(normals_clean, axis=1, keepdims=True), 1e-8)
    del evecs
    return normals_clean


# ─────────────────────────────────────────────────────────────────────────────
# Ridge-based midband 提取（概率图沿法向的局部极大）
# ─────────────────────────────────────────────────────────────────────────────


def extract_midband_ridge(
    vol_gpu: cp.ndarray,
    hessian_sigma: float,
    normal_smooth_sigma: float,
    prob_min: float,
    sheet_saliency_threshold: float,
    border_width: int,
    min_component_size: int,
    debug_prefix: Optional[Path] = None,
    voxel: float = 1.0,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    从 AIS 概率图抽 midband (膜中面, 1-vox 厚 2D 流形), 替代 v1 的
    threshold + fill + EDT midband 链路。

    什么是 midband / ridge
    ----------------------
    AIS 膜预测是 bright-on-dark 概率场, 沿膜法向 prob(p) 呈尖峰 (峰值
    落在膜的几何中心, 两侧快速衰减)。midband 就是这些沿法向局部极大点
    构成的 1-vox 厚中面, 也叫 "ridge" (概率图的脊)。
    相比 v1:
      - 不需要二值化阈值 (ridge 只在膜脊响应, 阈值敏感度降低)
      - 孔洞处 prob 偏低 -> ridge 自然不响应 -> 无需二次 fill
      - Hessian 只算一次, sheet saliency 用各向异性而非幅度 -> 跨 tomo 可泛化

    算法 (逐 ROI-CC bbox, 省 Hessian 全局计算的内存):
      1. ROI = prob > prob_min  (宽松, 仅限 Hessian 计算域)
      2. Hessian 3x3 特征分解 (升序 lam1 <= lam2 <= lam3):
           - normal = max-|eigval| 对应的 eigenvector
           - 再做 n(x)n 张量平滑 (sign-agnostic) 去噪
      3. Sheet saliency:
           S = max(0, 1 - |lam2|/|lam1|),  当 lam1 < 0
         - 纯片状 (|lam2| << |lam1|) -> S ~ 1
         - 各向同性 (|lam2| ~ |lam1|) -> S ~ 0   (抑制线状/点状)
         - 与 Hessian 幅度无关, 跨 tomo 可比
      4. Ridge test (严格沿法向局部极大, 无参数):
           prob(p) >= prob(p + n)  且  prob(p) >= prob(p - n)
         - step 固定为 1 vox, 无 contrast 松弛
         - 过滤非膜结构的主力是 sheet saliency (纤维 S~0 已被抑制);
           ridge test 仅作"沿法向单峰"最终确认, 无需暴露参数
      5. midband = ROI 且 {S > tau_s} 且 {ridge_peak}
      6. 26-conn 重标号, 丢弃 < min_component_size 的小 CC

    返回 (midband_mask, midband_labels, n_field_gpu)。
    """
    Z, Y, X = vol_gpu.shape
    midband = cp.zeros((Z, Y, X), dtype=bool)
    n_field = cp.zeros((3, Z, Y, X), dtype=cp.float32)

    # 1. 宽松 ROI 限定 Hessian 域
    roi = vol_gpu > cp.float32(prob_min)
    if border_width > 0:
        bw = int(border_width)
        roi[:bw, :, :] = False; roi[-bw:, :, :] = False
        roi[:, :bw, :] = False; roi[:, -bw:, :] = False
        roi[:, :, :bw] = False; roi[:, :, -bw:] = False
    if not bool(roi.any().get()):
        empty_labels = cp.zeros((Z, Y, X), dtype=cp.int32)
        return midband, empty_labels, n_field
    if debug_prefix is not None:
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_02_roi_thresholded.mrc"), roi.astype(cp.float32), voxel)

    roi_labels = cskimage.measure.label(roi.astype(cp.uint8), connectivity=1)
    n_lab = int(roi_labels.max().get())
    del roi

    hsig = max(float(hessian_sigma), 0.5)
    nss = max(float(normal_smooth_sigma), 0.1)
    pad = int(max(2, math.ceil(3.0 * hsig + 3.0 * nss + 2.0)))
    bboxes = _label_bboxes(roi_labels, pad=pad)
    if debug_prefix is not None:
        saliency_dbg = cp.zeros((Z, Y, X), dtype=cp.float32)
        saliency_passed_dbg = cp.zeros((Z, Y, X), dtype=bool)
        ridge_passed_dbg = cp.zeros((Z, Y, X), dtype=bool)
    else:
        saliency_dbg = saliency_passed_dbg = ridge_passed_dbg = None

    for lab in range(1, n_lab + 1):
        bbox = bboxes[lab]
        if bbox is None:
            continue
        comp = roi_labels[bbox] == lab
        if int(comp.sum().get()) < 10:
            continue
        vol_local = vol_gpu[bbox]

        # 2. Hessian + eigendecomposition at ROI voxels
        Hzz, Hzy, Hzx, Hyy, Hyx, Hxx = cskimage.feature.hessian_matrix(
            vol_local, sigma=hsig, use_gaussian_derivatives=True,
        )
        fg = cp.argwhere(comp)
        iz, iy, ix = fg[:, 0], fg[:, 1], fg[:, 2]
        N = int(iz.shape[0])

        mats = cp.empty((N, 3, 3), dtype=cp.float32)
        mats[:, 0, 0] = Hzz[iz, iy, ix]
        a01 = Hzy[iz, iy, ix]; mats[:, 0, 1] = a01; mats[:, 1, 0] = a01
        a02 = Hzx[iz, iy, ix]; mats[:, 0, 2] = a02; mats[:, 2, 0] = a02
        mats[:, 1, 1] = Hyy[iz, iy, ix]
        a12 = Hyx[iz, iy, ix]; mats[:, 1, 2] = a12; mats[:, 2, 1] = a12
        mats[:, 2, 2] = Hxx[iz, iy, ix]
        del Hzz, Hzy, Hzx, Hyy, Hyx, Hxx, a01, a02, a12

        evals, evecs = cp.linalg.eigh(mats)  # ascending: lam1 <= lam2 <= lam3
        del mats

        # 3. Normal = eigvec at largest |eigval|; 张量平滑去噪
        abs_ev = cp.abs(evals)
        idx_max = cp.argmax(abs_ev, axis=1)
        rows = cp.arange(N)
        normals_raw = evecs[rows, :, idx_max]
        normals_raw /= cp.maximum(cp.linalg.norm(normals_raw, axis=1, keepdims=True), 1e-8)
        del evecs, abs_ev, idx_max, rows

        normals_clean = _tensor_smooth_normals(iz, iy, ix, normals_raw, comp.shape, nss)
        del normals_raw, comp

        # 4. Sheet saliency (bright sheet: lam1 < 0, lam2/lam3 小)
        # S = 1 - |lam2|/|lam1| in [0, 1]: 纯片 -> 1, 各向同性 -> 0
        # 纯 anisotropy, 与 Hessian 幅度无关 -> 跨 tomo 可泛化
        lam1 = evals[:, 0]
        lam2 = evals[:, 1]
        lam1_abs = cp.abs(lam1)
        lam2_abs = cp.abs(lam2)
        sheet = cp.where(
            lam1 < 0,
            cp.maximum(cp.float32(0.0), cp.float32(1.0) - lam2_abs / cp.maximum(lam1_abs, cp.float32(1e-6))),
            cp.float32(0.0),
        )
        del evals, lam1, lam2, lam1_abs, lam2_abs
        saliency_passes = sheet > cp.float32(sheet_saliency_threshold)

        # 5. Ridge test: 严格局部极大 (prob(p) >= prob(p ± n), step=1 vox)
        iz_f = iz.astype(cp.float32)
        iy_f = iy.astype(cp.float32)
        ix_f = ix.astype(cp.float32)
        plus = cp.stack([iz_f + normals_clean[:, 0],
                         iy_f + normals_clean[:, 1],
                         ix_f + normals_clean[:, 2]], axis=0)
        minus = cp.stack([iz_f - normals_clean[:, 0],
                          iy_f - normals_clean[:, 1],
                          ix_f - normals_clean[:, 2]], axis=0)
        prob_c = vol_local[iz, iy, ix]
        prob_p = cp_ndimage.map_coordinates(vol_local, plus, order=1, mode="nearest")
        prob_m = cp_ndimage.map_coordinates(vol_local, minus, order=1, mode="nearest")
        del plus, minus, iz_f, iy_f, ix_f, vol_local
        is_peak = (prob_c >= prob_p) & (prob_c >= prob_m)
        del prob_c, prob_p, prob_m

        # 6. Scatter normals + midband voxels 到全局
        z0, y0, x0 = bbox[0].start, bbox[1].start, bbox[2].start
        giz = iz + z0; giy = iy + y0; gix = ix + x0
        n_field[0, giz, giy, gix] = normals_clean[:, 0]
        n_field[1, giz, giy, gix] = normals_clean[:, 1]
        n_field[2, giz, giy, gix] = normals_clean[:, 2]
        if saliency_dbg is not None:
            saliency_dbg[giz, giy, gix] = sheet
            saliency_passed_dbg[giz[saliency_passes], giy[saliency_passes], gix[saliency_passes]] = True
            ridge_passed_dbg[giz[is_peak], giy[is_peak], gix[is_peak]] = True
        del normals_clean, sheet

        keep = saliency_passes & is_peak
        midband[giz[keep], giy[keep], gix[keep]] = True
        del saliency_passes, is_peak, keep, iz, iy, ix, giz, giy, gix
        cp.get_default_memory_pool().free_all_blocks()

    del roi_labels

    if debug_prefix is not None:
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_03a_saliency.mrc"), saliency_dbg, voxel)
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_03b_saliency_passed.mrc"), saliency_passed_dbg.astype(cp.float32), voxel)
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_03c_ridge_passed.mrc"), ridge_passed_dbg.astype(cp.float32), voxel)
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_04a_midband_intersected.mrc"), midband.astype(cp.float32), voxel)
        del saliency_dbg, saliency_passed_dbg, ridge_passed_dbg

    # 7. Drop small CCs, 26-conn 重标号
    labels = cskimage.measure.label(midband.astype(cp.uint8), connectivity=3)
    counts = cp.bincount(labels.ravel().astype(cp.int64))
    small = counts < int(max(min_component_size, 1))
    small[0] = False
    to_drop = small[labels]
    midband = midband & ~to_drop
    labels = cp.where(to_drop, cp.int32(0), labels)
    labels = _densify_labels(labels)
    del counts, small, to_drop

    if debug_prefix is not None:
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_04b_midband_dropsmall.mrc"), midband.astype(cp.float32), voxel)
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_04c_midband_labels.mrc"), labels.astype(cp.float32), voxel)
        bs = draw_normals_ball_stick(n_field)
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_04d_normals_ballstick.mrc"), bs, voxel)
        del bs

    cp.get_default_memory_pool().free_all_blocks()
    return midband, labels, n_field


# ─────────────────────────────────────────────────────────────────────────────
# 张量特征值桥检测（n⊗n 结构张量的 λ₂/λ₁，在 midband 上切）
# ─────────────────────────────────────────────────────────────────────────────


def _normal_junction_field(
    n_field_gpu: cp.ndarray,
    eval_mask_gpu: cp.ndarray,
    sigma: float,
) -> cp.ndarray:
    """
    结构张量桥检测场：J(p) = λ₂(T) / λ₁(T)，其中
      T(p) = E_{q∈N_σ(p)}[ n(q) ⊗ n(q) ]
    是 n⊗n 在尺度 σ 下按 valid-normal 权重归一的空间 Gaussian 平滑张量。

    - 光滑膜（含高曲率）：邻域所有 n 同向 → T ≈ n·nᵀ (rank 1)
      → λ₁ ≈ 1, λ₂ ≈ 0 → J ≈ 0
    - 膜截面边缘：邻域仍是同一张膜 → T 近 rank 1 → J ≈ 0（无假阳性）
    - 两膜桥接：T ≈ α·n_A n_Aᵀ + β·n_B n_Bᵀ (rank 2) → λ₁ ≈ λ₂ → J → 1

    张量平滑跑在全体积（w-normalize 消除边界效应），特征值只在 eval_mask_gpu
    上算。使用 3×3 对称矩阵的 Smith 解析公式（纯逐元素 cupy），避开
    cuSolver syevj_batched 的大 workspace（实测可达 6 GB+）。
    """
    n = n_field_gpu
    _, Z, Y, X = n.shape

    nmag = cp.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    valid = (nmag > 0.5).astype(cp.float32)
    del nmag

    nn = cp.empty((6, Z, Y, X), dtype=cp.float32)
    nn[0] = n[0] * n[0] * valid   # zz
    nn[1] = n[0] * n[1] * valid   # zy
    nn[2] = n[0] * n[2] * valid   # zx
    nn[3] = n[1] * n[1] * valid   # yy
    nn[4] = n[1] * n[2] * valid   # yx
    nn[5] = n[2] * n[2] * valid   # xx

    sig = max(float(sigma), 0.1)
    for i in range(6):
        nn[i] = cp_ndimage.gaussian_filter(nn[i], sigma=sig, mode="constant", cval=0.0)
    w = cp_ndimage.gaussian_filter(valid, sigma=sig, mode="constant", cval=0.0)
    w = cp.maximum(w, 1e-6)
    for i in range(6):
        nn[i] = nn[i] / w
    del w, valid

    coords = cp.argwhere(eval_mask_gpu)
    N = int(coords.shape[0])
    J = cp.zeros((Z, Y, X), dtype=cp.float32)
    if N == 0:
        del nn
        return J

    # 分块：峰值内存 ~ 10 * chunk * 4 bytes。chunk=2M → ~80 MB 额外
    chunk = 2_000_000
    third = cp.float32(1.0 / 3.0)
    sixth = cp.float32(1.0 / 6.0)
    two_third_pi = cp.float32(2.0 * math.pi / 3.0)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        iz_c = coords[start:end, 0]
        iy_c = coords[start:end, 1]
        ix_c = coords[start:end, 2]

        Azz = nn[0, iz_c, iy_c, ix_c]
        Azy = nn[1, iz_c, iy_c, ix_c]
        Azx = nn[2, iz_c, iy_c, ix_c]
        Ayy = nn[3, iz_c, iy_c, ix_c]
        Ayx = nn[4, iz_c, iy_c, ix_c]
        Axx = nn[5, iz_c, iy_c, ix_c]

        # Smith 解析公式（3×3 对称矩阵特征值，降序），
        # ref: https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3×3_matrices
        p1 = Azy * Azy + Azx * Azx + Ayx * Ayx
        q = (Azz + Ayy + Axx) * third
        Mzz = Azz - q; Myy = Ayy - q; Mxx = Axx - q
        p2 = Mzz * Mzz + Myy * Myy + Mxx * Mxx + cp.float32(2.0) * p1
        del p1
        p = cp.sqrt(p2 * sixth); del p2
        p_safe = cp.maximum(p, cp.float32(1e-12))

        detM = (Mzz * (Myy * Mxx - Ayx * Ayx)
                - Azy * (Azy * Mxx - Ayx * Azx)
                + Azx * (Azy * Ayx - Myy * Azx))
        del Mzz, Myy, Mxx, Azz, Azy, Azx, Ayy, Ayx, Axx
        r = cp.clip(detM / (cp.float32(2.0) * p_safe * p_safe * p_safe),
                    cp.float32(-1.0), cp.float32(1.0))
        del detM, p_safe
        phi = cp.arccos(r) * third
        del r

        two_p = cp.float32(2.0) * p
        del p
        lam1 = q + two_p * cp.cos(phi)
        lam3 = q + two_p * cp.cos(phi + two_third_pi)
        del two_p, phi
        lam2 = cp.float32(3.0) * q - lam1 - lam3
        del lam3, q

        J[iz_c, iy_c, ix_c] = lam2 / cp.maximum(lam1, cp.float32(1e-6))
        del lam1, lam2, iz_c, iy_c, ix_c

    del nn, coords
    cp.get_default_memory_pool().free_all_blocks()
    return J


def _keep_only_separating_cuts(
    midband_before_gpu: cp.ndarray,
    cut_voxels_gpu: cp.ndarray,
    min_component_size: int,
) -> cp.ndarray:
    """
    只保留"真切且至少切出两个大碎片"的 cut 簇, 其余还回。

    J 阈值不可避免在高曲率处产生假阳性。两类应还回 (视为假阳性):
      (a) 不拓扑分离: cut 团紧邻的 midband CC < 2 个 -> 拿掉只是在整片
          上挖孔, 不隔开任何东西
      (b) 切不出大碎片: 紧邻的 CC 都 < min_component_size -> 桥两侧
          都是小片, 切了也没意义 (后续 drop_small 会连带删掉)

    (邻接 CC >= 2) 且 (至少两个邻接 CC >= min_component_size) 才保留切;
    切出的 < min_component_size 的小碎片交给下游 drop_small_ccs 清理。
    """
    midband_after = midband_before_gpu & ~cut_voxels_gpu
    cut_labels = cskimage.measure.label(cut_voxels_gpu.astype(cp.uint8), connectivity=3)
    mid_labels = cskimage.measure.label(midband_after.astype(cp.uint8), connectivity=3)
    n_cuts = int(cut_labels.max().get())
    if n_cuts == 0:
        return cut_voxels_gpu

    mid_sizes = cp.bincount(mid_labels.ravel().astype(cp.int64))
    cut_bboxes = _label_bboxes(cut_labels, pad=1)
    restore = cp.zeros_like(cut_voxels_gpu)
    min_sz = int(max(min_component_size, 1))

    for cid in range(1, n_cuts + 1):
        bbox = cut_bboxes[cid]
        if bbox is None:
            continue
        cut_patch = cut_labels[bbox] == cid
        mid_patch = mid_labels[bbox]
        dilated = cp_ndimage.binary_dilation(cut_patch)
        adj = cp.unique(mid_patch[dilated & ~cut_patch])
        adj = adj[adj > 0]
        n_adj = int(adj.size)
        if n_adj < 2:
            restore[bbox] |= cut_patch
            continue
        adj_sizes = mid_sizes[adj]
        if len(adj_sizes >= min_sz) < 2:
            restore[bbox] |= cut_patch

    return cut_voxels_gpu & ~restore


def _fill_midband_holes_per_cc(
    midband_gpu: cp.ndarray,
    labels_gpu: cp.ndarray,
    close_iters: int,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    逐 CC 做 3D binary_closing 填补 midband 孔洞；`labels==0` 守卫确保不跨 CC 桥接。
    新填体素继承该 CC 的 label。close_iters=r 能弥合约 ~2r vox 宽的孔。
    """
    iters = int(max(close_iters, 0))
    if iters == 0:
        return midband_gpu, labels_gpu
    n_lab = int(labels_gpu.max().get())
    if n_lab == 0:
        return midband_gpu, labels_gpu
    bboxes = _label_bboxes(labels_gpu, pad=iters + 1)
    for lab in range(1, n_lab + 1):
        bbox = bboxes[lab]
        if bbox is None:
            continue
        comp = labels_gpu[bbox] == lab
        closed = cp_ndimage.binary_closing(comp, iterations=iters, brute_force=True)
        new_fill = closed & ~comp & (labels_gpu[bbox] == 0)
        midband_gpu[bbox] |= new_fill
        labels_gpu[bbox] = cp.where(new_fill, cp.int32(lab), labels_gpu[bbox])
        del comp, closed, new_fill
    cp.get_default_memory_pool().free_all_blocks()
    return midband_gpu, labels_gpu


def _drop_small_ccs(
    mask_gpu: cp.ndarray,
    min_size: int,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """26-conn 重标号, 直接删除 < min_size 体素的 CC。返回 (mask_new, labels)。"""
    labels = cskimage.measure.label(mask_gpu.astype(cp.uint8), connectivity=3)
    counts = cp.bincount(labels.ravel().astype(cp.int64))
    small = counts < int(max(min_size, 1))
    small[0] = False
    to_drop = small[labels]
    mask_new = mask_gpu & ~to_drop
    labels = cp.where(to_drop, cp.int32(0), labels)
    labels = _densify_labels(labels)
    del counts, small, to_drop
    return mask_new, labels


def smooth_midband(
    midband_gpu: cp.ndarray,
    debug_prefix: Optional[Path] = None,
    voxel: float = 1.0,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    采样前对 midband 做几何平滑: 按 3×3×3 邻居计数删毛刺 (2 轮迭代).

    midband 是 1-vox 厚 2D 流形, 平坦区体素 count≈9 (self + 8 同平面),
    边缘 5-6, 毛刺 1-2。count < 4 的是毛刺 / 孤立小团 -> 删除。只做删除、
    不做加入, 所以不会让流形变厚, 也不会桥接 CC。

    迭代两次让"次级毛刺"(去毛刺后暴露的 count<4 新点) 也被清理。

    平滑可能切断薄桥 -> 重新 label (connectivity=3) 返回新 labels。
    """
    kernel = cp.ones((3, 3, 3), dtype=cp.int32)
    mask = midband_gpu
    for _ in range(2):
        count = cp_ndimage.convolve(mask.astype(cp.int32), kernel, mode="constant", cval=0)
        mask = mask & (count >= 4)
    labels = cskimage.measure.label(mask.astype(cp.uint8), connectivity=3)
    labels = _densify_labels(labels)
    if debug_prefix is not None:
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_07e_midband_smoothed.mrc"), mask.astype(cp.float32), voxel)
    cp.get_default_memory_pool().free_all_blocks()
    return mask, labels


def midband_junction_cut(
    midband_mask_gpu: cp.ndarray,
    full_mask_gpu: cp.ndarray,
    n_field_gpu: cp.ndarray,
    junction_sigma: float,
    junction_threshold: float,
    min_component_size: int,
    hole_fill_iters: int = 0,
    debug_prefix: Optional[Path] = None,
    voxel: float = 1.0,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    在 midband (1-vox 厚中面流形) 上用 J 场切桥。流程:
      1. J = 法向结构张量 lam2/lam1 (两膜相接处 rank-2 -> J 高, 单膜 rank-1 -> J 低)
      2. cut_raw = midband 且 (J > threshold)
      3. cut_kept = _keep_only_separating_cuts(cut_raw, min_component_size):
           只保留 (邻接 CC >= 2) 且 (至少两个邻接 CC >= min_component_size) 的 cut
      4. midband_cut = midband 且 非 cut_kept
      5. 26-conn 重标号
      6. per-CC binary_closing 填洞 (hole_fill_iters 次; label==0 守卫防跨 CC)
      7. drop small CC: hole_fill 后再次 label, 直接删 < min_component_size 的 CC

    返回 (midband_mask_final, midband_labels_final)。
    """
    if not bool(midband_mask_gpu.any().get()):
        empty_labels = cp.zeros(midband_mask_gpu.shape, dtype=cp.int32)
        return midband_mask_gpu, empty_labels

    # 1. J 场
    eval_mask = full_mask_gpu | midband_mask_gpu
    J = _normal_junction_field(n_field_gpu, eval_mask, junction_sigma)
    del eval_mask
    if debug_prefix is not None:
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_05_junction_field.mrc"), J, voxel)

    # 2. J > threshold raw cut
    cut_raw = midband_mask_gpu & (J > cp.float32(junction_threshold))
    del J
    if debug_prefix is not None:
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_06a_cut_thresholded.mrc"), cut_raw.astype(cp.float32), voxel)

    # 3. 过滤非分离性 / 切出小碎片的假阳性
    if bool(cut_raw.any().get()):
        cut_kept = _keep_only_separating_cuts(midband_mask_gpu, cut_raw, min_component_size)
    else:
        cut_kept = cut_raw
    del cut_raw
    if debug_prefix is not None:
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_06b_cut_separating_only.mrc"), cut_kept.astype(cp.float32), voxel)

    # 4. 执行切桥
    midband_cut = midband_mask_gpu & ~cut_kept
    del cut_kept
    if debug_prefix is not None:
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_07a_midband_cut.mrc"), midband_cut.astype(cp.float32), voxel)

    # 5. 重 label
    labels_cut = cskimage.measure.label(midband_cut.astype(cp.uint8), connectivity=3)
    labels_cut = _densify_labels(labels_cut)

    # 6. hole fill per CC
    midband_holefilled, _ = _fill_midband_holes_per_cc(midband_cut, labels_cut, hole_fill_iters)
    del midband_cut, labels_cut
    if debug_prefix is not None:
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_07b_midband_holefilled.mrc"), midband_holefilled.astype(cp.float32), voxel)

    # 7. drop small (直接删)
    midband_final, labels_final = _drop_small_ccs(midband_holefilled, min_component_size)
    del midband_holefilled
    if debug_prefix is not None:
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_07c_midband_dropsmall.mrc"), midband_final.astype(cp.float32), voxel)
        save_mrc_gpu(debug_prefix.parent / (debug_prefix.name + "_07d_midband_labels.mrc"), labels_final.astype(cp.float32), voxel)

    cp.get_default_memory_pool().free_all_blocks()
    return midband_final, labels_final


# ─────────────────────────────────────────────────────────────────────────────
# Outward 定向：BFS 同 CC 内符号一致 + 质心锚定 CC 翻转
# ─────────────────────────────────────────────────────────────────────────────


def _gather_normals_at_coords(
    coords_gpu: cp.ndarray,
    n_field_gpu: cp.ndarray,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """从法向场中 gather 给定体素坐标的单位法向；valid 由范数判断。"""
    zz, yy, xx = coords_gpu[:, 0], coords_gpu[:, 1], coords_gpu[:, 2]
    n_zyx = cp.stack([
        n_field_gpu[0, zz, yy, xx],
        n_field_gpu[1, zz, yy, xx],
        n_field_gpu[2, zz, yy, xx],
    ], axis=1)
    norm = cp.linalg.norm(n_zyx, axis=1, keepdims=True)
    valid = (norm[:, 0] > 1e-3)
    n_zyx = n_zyx / cp.maximum(norm, 1e-8)
    return n_zyx, valid


def _flip_cc_by_divergence(
    coords_np: np.ndarray,
    normals_np: np.ndarray,
    labels_np: np.ndarray,
) -> np.ndarray:
    """
    按 CC 的离散 divergence 符号决定是否整片翻: div(n) = 2H (平均曲率),
    n 指凸面 (远离曲率中心) -> 法向 "散开" -> div > 0; 指凹面 -> "汇聚"
    -> div < 0。对开放浅弧 / 深弧 / 闭合囊泡 / 管都成立, 不依赖 CC 质心
    位置 (旧 centroid 法对浅弧会误判: centroid 偏向凸面, 造成 dot 符号反转)。

    点云离散化: 对每个 midband 点 p, 在同 CC 26-conn midband 邻居 q 上
    求 S(p) = sum_q (n_q - n_p) . (q - p)。S(p) 对应 2H 的体素尺度离散,
    同 CC 符号一致。CC-level sum(S) < 0 则翻整片。
    """
    out = normals_np.astype(np.float32, copy=True)
    N = len(coords_np)
    if N == 0:
        return out

    coord_labs = labels_np[coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]]
    idx_map: Dict[Tuple[int, int, int], int] = {tuple(c): i for i, c in enumerate(coords_np)}

    div_per_pt = np.zeros(N, dtype=np.float32)
    for i in range(N):
        z, y, x = coords_np[i]
        lab_i = int(coord_labs[i])
        np0, np1, np2 = float(out[i, 0]), float(out[i, 1]), float(out[i, 2])
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    j = idx_map.get((int(z + dz), int(y + dy), int(x + dx)))
                    if j is None or int(coord_labs[j]) != lab_i:
                        continue
                    dn0 = out[j, 0] - np0
                    dn1 = out[j, 1] - np1
                    dn2 = out[j, 2] - np2
                    div_per_pt[i] += dn0 * dz + dn1 * dy + dn2 * dx

    for lab in np.unique(coord_labs):
        if lab == 0:
            continue
        sel = coord_labs == lab
        if float(div_per_pt[sel].sum()) < 0.0:
            out[sel] *= -1.0
    return out


def _propagate_component_sign(
    coords_zyx: np.ndarray,
    normals_zyx: np.ndarray,
    membrane_labels_np: np.ndarray,
    priority: np.ndarray,
) -> np.ndarray:
    """
    在每个 label 内做符号传播（CPU BFS）：从置信度最高种子沿 26 邻居，
    若相邻法向点积 < 0 则翻转，保证 label 内符号一致。

    翻转必须被 visited 保护: 若允许翻转已访问节点, BFS 推进时相邻已定节点
    会被新 cur 反复翻转, 导致整片符号震荡 (看起来"随机化")。
    """
    n_pts = len(coords_zyx)
    if n_pts == 0:
        return normals_zyx
    out = normals_zyx.astype(np.float32, copy=True)
    idx_map: Dict[Tuple[int, int, int], int] = {tuple(c): i for i, c in enumerate(coords_zyx)}
    neigh26 = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if not (dz == dy == dx == 0)]
    visited = np.zeros(n_pts, dtype=bool)
    order = np.argsort(-priority.astype(np.float32))
    for seed in order:
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [int(seed)]
        while stack:
            cur = stack.pop()
            z, y, x = coords_zyx[cur]
            cur_lab = int(membrane_labels_np[z, y, x])
            for dz, dy, dx in neigh26:
                key = (int(z + dz), int(y + dy), int(x + dx))
                if (
                    key[0] < 0 or key[1] < 0 or key[2] < 0
                    or key[0] >= membrane_labels_np.shape[0]
                    or key[1] >= membrane_labels_np.shape[1]
                    or key[2] >= membrane_labels_np.shape[2]
                ):
                    continue
                nxt = idx_map.get(key)
                if nxt is None:
                    continue
                if int(membrane_labels_np[key]) != cur_lab:
                    continue
                if visited[nxt]:
                    continue
                if float(np.dot(out[cur], out[nxt])) < 0.0:
                    out[nxt] *= -1.0
                visited[nxt] = True
                stack.append(int(nxt))
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    return out.astype(np.float32)


def estimate_outward_normals(
    coords_gpu: cp.ndarray,
    n_field_gpu: cp.ndarray,
    labels_gpu: cp.ndarray,
) -> NormalEstimationResult:
    """
    候选点定向: gather 预计算法向 -> BFS 在 CC 内符号传播 -> CC 级 divergence
    符号决定整片朝向。

    divergence 法替代旧 centroid 法: 旧法假设 centroid 在凹面一侧 (闭合囊泡 /
    深弧成立, 但开放浅弧 centroid 偏向凸面 -> 判反)。div(n) = 2H 物理上对所有
    凸率方向一致的膜正确, 不受 CC 几何中心位置影响。

    为何不用 probe: v2 midband 是 1-vox 厚, 在 +/-n 采样 mask 占有率两侧都是 0,
    探针无信号。
    """
    def _empty_result() -> NormalEstimationResult:
        return NormalEstimationResult(
            coords_zyx=np.empty((0, 3), dtype=np.int64),
            normals_zyx=np.empty((0, 3), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int32),
        )

    if int(coords_gpu.size) == 0:
        return _empty_result()

    raw_normals_gpu, valid_gpu = _gather_normals_at_coords(coords_gpu, n_field_gpu)
    keep = valid_gpu & (cp.linalg.norm(raw_normals_gpu, axis=1) > 1e-3)
    if not bool(keep.any().get()):
        return _empty_result()
    coords_kept = coords_gpu[keep]
    normals_kept = raw_normals_gpu[keep]

    # BFS + CC-divergence 翻转 (CPU, 依赖 Python dict)
    coords_np = cp.asnumpy(coords_kept).astype(np.int64)
    normals_np = cp.asnumpy(normals_kept)
    labels_np = cp.asnumpy(labels_gpu).astype(np.int32)
    priority = np.ones(len(coords_np), dtype=np.float32)  # 无 probe 置信度, 均一
    stable = _propagate_component_sign(coords_np, normals_np, labels_np, priority)
    stable = _flip_cc_by_divergence(coords_np, stable, labels_np)
    cand_labels = labels_np[coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]].astype(np.int32)
    return NormalEstimationResult(
        coords_zyx=coords_np,
        normals_zyx=stable,
        confidence=priority,
        labels=cand_labels,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 欧几里得最远点采样（GPU；每步 O(N) 向量运算）
# ─────────────────────────────────────────────────────────────────────────────


def fps_gpu(
    coords_xyz: np.ndarray,
    normals_xyz: np.ndarray,
    spacing_px: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU 欧几里得 FPS：迭代挑选距"已选集"最远的候选，直到所有候选都距已选 < spacing。

    返回 (coords, normals, idx)；idx 为被选中的候选在输入数组中的索引（已排序），
    调用方可用它在同序的辅助数组（如 CC 标签）上做同一子采样。

    - 每次迭代 O(N) 向量运算（cp.sum + cp.argmax），N=10⁶ 时单次 ~1 ms
    - midband 是 1-vox 厚 2D 流形，Euclidean 距离 ≈ 测地距离（误差 < 1%）
    - 断图（多膜片）自动处理：跨片的 Euclidean 距离也真的大，FPS 按顺序依次取到
    """
    N = len(coords_xyz)
    if N == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    if N == 1:
        return (
            coords_xyz.astype(np.float32),
            normals_xyz.astype(np.float32),
            np.asarray([0], dtype=np.int64),
        )

    coords_gpu = cp.asarray(coords_xyz, dtype=cp.float32)
    min_dist_sq = cp.full(N, cp.float32(cp.inf), dtype=cp.float32)

    center = coords_gpu.mean(axis=0, keepdims=True)
    seed = int(cp.argmax(cp.sum((coords_gpu - center) ** 2, axis=1)).item())
    del center

    min_dist_sq = cp.minimum(min_dist_sq, cp.sum((coords_gpu - coords_gpu[seed:seed + 1]) ** 2, axis=1))
    chosen: List[int] = [seed]
    sp_sq = float(spacing_px) ** 2

    while len(chosen) < N:
        next_idx = int(cp.argmax(min_dist_sq).item())
        if float(min_dist_sq[next_idx].item()) < sp_sq:
            break
        min_dist_sq = cp.minimum(
            min_dist_sq,
            cp.sum((coords_gpu - coords_gpu[next_idx:next_idx + 1]) ** 2, axis=1),
        )
        chosen.append(next_idx)

    del coords_gpu, min_dist_sq
    cp.get_default_memory_pool().free_all_blocks()

    idx = np.asarray(sorted(set(chosen)), dtype=np.int64)
    return coords_xyz[idx].astype(np.float32), normals_xyz[idx].astype(np.float32), idx


# ─────────────────────────────────────────────────────────────────────────────
# Debug 可视化
# ─────────────────────────────────────────────────────────────────────────────


def _draw_points_volume(shape_zyx: Tuple[int, int, int], coords_xyz: np.ndarray, value: float = 1.0) -> np.ndarray:
    vol = np.zeros(shape_zyx, dtype=np.float32)
    if len(coords_xyz) == 0:
        return vol
    xyz = np.round(coords_xyz).astype(np.int64)
    x = np.clip(xyz[:, 0], 0, shape_zyx[2] - 1)
    y = np.clip(xyz[:, 1], 0, shape_zyx[1] - 1)
    z = np.clip(xyz[:, 2], 0, shape_zyx[0] - 1)
    vol[z, y, x] = float(value)
    return vol


def draw_normals_ball_stick(
    n_field_gpu: cp.ndarray,
    stride: int = 6,
    ball_radius: int = 1,
    stick_length: int = 6,
    ball_value: float = 2.0,
    stick_value: float = 1.0,
) -> cp.ndarray:
    """
    法向场 → ball+stick 可视化体积：每 stride 体素一个采样点，中心画球、
    沿法向画棍。切片查看即可直观看到法向走向。
    """
    _, Z, Y, X = n_field_gpu.shape
    vol = cp.zeros((Z, Y, X), dtype=cp.float32)

    mag = cp.sqrt(n_field_gpu[0] ** 2 + n_field_gpu[1] ** 2 + n_field_gpu[2] ** 2)
    valid = mag > 0.5
    s = max(int(stride), 1)
    z_on = (cp.arange(Z) % s == 0)[:, None, None]
    y_on = (cp.arange(Y) % s == 0)[None, :, None]
    x_on = (cp.arange(X) % s == 0)[None, None, :]
    sampled = valid & z_on & y_on & x_on
    coords = cp.argwhere(sampled)
    if int(coords.shape[0]) == 0:
        return vol

    iz, iy, ix = coords[:, 0], coords[:, 1], coords[:, 2]
    nz = n_field_gpu[0, iz, iy, ix]
    ny = n_field_gpu[1, iz, iy, ix]
    nx = n_field_gpu[2, iz, iy, ix]

    fz = iz.astype(cp.float32); fy = iy.astype(cp.float32); fx = ix.astype(cp.float32)
    for t in range(1, int(stick_length) + 1):
        ft = cp.float32(t)
        pz = cp.clip(cp.rint(fz + ft * nz), 0, Z - 1).astype(cp.int64)
        py = cp.clip(cp.rint(fy + ft * ny), 0, Y - 1).astype(cp.int64)
        px = cp.clip(cp.rint(fx + ft * nx), 0, X - 1).astype(cp.int64)
        vol[pz, py, px] = cp.float32(stick_value)

    rr = int(max(ball_radius, 0))
    for dz in range(-rr, rr + 1):
        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                if dz * dz + dy * dy + dx * dx > rr * rr:
                    continue
                zz = cp.clip(iz + dz, 0, Z - 1)
                yy = cp.clip(iy + dy, 0, Y - 1)
                xx = cp.clip(ix + dx, 0, X - 1)
                vol[zz, yy, xx] = cp.float32(ball_value)

    cp.get_default_memory_pool().free_all_blocks()
    return vol


# ─────────────────────────────────────────────────────────────────────────────
# 主流程（单个 tomogram，全 GPU 驻留）
# ─────────────────────────────────────────────────────────────────────────────


def _process_single_tomogram(args: argparse.Namespace, tomo_path: Path, out_prefix: Path) -> None:
    t0 = time.time()
    max_step = int(getattr(args, "max_step", 99))
    dbg = out_prefix if args.debug else None
    print(f"loading: {tomo_path}")
    vol_np, mrc_voxel = load_mrc(tomo_path)
    voxel = float(args.voxel_size) if float(args.voxel_size) > 0 else mrc_voxel
    print(f"  shape (z,y,x): {vol_np.shape}, voxel: {voxel:.3f} Å")
    vol_gpu = cp.asarray(vol_np, dtype=cp.float32)
    del vol_np

    bin_factor = max(1, int(args.bin))
    if bin_factor > 1:
        print(f"binning by factor {bin_factor}")
        vol_gpu = cskimage.transform.downscale_local_mean(vol_gpu, (bin_factor, bin_factor, bin_factor))
        voxel *= bin_factor
        print(f"  binned shape: {tuple(vol_gpu.shape)}, effective voxel: {voxel:.3f} Å")

    spacing_px = float(args.spacing) / max(voxel, 1e-6)
    if spacing_px <= 0:
        raise ValueError("spacing must be positive (Å)")

    if max_step < 2:
        return
    if args.sigma_z > 0 or args.sigma_xy > 0:
        print(f"smoothing tomogram: sigma_z={args.sigma_z}, sigma_xy={args.sigma_xy}")
        vol_gpu = smooth_intensity(vol_gpu, args.sigma_z, args.sigma_xy, debug_prefix=dbg, voxel=voxel)

    # ---- 概率图 ridge 直接提取 midband + 法向 ----
    if max_step < 3:
        return
    print(f"ridge midband: hessian_σ={args.mask_hessian_sigma}, normal_σ={args.mask_normal_smooth_sigma}, "
          f"prob_min={args.prob_min}, saliency>{args.sheet_saliency_threshold}")
    mid_mask_gpu, mid_labels_gpu, n_field_gpu = extract_midband_ridge(
        vol_gpu,
        hessian_sigma=float(args.mask_hessian_sigma),
        normal_smooth_sigma=float(args.mask_normal_smooth_sigma),
        prob_min=float(args.prob_min),
        sheet_saliency_threshold=float(args.sheet_saliency_threshold),
        border_width=int(args.border_width),
        min_component_size=int(args.min_component_size),
        debug_prefix=dbg, voxel=voxel,
    )
    if not bool(mid_mask_gpu.any().get()):
        print("  midband empty; exiting")
        return

    # ---- 切桥（midband-only）----
    if max_step < 4:
        return
    print(f"midband junction cut: σ={args.cut_junction_sigma}, "
          f"J>{args.cut_junction_threshold}, min_component={args.min_component_size}, "
          f"hole_fill_iters={args.midband_close_iters}")
    mid_mask_gpu, mid_labels_gpu = midband_junction_cut(
        mid_mask_gpu, mid_mask_gpu, n_field_gpu,
        junction_sigma=float(args.cut_junction_sigma),
        junction_threshold=float(args.cut_junction_threshold),
        min_component_size=int(args.min_component_size),
        hole_fill_iters=int(args.midband_close_iters),
        debug_prefix=dbg, voxel=voxel,
    )

    # ---- midband 几何平滑 (删毛刺) ----
    mid_mask_gpu, mid_labels_gpu = smooth_midband(mid_mask_gpu, debug_prefix=dbg, voxel=voxel)
    n_mid_cc = int(mid_labels_gpu.max().get())
    mid_coords_gpu = cp.argwhere(mid_mask_gpu)
    print(f"  {n_mid_cc} midband CCs, {int(mid_coords_gpu.shape[0])} candidates")

    # ---- 定向 + FPS → STAR ----
    if max_step < 5:
        return
    sample_candidates = estimate_outward_normals(mid_coords_gpu, n_field_gpu, mid_labels_gpu)
    vol_shape = tuple(vol_gpu.shape)

    # debug _07f: 把定向后的法向 scatter 回 n_field 做 ball-stick 可视化 (验证定向)
    if dbg is not None and len(sample_candidates.coords_zyx) > 0:
        n_field_oriented = cp.zeros_like(n_field_gpu)
        coo_o = cp.asarray(sample_candidates.coords_zyx)
        nrm_o = cp.asarray(sample_candidates.normals_zyx)
        n_field_oriented[0, coo_o[:, 0], coo_o[:, 1], coo_o[:, 2]] = nrm_o[:, 0]
        n_field_oriented[1, coo_o[:, 0], coo_o[:, 1], coo_o[:, 2]] = nrm_o[:, 1]
        n_field_oriented[2, coo_o[:, 0], coo_o[:, 1], coo_o[:, 2]] = nrm_o[:, 2]
        bs_o = draw_normals_ball_stick(n_field_oriented)
        save_mrc_gpu(dbg.parent / (dbg.name + "_07f_normals_oriented_bs.mrc"), bs_o, voxel)
        del n_field_oriented, coo_o, nrm_o, bs_o

    del mid_coords_gpu, n_field_gpu, mid_mask_gpu, mid_labels_gpu, vol_gpu
    cp.get_default_memory_pool().free_all_blocks()
    if len(sample_candidates.coords_zyx) == 0:
        print("  no valid sample candidates; exiting")
        return

    if max_step < 6:
        return
    coords_xyz = sample_candidates.coords_zyx[:, [2, 1, 0]].astype(np.float32)
    normals_xyz = sample_candidates.normals_zyx[:, [2, 1, 0]].astype(np.float32)
    candidate_labels = sample_candidates.labels

    sampled_coords, sampled_normals, fps_idx = fps_gpu(coords_xyz, normals_xyz, spacing_px)
    if len(sampled_coords) == 0:
        print("  no particles sampled; exiting")
        return
    sampled_labels = candidate_labels[fps_idx] if len(candidate_labels) else np.zeros(len(sampled_coords), dtype=np.int32)

    if float(args.shift_along_normal) != 0.0:
        sampled_coords = sampled_coords + sampled_normals * float(args.shift_along_normal)

    print(f"  sampled particles: {len(sampled_coords)} (spacing {args.spacing:.1f} Å)")
    if dbg is not None:
        save_mrc(dbg.parent / (dbg.name + "_08_sample_points.mrc"), _draw_points_volume(vol_shape, sampled_coords, value=1.0), voxel)

    all_angles = np.zeros((len(sampled_normals), 3), dtype=np.float32)
    for i, n in enumerate(sampled_normals):
        all_angles[i] = vector_to_euler_zyz(n)

    star_path = out_prefix.parent / (out_prefix.name + "_particles.star")
    write_star(star_path, sampled_coords, all_angles, shift_z=float(args.shift_z), cc_ids=sampled_labels)
    print(f"saved {len(sampled_coords)} particles -> {star_path}  ({time.time() - t0:.1f}s)")


def main() -> None:
    p = argparse.ArgumentParser(description="mem2star: AIS 膜预测 → RELION 颗粒（ridge-based）")

    # I/O
    p.add_argument("--tomo", required=True, help="输入 AIS 膜预测 .mrc；--recursive 时支持 glob/dir/regex")
    p.add_argument("--recursive", action="store_true", help="批量模式")
    p.add_argument("--voxel-size", type=float, default=-1.0, help="体素 Å；<=0 则读 MRC 头")
    p.add_argument("--bin", type=int, default=2, help="整数 binning 因子")
    p.add_argument("--out-prefix", default=None, help="输出前缀（默认 = tomogram 文件名 stem）")
    p.add_argument("--debug", action="store_true", help="保存各阶段中间 MRC")

    # 输入平滑
    p.add_argument("--sigma-z", type=float, default=1.5, help="Z 方向高斯 σ（voxel）；消 Z 向伪条纹")
    p.add_argument("--sigma-xy", type=float, default=0.0, help="XY 方向高斯 σ（voxel）；一般留 0")
    p.add_argument("--border-width", type=int, default=4, help="边界剥离宽度（voxel）")

    # Hessian 法向场
    p.add_argument("--mask-hessian-sigma", type=float, default=3.0, help="Hessian σ（voxel）；大 = 法向更光滑")
    p.add_argument("--mask-normal-smooth-sigma", type=float, default=1.5, help="n⊗n 张量平滑 σ（voxel）；大 = 更平滑")

    # Ridge midband（v2 核心）
    p.add_argument("--prob-min", type=float, default=32.0, help="ROI 宽松概率阈值（Hessian 只在此以上算）；低 = 覆盖更多弱信号区")
    p.add_argument("--sheet-saliency-threshold", type=float, default=0.5, help="sheet 判据 S 阈值；高 = 仅保留更锐的片状结构")
    p.add_argument("--min-component-size", type=int, default=1000, help="小于此体素数的 midband CC 直接丢弃；也作切桥判据（切出 < 该值的碎片不切）")

    # 切桥
    p.add_argument("--cut-junction-sigma", type=float, default=3.0, help="n⊗n 张量平滑 σ（voxel）；≈ 桥宽度")
    p.add_argument("--cut-junction-threshold", type=float, default=0.07, help="切桥阈值 [0–1]；低 = 切更多")
    p.add_argument("--midband-close-iters", type=int, default=10, help="切桥后 per-CC binary_closing 迭代次数；弥合 ~2r vox 宽的孔洞（不跨 CC 桥接）。0 = 不填")

    # 颗粒输出
    p.add_argument("--spacing", type=float, default=200.0, help="颗粒间距（Å）")
    p.add_argument("--shift-z", type=float, default=0.0, help="全局 Z 平移（voxel，写入 STAR）")
    p.add_argument("--shift-along-normal", type=float, default=0.0, help="沿各颗粒法向位移（voxel）")
    p.add_argument("--max-step", type=int, default=99, help="stop after this pipeline step (1..6; default 99 = run all)")

    args = p.parse_args()

    tomo_paths = _resolve_paths(args.tomo, recursive=bool(args.recursive))
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
        finally:
            cp.get_default_memory_pool().free_all_blocks()

    print("=" * 88)
    print(f"done: {n_ok} ok, {n_fail} failed, total {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
