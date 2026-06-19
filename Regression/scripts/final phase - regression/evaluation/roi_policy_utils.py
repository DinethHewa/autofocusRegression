#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS_ROOT = HERE.parent.parent
ROI_DIR = SCRIPTS_ROOT / 'phase 2 - roi selection'
PHASE3_DIR = HERE.parent / 'phase3_preprocessing'

for p in [ROI_DIR, PHASE3_DIR]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from roi_cnn_occupancy import (  # type: ignore
    CellnessModelInterface,
    DummyThresholdCellness,
    compute_tile_occupancy_from_map,
    normalize_image,
)
from roi_selection_cnn_v1 import ROISelectorCNNConfig, _load_calibration, _load_predictor  # type: ignore
from roi_selection_v2 import (  # type: ignore
    D_MIN_NON_DENSE,
    composite_focus_measure,
    load_image_rgb,
    rin_for_image_rgb_calib,
    select_topk_tiles_diverse,
    to_gray,
)
from tiling import cut_to_grid  # type: ignore


SUPPORTED_POLICIES = {
    'center_top1',
    'random_k',
    'focus_only_topk',
    'occupancy_only_topk',
    'hybrid_proposed',
    'all_rois',
    'oracle_best_single_roi',
    'legacy_adaptive',
    'cnn_adaptive',
}

K_POLICIES = {'random_k', 'focus_only_topk', 'occupancy_only_topk'}


@dataclass
class PolicySelectionResult:
    policy_name: str
    policy_label: str
    selected_df: pd.DataFrame
    candidate_df: pd.DataFrame
    selection_time_ms: float
    available: bool = True
    warning: str | None = None
    backend: str | None = None


@dataclass
class ROIPolicyContext:
    seed: int = 42
    legacy_calibration_csv: str | None = None
    cnn_calibration_csv: str | None = None
    cellness_predictor: str | None = None
    predictor_mode: str = 'auto'
    dummy_threshold: float = 0.5
    cellness_model_path: str | None = None
    model_batch_size: int = 64
    model_input_size: int | None = None
    tau_empty: float = 0.2
    beta: float = 0.4
    k_min: int = 1
    k_max: int = 100
    use_sum_occ: bool = True
    w_occ: float = 0.85
    w_focus: float = 0.15
    focus_on_candidates_only: bool = True
    candidate_multiplier: float = 2.0
    d_min_nondense: int = D_MIN_NON_DENSE
    dense_cap_ratio: float = 0.55
    dense_min_ratio: float | None = None
    downsample_for_cnn: int | None = None
    force_dummy_cellness: bool = False
    warnings: list[str] = field(default_factory=list)
    _legacy_calibration: dict[str, float] | None = field(default=None, init=False)
    _cnn_calibration: dict[str, float] | None = field(default=None, init=False)
    _predictor: CellnessModelInterface | None = field(default=None, init=False)
    _predictor_source: str = field(default='unknown', init=False)
    _focus_cache: dict[str, dict[str, float]] = field(default_factory=dict, init=False)
    _occupancy_cache: dict[str, dict[str, float]] = field(default_factory=dict, init=False)
    _hybrid_cache: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def legacy_calibration(self) -> dict[str, float] | None:
        if self._legacy_calibration is not None:
            return self._legacy_calibration
        if not self.legacy_calibration_csv:
            self._legacy_calibration = None
            return None
        p = Path(self.legacy_calibration_csv)
        if not p.is_file():
            self._legacy_calibration = None
            self.warnings.append(f'legacy calibration file not found: {p}')
            return None
        df = pd.read_csv(p)
        if df.empty:
            self._legacy_calibration = None
            self.warnings.append(f'legacy calibration CSV is empty: {p}')
            return None
        row = df.iloc[0].to_dict()
        calib: dict[str, float] = {}
        for k, v in row.items():
            try:
                calib[str(k)] = float(v)
            except Exception:
                continue
        self._legacy_calibration = calib
        return calib

    def cnn_calibration(self) -> dict[str, float] | None:
        if self._cnn_calibration is not None:
            return self._cnn_calibration
        if not self.cnn_calibration_csv:
            self._cnn_calibration = None
            return None
        p = Path(self.cnn_calibration_csv)
        if not p.is_file():
            self._cnn_calibration = None
            self.warnings.append(f'cnn calibration file not found: {p}')
            return None
        self._cnn_calibration = _load_calibration(str(p))
        return self._cnn_calibration

    def predictor(self) -> tuple[CellnessModelInterface, str]:
        if self._predictor is not None:
            return self._predictor, self._predictor_source
        if self.force_dummy_cellness:
            self._predictor = DummyThresholdCellness(threshold=float(self.dummy_threshold))
            self._predictor_source = 'dummy_threshold'
            return self._predictor, self._predictor_source
        if self.cellness_predictor:
            self._predictor = _load_predictor(self.cellness_predictor, self.predictor_mode, self.dummy_threshold)
            self._predictor_source = 'callable'
            return self._predictor, self._predictor_source
        if self.cellness_model_path:
            self._predictor = _load_keras_tile_predictor(
                model_path=Path(self.cellness_model_path),
                batch_size=int(self.model_batch_size),
                input_size=self.model_input_size,
            )
            self._predictor_source = 'keras_model'
            return self._predictor, self._predictor_source
        self._predictor = DummyThresholdCellness(threshold=float(self.dummy_threshold))
        self._predictor_source = 'dummy_threshold'
        self.warnings.append(
            'No cellness predictor/model provided; occupancy/hybrid policies use DummyThresholdCellness fallback.'
        )
        return self._predictor, self._predictor_source


class _KerasTilePredictor:
    def __init__(self, model_path: Path, batch_size: int = 64, input_size: int | None = None):
        try:
            import tensorflow as tf  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError('TensorFlow is required to load a Keras cellness model.') from exc
        if not model_path.is_file():
            raise FileNotFoundError(f'Cellness model not found: {model_path}')
        self.tf = tf
        self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        shape = self.model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        h = int(shape[1] or input_size or 224)
        w = int(shape[2] or input_size or h)
        c = int(shape[3] or 3)
        self.target_hw = (h, w)
        self.channels = c
        self.batch_size = int(batch_size)

    def __call__(self, tiles: np.ndarray) -> np.ndarray:
        arr = np.asarray(tiles, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, :, None]
        if self.channels == 1 and arr.shape[-1] == 3:
            arr = np.mean(arr, axis=-1, keepdims=True)
        elif self.channels == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[1] != self.target_hw[0] or arr.shape[2] != self.target_hw[1]:
            arr = self.tf.image.resize(arr, self.target_hw).numpy()
        preds = self.model.predict(arr, batch_size=self.batch_size, verbose=0)
        return np.asarray(preds, dtype=np.float32).reshape(-1)


def _load_keras_tile_predictor(model_path: Path, batch_size: int = 64, input_size: int | None = None) -> CellnessModelInterface:
    pred = _KerasTilePredictor(model_path=model_path, batch_size=batch_size, input_size=input_size)
    return CellnessModelInterface.from_callable(pred, mode='tiles')


def policy_label(policy_name: str, k: int | None = None) -> str:
    if policy_name in K_POLICIES and k is not None:
        return f'{policy_name}@k={int(k)}'
    return str(policy_name)


def sanitize_policy_label(label: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in {'_', '-', '='} else '_' for ch in str(label))


def expand_policy_jobs(roi_policies: Sequence[str], k_values: Sequence[int]) -> list[tuple[str, int | None]]:
    jobs: list[tuple[str, int | None]] = []
    seen: set[tuple[str, int | None]] = set()
    for policy in roi_policies:
        if policy not in SUPPORTED_POLICIES:
            raise ValueError(f'Unsupported ROI policy: {policy}. Expected one of {sorted(SUPPORTED_POLICIES)}')
        if policy in K_POLICIES:
            for k in k_values:
                key = (policy, int(k))
                if key not in seen:
                    jobs.append(key)
                    seen.add(key)
        else:
            key = (policy, None)
            if key not in seen:
                jobs.append(key)
                seen.add(key)
    return jobs


def parse_patch_rc(patch_id: str) -> tuple[int, int] | None:
    text = str(patch_id)
    try:
        if not text.startswith('r') or '_c' not in text:
            return None
        r_txt, c_txt = text.split('_c', 1)
        return int(r_txt[1:]), int(c_txt)
    except Exception:
        return None


def patch_id_from_index(idx: int, grid_w: int) -> str:
    r = int(idx) // int(grid_w)
    c = int(idx) % int(grid_w)
    return f'r{r:02d}_c{c:02d}'


def patch_index_from_id(patch_id: str, grid_w: int) -> int | None:
    rc = parse_patch_rc(patch_id)
    if rc is None:
        return None
    r, c = rc
    return int(r) * int(grid_w) + int(c)


def stable_rng(seed: int, *parts: Any) -> np.random.Generator:
    payload = '|'.join([str(seed)] + [str(p) for p in parts]).encode('utf-8')
    h = hashlib.sha1(payload).hexdigest()[:16]
    return np.random.default_rng(int(h, 16) % (2 ** 32))


def _resolve_source_image_path(fov_df: pd.DataFrame) -> str | None:
    for col in ['source_image_path', 'image_path', 'fov_id']:
        if col in fov_df.columns:
            vals = fov_df[col].dropna().astype(str)
            if not vals.empty:
                return vals.iloc[0]
    return None


def _grid_shape_from_fov(fov_df: pd.DataFrame) -> tuple[int, int]:
    if 'patch_id' not in fov_df.columns:
        return (1, 1)
    coords = [parse_patch_rc(v) for v in fov_df['patch_id'].astype(str).tolist()]
    coords = [xy for xy in coords if xy is not None]
    if not coords:
        return (1, 1)
    max_r = max(r for r, _ in coords)
    max_c = max(c for _, c in coords)
    return (max_r + 1, max_c + 1)


def _fixed_grid_boxes(image_shape: Sequence[int], grid_h: int, grid_w: int) -> dict[str, tuple[int, int, int, int]]:
    h, w = int(image_shape[0]), int(image_shape[1])
    row_edges = np.linspace(0, h, grid_h + 1, dtype=np.int32)
    col_edges = np.linspace(0, w, grid_w + 1, dtype=np.int32)
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for r in range(grid_h):
        y0 = int(row_edges[r])
        y1 = int(row_edges[r + 1])
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        for c in range(grid_w):
            x0 = int(col_edges[c])
            x1 = int(col_edges[c + 1])
            if x1 <= x0:
                x1 = min(w, x0 + 1)
            boxes[f'r{r:02d}_c{c:02d}'] = (x0, y0, x1, y1)
    return boxes


def _extract_grid_patches(image: np.ndarray, grid_h: int, grid_w: int, roi_size: int = 200) -> dict[str, np.ndarray]:
    patches = cut_to_grid(image, grid=(int(grid_h), int(grid_w)), roi_size=int(roi_size))
    return {str(pid): np.asarray(patch, dtype=np.float32) for pid, patch in patches}


def _robust_focus_norm(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(vals)
    if not np.any(valid):
        return np.zeros_like(vals)
    v = vals[valid]
    med = float(np.median(v))
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
    denom = iqr if iqr > 1e-6 else 1.0
    # Clip the normalized score before applying the logistic map so large
    # outliers do not spam overflow warnings during long ROI-ablation runs.
    z = np.clip((vals - med) / denom, -60.0, 60.0)
    out = 1.0 / (1.0 + np.exp(-z))
    out[~valid] = 0.0
    return out.astype(np.float32)


def _candidate_count(k: int, valid_count: int, multiplier: float | int) -> int:
    if valid_count <= 0:
        return 0
    if isinstance(multiplier, float):
        if multiplier <= 1.0:
            return max(1, int(math.ceil(valid_count * multiplier)))
        return max(1, int(math.ceil(max(1, k) * multiplier)))
    return max(1, int(multiplier))


def _compute_focus_scores(source_path: str, grid_h: int, grid_w: int, ctx: ROIPolicyContext) -> dict[str, float]:
    key = f'{source_path}|{grid_h}x{grid_w}'
    if key in ctx._focus_cache:
        return ctx._focus_cache[key]
    img = load_image_rgb(source_path)
    patches = _extract_grid_patches(img, grid_h=grid_h, grid_w=grid_w, roi_size=200)
    scores: dict[str, float] = {}
    for patch_id, patch in patches.items():
        gray = to_gray(np.asarray(patch, dtype=np.float32))
        scores[patch_id] = float(composite_focus_measure(gray))
    ctx._focus_cache[key] = scores
    return scores


def _compute_occupancy_scores(source_path: str, grid_h: int, grid_w: int, ctx: ROIPolicyContext) -> tuple[dict[str, float], str]:
    source_key = f'{source_path}|{grid_h}x{grid_w}|{ctx.predictor_mode}|{ctx.dummy_threshold}|{ctx.cellness_model_path}|{ctx.cellness_predictor}'
    if source_key in ctx._occupancy_cache:
        return ctx._occupancy_cache[source_key], ctx._predictor_source or 'cached'
    predictor, pred_source = ctx.predictor()
    img = load_image_rgb(source_path)
    boxes = _fixed_grid_boxes(img.shape, grid_h=grid_h, grid_w=grid_w)
    if predictor.has_map:
        p_map = predictor.predict_cellness_map(normalize_image(img))
        occ = compute_tile_occupancy_from_map(p_map, boxes.values())
    else:
        patches = _extract_grid_patches(img, grid_h=grid_h, grid_w=grid_w, roi_size=200)
        arr = np.stack([patches[pid] for pid in sorted(patches.keys())], axis=0).astype(np.float32)
        occ = predictor.predict_tile_occupancy(arr)
    out = {pid: float(val) for pid, val in zip(sorted(boxes.keys()), np.asarray(occ, dtype=np.float32).reshape(-1))}
    ctx._occupancy_cache[source_key] = out
    return out, pred_source


def _compute_hybrid_selection(source_path: str, fov_df: pd.DataFrame, ctx: ROIPolicyContext, policy_name: str) -> tuple[list[str], pd.DataFrame, str]:
    grid_h, grid_w = _grid_shape_from_fov(fov_df)
    if grid_h <= 0 or grid_w <= 0:
        return [], pd.DataFrame(), 'invalid_grid'
    occ_dict, pred_source = _compute_occupancy_scores(source_path, grid_h, grid_w, ctx)
    focus_dict = _compute_focus_scores(source_path, grid_h, grid_w, ctx)

    row_order = sorted(fov_df['patch_id'].astype(str).tolist()) if 'patch_id' in fov_df.columns else []
    if not row_order:
        row_order = [patch_id_from_index(i, grid_w) for i in range(grid_h * grid_w)]

    occ = np.asarray([occ_dict.get(pid, np.nan) for pid in row_order], dtype=np.float32)
    focus = np.asarray([focus_dict.get(pid, np.nan) for pid in row_order], dtype=np.float32)
    focus_norm = _robust_focus_norm(focus)

    calib = ctx.cnn_calibration() or {}
    tau_empty = float(calib.get('tau_empty', ctx.tau_empty))
    beta = float(calib.get('beta', ctx.beta))
    cfg = ROISelectorCNNConfig(
        tau_empty=tau_empty,
        beta=beta,
        k_min=int(calib.get('k_min', ctx.k_min)),
        k_max=int(calib.get('k_max', ctx.k_max)),
        use_sum_occ=bool(calib.get('use_sum_occ', ctx.use_sum_occ)),
        w_occ=float(calib.get('w_occ', ctx.w_occ)),
        w_focus=float(calib.get('w_focus', ctx.w_focus)),
        focus_on_candidates_only=bool(calib.get('focus_on_candidates_only', ctx.focus_on_candidates_only)),
        candidate_multiplier=float(calib.get('candidate_multiplier', ctx.candidate_multiplier)),
        d_min_nondense=int(calib.get('d_min_nondense', ctx.d_min_nondense)),
        dense_cap_ratio=float(calib.get('dense_cap_ratio', ctx.dense_cap_ratio)),
        dense_min_ratio=calib.get('dense_min_ratio', ctx.dense_min_ratio),
        downsample_for_cnn=calib.get('downsample_for_cnn', ctx.downsample_for_cnn),
        batch_size=int(calib.get('batch_size', ctx.model_batch_size)),
    )

    valid = occ >= float(cfg.tau_empty)
    n_valid = int(np.sum(valid))
    if cfg.use_sum_occ:
        k_est = int(round(cfg.beta * float(np.nansum(occ[valid])))) if n_valid else 0
    else:
        k_est = int(round(cfg.beta * n_valid)) if n_valid else 0
    k = max(int(cfg.k_min), min(int(cfg.k_max), int(k_est), int(n_valid))) if n_valid else 0
    if cfg.dense_cap_ratio and len(row_order) > 0:
        k = min(k, int(math.ceil(float(cfg.dense_cap_ratio) * len(row_order))))
    if cfg.dense_min_ratio and len(row_order) > 0 and n_valid > 0:
        k = max(k, int(math.ceil(float(cfg.dense_min_ratio) * len(row_order))))
        k = min(k, n_valid)

    score = (float(cfg.w_occ) * occ) + (float(cfg.w_focus) * focus_norm)
    score[~valid] = -np.inf

    candidate_ids: list[int] = []
    fallback_focus = False
    if n_valid < int(cfg.k_min) and len(row_order) > 0:
        fallback_focus = True
        order = np.argsort(-np.nan_to_num(focus, nan=-np.inf))
        k = min(max(int(cfg.k_min), 1), len(row_order))
        selected_idx = order[:k].tolist()
    elif n_valid == 0 or k == 0:
        selected_idx = []
    else:
        valid_idx = np.where(valid)[0]
        occ_valid = occ[valid]
        order = valid_idx[np.argsort(-occ_valid)]
        cand_count = min(len(order), _candidate_count(k, len(order), cfg.candidate_multiplier))
        candidate_ids = order[:cand_count].tolist()
        if cfg.focus_on_candidates_only:
            tmp_score = np.full_like(score, -np.inf, dtype=np.float32)
            for idx in candidate_ids:
                tmp_score[int(idx)] = score[int(idx)]
            score_use = tmp_score
        else:
            score_use = score
        selected_idx, _ = select_topk_tiles_diverse(
            score_use,
            grid_h,
            grid_w,
            k=k,
            d_min=int(cfg.d_min_nondense),
        )

    selected_patch_ids = [row_order[int(i)] for i in selected_idx]

    diag = pd.DataFrame(
        {
            'patch_id': row_order,
            'occupancy_score': occ,
            'focus_score': focus,
            'focus_norm': focus_norm,
            'hybrid_score': score,
            'valid_by_tau_empty': valid.astype(int),
            'tau_empty': float(cfg.tau_empty),
            'beta': float(cfg.beta),
            'k_est': int(k_est),
            'k_selected_target': int(k),
            'candidate_count': int(len(candidate_ids)),
            'selection_backend': pred_source,
            'policy_name': str(policy_name),
            'fallback_focus': int(fallback_focus),
        }
    )
    return selected_patch_ids, diag, pred_source


def _order_and_annotate(
    fov_df: pd.DataFrame,
    selected_patch_ids: Sequence[str],
    policy_name: str,
    policy_label_text: str,
    selection_score_map: dict[str, float] | None = None,
    extra_maps: dict[str, dict[str, Any]] | None = None,
    backend: str | None = None,
    selected_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = fov_df.copy().reset_index(drop=True)
    if 'patch_id' not in work.columns:
        work['patch_id'] = [f'r00_c00' for _ in range(len(work))]
    work['roi_id'] = work['patch_id'].astype(str)
    work['roi_policy'] = str(policy_name)
    work['roi_policy_label'] = str(policy_label_text)
    work['selected'] = 0
    work['selection_rank'] = np.nan
    work['selection_score'] = np.nan
    if backend is not None:
        work['selection_backend'] = str(backend)
    if extra_maps:
        for col, mapping in extra_maps.items():
            work[col] = work['patch_id'].astype(str).map(mapping)

    rank_map = {str(pid): idx + 1 for idx, pid in enumerate(selected_patch_ids)}
    selected_set = set(map(str, selected_patch_ids))
    work.loc[work['patch_id'].astype(str).isin(selected_set), 'selected'] = 1
    work.loc[work['patch_id'].astype(str).isin(selected_set), 'selection_rank'] = work['patch_id'].astype(str).map(rank_map)
    if selection_score_map is not None:
        work['selection_score'] = work['patch_id'].astype(str).map(selection_score_map)

    selected_df = (
        work[work['selected'] == 1]
        .sort_values(['selection_rank', 'patch_id'], ascending=[True, True])
        .reset_index(drop=True)
    )
    if selected_only:
        return selected_df, selected_df.copy()
    return selected_df, work


def select_rois_for_policy(
    fov_df: pd.DataFrame,
    policy_name: str,
    ctx: ROIPolicyContext,
    k: int | None = None,
    oracle_error_by_patch: dict[str, float] | None = None,
) -> PolicySelectionResult:
    t0 = time.perf_counter()
    label = policy_label(policy_name, k)
    backend = None

    if fov_df.empty:
        return PolicySelectionResult(policy_name, label, pd.DataFrame(), pd.DataFrame(), 0.0, available=False, warning='empty_fov')

    work = fov_df.copy().reset_index(drop=True)
    if 'patch_id' not in work.columns:
        work['patch_id'] = ['r00_c00'] * len(work)

    n_avail = int(len(work))
    if n_avail == 1 and policy_name != 'all_rois':
        only_pid = str(work.iloc[0]['patch_id'])
        score_map = {only_pid: 1.0}
        selected_df, cand_df = _order_and_annotate(
            work,
            [only_pid],
            policy_name=policy_name,
            policy_label_text=label,
            selection_score_map=score_map,
            backend='single_roi',
        )
        return PolicySelectionResult(policy_name, label, selected_df, cand_df, (time.perf_counter() - t0) * 1000.0, backend='single_roi')

    source_path = _resolve_source_image_path(work)
    if source_path is None and policy_name not in {'random_k', 'center_top1', 'all_rois', 'oracle_best_single_roi'}:
        return PolicySelectionResult(
            policy_name,
            label,
            pd.DataFrame(),
            pd.DataFrame(),
            (time.perf_counter() - t0) * 1000.0,
            available=False,
            warning='missing_source_image_path',
        )

    patch_ids = work['patch_id'].astype(str).tolist()

    if policy_name == 'all_rois':
        selected_df, cand_df = _order_and_annotate(
            work,
            sorted(patch_ids),
            policy_name=policy_name,
            policy_label_text=label,
            selection_score_map={pid: 1.0 for pid in patch_ids},
            backend='all',
        )
        return PolicySelectionResult(policy_name, label, selected_df, cand_df, (time.perf_counter() - t0) * 1000.0, backend='all')

    if policy_name == 'center_top1':
        coords = {pid: parse_patch_rc(pid) for pid in patch_ids}
        valid = {pid: rc for pid, rc in coords.items() if rc is not None}
        if valid:
            max_r = max(rc[0] for rc in valid.values())
            max_c = max(rc[1] for rc in valid.values())
            ctr_r = max_r / 2.0
            ctr_c = max_c / 2.0
            best_pid = min(valid.keys(), key=lambda pid: ((valid[pid][0] - ctr_r) ** 2 + (valid[pid][1] - ctr_c) ** 2, pid))
            score_map = {pid: -((valid[pid][0] - ctr_r) ** 2 + (valid[pid][1] - ctr_c) ** 2) for pid in valid}
        else:
            best_pid = str(work.iloc[0]['patch_id'])
            score_map = {best_pid: 0.0}
        selected_df, cand_df = _order_and_annotate(
            work,
            [best_pid],
            policy_name=policy_name,
            policy_label_text=label,
            selection_score_map=score_map,
            backend='center',
        )
        return PolicySelectionResult(policy_name, label, selected_df, cand_df, (time.perf_counter() - t0) * 1000.0, backend='center')

    if policy_name == 'random_k':
        if k is None:
            raise ValueError('random_k requires k')
        rng = stable_rng(ctx.seed, label, work.iloc[0]['fov_id'])
        k_use = min(int(k), len(work))
        order = rng.permutation(len(work))[:k_use]
        selected_patch_ids = [str(work.iloc[i]['patch_id']) for i in order]
        score_map = {pid: float(k_use - idx) for idx, pid in enumerate(selected_patch_ids)}
        selected_df, cand_df = _order_and_annotate(
            work,
            selected_patch_ids,
            policy_name=policy_name,
            policy_label_text=label,
            selection_score_map=score_map,
            backend='random_seeded',
        )
        return PolicySelectionResult(policy_name, label, selected_df, cand_df, (time.perf_counter() - t0) * 1000.0, backend='random_seeded')

    if policy_name == 'oracle_best_single_roi':
        if oracle_error_by_patch is None:
            return PolicySelectionResult(
                policy_name,
                label,
                pd.DataFrame(),
                pd.DataFrame(),
                (time.perf_counter() - t0) * 1000.0,
                available=False,
                warning='oracle_error_not_provided',
            )
        valid_errors = {pid: oracle_error_by_patch.get(pid, np.nan) for pid in patch_ids}
        finite = {pid: val for pid, val in valid_errors.items() if np.isfinite(val)}
        if not finite:
            return PolicySelectionResult(
                policy_name,
                label,
                pd.DataFrame(),
                pd.DataFrame(),
                (time.perf_counter() - t0) * 1000.0,
                available=False,
                warning='oracle_error_missing',
            )
        best_pid = min(finite.keys(), key=lambda pid: (finite[pid], pid))
        score_map = {pid: -float(val) for pid, val in finite.items()}
        selected_df, cand_df = _order_and_annotate(
            work,
            [best_pid],
            policy_name=policy_name,
            policy_label_text=label,
            selection_score_map=score_map,
            backend='oracle_gt',
        )
        return PolicySelectionResult(policy_name, label, selected_df, cand_df, (time.perf_counter() - t0) * 1000.0, backend='oracle_gt')

    if source_path is None or not Path(source_path).is_file():
        return PolicySelectionResult(
            policy_name,
            label,
            pd.DataFrame(),
            pd.DataFrame(),
            (time.perf_counter() - t0) * 1000.0,
            available=False,
            warning=f'source image missing for policy={policy_name}: {source_path}',
        )

    grid_h, grid_w = _grid_shape_from_fov(work)

    if policy_name == 'focus_only_topk':
        if k is None:
            raise ValueError('focus_only_topk requires k')
        focus_scores = _compute_focus_scores(source_path, grid_h, grid_w, ctx)
        score_series = work['patch_id'].astype(str).map(focus_scores).astype(float)
        order_df = work.assign(_focus=score_series).sort_values(['_focus', 'patch_id'], ascending=[False, True]).head(min(int(k), len(work)))
        selected_patch_ids = order_df['patch_id'].astype(str).tolist()
        selected_df, cand_df = _order_and_annotate(
            work,
            selected_patch_ids,
            policy_name=policy_name,
            policy_label_text=label,
            selection_score_map=focus_scores,
            extra_maps={'focus_score': focus_scores},
            backend='handcrafted_focus',
        )
        return PolicySelectionResult(policy_name, label, selected_df, cand_df, (time.perf_counter() - t0) * 1000.0, backend='handcrafted_focus')

    if policy_name == 'occupancy_only_topk':
        if k is None:
            raise ValueError('occupancy_only_topk requires k')
        occ_scores, pred_source = _compute_occupancy_scores(source_path, grid_h, grid_w, ctx)
        order_df = work.assign(_occ=work['patch_id'].astype(str).map(occ_scores)).sort_values(['_occ', 'patch_id'], ascending=[False, True]).head(min(int(k), len(work)))
        selected_patch_ids = order_df['patch_id'].astype(str).tolist()
        selected_df, cand_df = _order_and_annotate(
            work,
            selected_patch_ids,
            policy_name=policy_name,
            policy_label_text=label,
            selection_score_map=occ_scores,
            extra_maps={'occupancy_score': occ_scores},
            backend=pred_source,
        )
        return PolicySelectionResult(policy_name, label, selected_df, cand_df, (time.perf_counter() - t0) * 1000.0, backend=pred_source)

    if policy_name in {'hybrid_proposed', 'cnn_adaptive'}:
        selected_patch_ids, diag_df, pred_source = _compute_hybrid_selection(source_path, work, ctx, policy_name=policy_name)
        score_map = dict(zip(diag_df['patch_id'].astype(str), diag_df['hybrid_score'].astype(float)))
        extras = {
            'occupancy_score': dict(zip(diag_df['patch_id'].astype(str), diag_df['occupancy_score'].astype(float))),
            'focus_score': dict(zip(diag_df['patch_id'].astype(str), diag_df['focus_score'].astype(float))),
            'focus_norm': dict(zip(diag_df['patch_id'].astype(str), diag_df['focus_norm'].astype(float))),
            'hybrid_score': score_map,
            'valid_by_tau_empty': dict(zip(diag_df['patch_id'].astype(str), diag_df['valid_by_tau_empty'].astype(int))),
        }
        selected_df, cand_df = _order_and_annotate(
            work,
            selected_patch_ids,
            policy_name=policy_name,
            policy_label_text=label,
            selection_score_map=score_map,
            extra_maps=extras,
            backend=pred_source,
        )
        return PolicySelectionResult(policy_name, label, selected_df, cand_df, (time.perf_counter() - t0) * 1000.0, backend=pred_source)

    if policy_name == 'legacy_adaptive':
        calib = ctx.legacy_calibration()
        if not calib:
            return PolicySelectionResult(
                policy_name,
                label,
                pd.DataFrame(),
                pd.DataFrame(),
                (time.perf_counter() - t0) * 1000.0,
                available=False,
                warning='legacy calibration unavailable',
            )
        img = load_image_rgb(source_path)
        res = rin_for_image_rgb_calib(img, calib, beta_override=None)
        gh = int(res.get('grid_h', 0))
        gw = int(res.get('grid_w', 0))
        if gh <= 0 or gw <= 0:
            return PolicySelectionResult(policy_name, label, pd.DataFrame(), pd.DataFrame(), (time.perf_counter() - t0) * 1000.0, available=False, warning='legacy selector returned invalid grid')
        selector_patch_ids = [patch_id_from_index(int(idx), gw) for idx in np.asarray(res.get('top_indices', []), dtype=np.int32).tolist()]
        available_patch_ids = set(work['patch_id'].astype(str).tolist())
        selected_patch_ids = [pid for pid in selector_patch_ids if pid in available_patch_ids]
        scores = np.asarray(res.get('scores', []), dtype=np.float32)
        score_map = {patch_id_from_index(i, gw): float(scores[i]) for i in range(len(scores))}
        extras = {
            'focus_score': {patch_id_from_index(i, gw): float(v) for i, v in enumerate(np.asarray(res.get('focus_norm', []), dtype=np.float32))},
            'hybrid_score': score_map,
        }
        selected_df, cand_df = _order_and_annotate(
            work,
            selected_patch_ids,
            policy_name=policy_name,
            policy_label_text=label,
            selection_score_map=score_map,
            extra_maps=extras,
            backend='legacy_rin',
        )
        return PolicySelectionResult(policy_name, label, selected_df, cand_df, (time.perf_counter() - t0) * 1000.0, backend='legacy_rin')

    raise ValueError(f'Unhandled ROI policy: {policy_name}')


def summarize_context_warnings(ctx: ROIPolicyContext) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for msg in ctx.warnings:
        if msg not in seen:
            out.append(msg)
            seen.add(msg)
    return out
