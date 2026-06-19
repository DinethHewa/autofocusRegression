#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent.parent / 'evaluation'
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from baseline_utils import (
    BaselineMethod,
    BaselinePredictionBundle,
    baseline_method_root,
    build_standard_fov_frame,
    ensure_baseline_tree,
    gt_per_fov,
    load_phase5_index_full,
    model_size_mb,
    near_focus_rows,
    resolve_master_splits,
    summarize_fov_frame,
)
from evaluation_utils import regression_metrics, save_json, weighted_median
from roi_policy_utils import ROIPolicyContext, select_rois_for_policy

PHASE5_DIR = HERE.parent.parent / 'phase5_regression'
if str(PHASE5_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE5_DIR))

from utils import load_XB_tensor  # type: ignore


FEATURE_COLUMNS = [
    'gray_mean', 'gray_std', 'gray_energy', 'lap_var', 'brenner', 'tenengrad',
    'dog1_mean', 'dog1_std', 'dog1_energy',
    'dog2_mean', 'dog2_std', 'dog2_energy',
    'ehf_mean', 'ehf_std', 'ehf_energy',
    'dog_delta_mean',
]
AGG_FEATURE_COLUMNS = [f'{prefix}_{col}' for prefix in ['mean', 'std'] for col in FEATURE_COLUMNS] + ['num_inputs_used']
CANDIDATE_SCOPES = [
    {'scope_name': 'center_crop_proxy', 'policy_name': 'center_top1', 'k': 1, 'scope_note': 'Center-most ROI only.'},
    {'scope_name': 'roi_selected_proxy', 'policy_name': 'hybrid_proposed', 'k': 7, 'scope_note': 'Content-aware selected ROI tiles.'},
    {'scope_name': 'full_field_tiled_proxy', 'policy_name': 'all_rois', 'k': None, 'scope_note': 'All cached ROI tiles as a tiled full-field proxy.'},
]
CANDIDATE_MODELS = ['ridge', 'huber']


@dataclass
class ClassicalAssets:
    feature_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    split_meta: dict[str, Any]


class ClassicalFocusBaseline(BaselineMethod):
    baseline_name = 'classical_focus_best'
    baseline_family = 'classical'
    input_scope = 'classical_focus_selected_input'
    evaluation_scope = 'train_val_select_then_test'
    training_required = True
    available = True

    def __init__(self, track: str, seed: int = 42, resume: bool = True):
        self.track = str(track)
        self.seed = int(seed)
        self.resume = bool(resume)
        self.root = baseline_method_root(self.track, self.baseline_name)
        self.paths = ensure_baseline_tree(self.root)
        self.model_path = self.paths['models'] / 'best_model.pkl'
        self.config_path = self.paths['models'] / 'best_config.json'
        self.feature_cache_path = self.paths['cache'] / 'roi_features.pkl'
        self.feature_meta_path = self.paths['cache'] / 'roi_features_meta.json'

    def describe(self) -> dict:
        return {
            'baseline_name': self.baseline_name,
            'baseline_family': self.baseline_family,
            'input_scope': self.input_scope,
            'evaluation_scope': self.evaluation_scope,
            'training_required': int(self.training_required),
            'available': int(self.available),
            'note': 'Classical focus-measure baseline using handcrafted features from cached ROI tensors and shallow signed-distance regression with internal model/scope selection.',
        }

    def runtime_profile(self, *args, **kwargs) -> dict[str, Any]:
        p = self.paths['metrics'] / 'eval.json'
        if p.is_file():
            with p.open('r', encoding='utf-8') as f:
                payload = json.load(f)
            return {
                'runtime_ms_per_fov': payload.get('runtime_ms_per_fov', float('nan')),
                'feature_ms_per_roi': payload.get('feature_ms_per_roi', float('nan')),
            }
        return {}

    def fit(self, *args, **kwargs) -> BaselinePredictionBundle:
        train_if_missing = bool(kwargs.get('train_if_missing', True))
        force = bool(kwargs.get('force', False))
        if self.resume and not force and self.model_path.is_file() and self.config_path.is_file():
            return self.predict_fov(*args, **kwargs)
        if not train_if_missing:
            raise FileNotFoundError(f'Missing classical focus baseline assets under {self.root}')

        assets = self._load_assets()
        selection_rows = []
        best_val = np.inf
        best_payload: dict[str, Any] | None = None
        best_model = None
        best_val_fov = None
        best_test_fov = None

        for scope in CANDIDATE_SCOPES:
            train_fov = self._aggregate_scope_features(assets.train_df, scope['policy_name'], scope['k'])
            val_fov = self._aggregate_scope_features(assets.val_df, scope['policy_name'], scope['k'])
            test_fov = self._aggregate_scope_features(assets.test_df, scope['policy_name'], scope['k'])
            for model_name in CANDIDATE_MODELS:
                model = self._build_model(model_name)
                x_train = train_fov[AGG_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
                y_train = pd.to_numeric(train_fov['y_true_signed_um'], errors='coerce').to_numpy(dtype=np.float32)
                valid_train = np.isfinite(y_train) & np.all(np.isfinite(x_train), axis=1)
                if int(np.sum(valid_train)) < 16:
                    continue
                model.fit(x_train[valid_train], y_train[valid_train])
                val_pred = model.predict(val_fov[AGG_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
                val_eval = self._evaluate_scope_predictions(val_fov, val_pred)
                row = {
                    'scope_name': scope['scope_name'],
                    'policy_name': scope['policy_name'],
                    'k': scope['k'] if scope['k'] is not None else -1,
                    'model_name': model_name,
                    'val_mae_um': val_eval['mae_um'],
                    'val_rmse_um': val_eval['rmse_um'],
                    'val_median_abs_error_um': val_eval['median_abs_error_um'],
                    'n_train_fov': int(len(train_fov)),
                    'n_val_fov': int(len(val_fov)),
                    'mean_inputs_train': float(pd.to_numeric(train_fov['num_inputs_used'], errors='coerce').mean()),
                    'mean_inputs_val': float(pd.to_numeric(val_fov['num_inputs_used'], errors='coerce').mean()),
                    'scope_note': scope['scope_note'],
                }
                selection_rows.append(row)
                if np.isfinite(row['val_mae_um']) and row['val_mae_um'] < best_val:
                    best_val = float(row['val_mae_um'])
                    best_payload = {
                        'scope_name': scope['scope_name'],
                        'policy_name': scope['policy_name'],
                        'k': scope['k'],
                        'model_name': model_name,
                        'scope_note': scope['scope_note'],
                        'split_source': assets.split_meta.get('source', 'unknown'),
                    }
                    best_model = model
                    best_val_fov = val_fov.copy()
                    best_val_fov['y_pred_signed_um'] = val_pred
                    test_pred = model.predict(test_fov[AGG_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
                    best_test_fov = test_fov.copy()
                    best_test_fov['y_pred_signed_um'] = test_pred

        if best_payload is None or best_model is None or best_test_fov is None:
            raise RuntimeError('Failed to fit any classical focus baseline candidate.')

        pd.DataFrame(selection_rows).sort_values(['val_mae_um', 'val_rmse_um']).to_csv(
            self.paths['metrics'] / 'internal_model_selection.csv', index=False
        )
        with self.model_path.open('wb') as f:
            pickle.dump(best_model, f)
        with self.config_path.open('w', encoding='utf-8') as f:
            json.dump(best_payload, f, indent=2)
        if best_val_fov is not None:
            best_val_fov.to_csv(self.paths['inference'] / 'val_predictions.csv', index=False)

        return self._finalize_test_outputs(best_test_fov, best_payload)

    def predict(self, *args, **kwargs) -> pd.DataFrame:
        bundle = self.predict_fov(*args, **kwargs)
        return bundle.fov_df

    def predict_fov(self, *args, **kwargs) -> BaselinePredictionBundle:
        fov_out = self.paths['inference'] / 'fov_predictions.csv'
        eval_out = self.paths['metrics'] / 'eval.json'
        if self.resume and fov_out.is_file() and eval_out.is_file():
            fov_df = pd.read_csv(fov_out, low_memory=False)
            return BaselinePredictionBundle(
                baseline_name=self.baseline_name,
                family=self.baseline_family,
                evaluation_scope=self.evaluation_scope,
                input_scope=self.input_scope,
                training_required=1,
                available=1,
                fov_df=fov_df,
                roi_df=None,
                metrics={},
                runtime=self.runtime_profile(),
                notes=self.describe()['note'],
            )
        if not self.model_path.is_file() or not self.config_path.is_file():
            return self.fit(*args, **kwargs)
        with self.config_path.open('r', encoding='utf-8') as f:
            cfg = json.load(f)
        with self.model_path.open('rb') as f:
            model = pickle.load(f)
        assets = self._load_assets()
        test_fov = self._aggregate_scope_features(assets.test_df, cfg['policy_name'], cfg.get('k'))
        preds = model.predict(test_fov[AGG_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        test_fov['y_pred_signed_um'] = preds
        return self._finalize_test_outputs(test_fov, cfg)

    def _load_assets(self) -> ClassicalAssets:
        index_df = load_phase5_index_full(self.track)
        train_df, val_df, test_df, split_meta = resolve_master_splits(self.track, index_df, seed=self.seed, resume=self.resume)
        feature_df = self._build_or_load_feature_cache(index_df)
        train_df = feature_df[feature_df['roi_uid'].isin(train_df['roi_uid'].astype(str))].copy().reset_index(drop=True)
        val_df = feature_df[feature_df['roi_uid'].isin(val_df['roi_uid'].astype(str))].copy().reset_index(drop=True)
        test_df = feature_df[feature_df['roi_uid'].isin(test_df['roi_uid'].astype(str))].copy().reset_index(drop=True)
        return ClassicalAssets(feature_df=feature_df, train_df=train_df, val_df=val_df, test_df=test_df, split_meta=split_meta)

    def _build_or_load_feature_cache(self, index_df: pd.DataFrame) -> pd.DataFrame:
        if self.resume and self.feature_cache_path.is_file():
            df = pd.read_pickle(self.feature_cache_path)
            if set(FEATURE_COLUMNS).issubset(df.columns):
                return df
        rows = []
        t0 = time.perf_counter()
        for i, row in enumerate(index_df.itertuples(index=False), start=1):
            xb = load_XB_tensor(str(row.cache_path_XB))
            feats = self._extract_roi_features(xb)
            rows.append({
                'roi_uid': str(row.roi_uid),
                'dataset': str(row.dataset),
                'group_id': str(row.group_id),
                'fov_id': str(row.fov_id),
                'patch_id': str(getattr(row, 'patch_id', 'r00_c00')),
                'source_image_path': str(getattr(row, 'source_image_path', '')),
                'roi_importance': float(getattr(row, 'roi_importance', 1.0)),
                'defocus_um': float(getattr(row, 'defocus_um', np.nan)),
                **feats,
            })
            if i % 5000 == 0:
                print(f'[INFO] classical feature cache rows: {i}/{len(index_df)}')
        df = pd.DataFrame(rows)
        df.to_pickle(self.feature_cache_path)
        elapsed = time.perf_counter() - t0
        save_json({'n_rois': int(len(df)), 'feature_time_s': float(elapsed), 'feature_ms_per_roi': 1000.0 * float(elapsed) / max(len(df), 1)}, self.feature_meta_path)
        return df

    @staticmethod
    def _extract_roi_features(xb: np.ndarray) -> dict[str, float]:
        x = np.asarray(xb, dtype=np.float32)
        gray = x[..., 0]
        dog1 = x[..., 1]
        dog2 = x[..., 2]
        ehf = x[..., 3]
        lap = -4.0 * gray
        lap[:-1, :] += gray[1:, :]
        lap[1:, :] += gray[:-1, :]
        lap[:, :-1] += gray[:, 1:]
        lap[:, 1:] += gray[:, :-1]
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        br = gray[:, 2:] - gray[:, :-2] if gray.shape[1] > 2 else np.zeros((gray.shape[0], 0), dtype=np.float32)
        feats = {
            'gray_mean': float(np.mean(gray)),
            'gray_std': float(np.std(gray)),
            'gray_energy': float(np.mean(np.square(gray))),
            'lap_var': float(np.var(lap)),
            'brenner': float(np.mean(np.square(br))) if br.size else 0.0,
            'tenengrad': float(np.mean(np.square(gx))) + float(np.mean(np.square(gy))),
            'dog1_mean': float(np.mean(dog1)),
            'dog1_std': float(np.std(dog1)),
            'dog1_energy': float(np.mean(np.square(dog1))),
            'dog2_mean': float(np.mean(dog2)),
            'dog2_std': float(np.std(dog2)),
            'dog2_energy': float(np.mean(np.square(dog2))),
            'ehf_mean': float(np.mean(ehf)),
            'ehf_std': float(np.std(ehf)),
            'ehf_energy': float(np.mean(np.square(ehf))),
            'dog_delta_mean': float(np.mean(dog1 - dog2)),
        }
        return feats

    def _aggregate_scope_features(self, feature_df: pd.DataFrame, policy_name: str, k: int | None) -> pd.DataFrame:
        rows = []
        gt_map = gt_per_fov(feature_df.rename(columns={'defocus_um': 'defocus_um'}))
        gt_map = gt_map.set_index('fov_id')['y_true_signed_um'].to_dict()
        ctx = ROIPolicyContext(seed=self.seed)
        feature_ms_per_roi = float('nan')
        if self.feature_meta_path.is_file():
            with self.feature_meta_path.open('r', encoding='utf-8') as f:
                feature_ms_per_roi = float(json.load(f).get('feature_ms_per_roi', np.nan))
        for fov_id, g in feature_df.groupby('fov_id', sort=True):
            result = select_rois_for_policy(g, policy_name=policy_name, ctx=ctx, k=k)
            selected = result.selected_df.copy() if result.available and not result.selected_df.empty else pd.DataFrame()
            if selected.empty:
                row = {c: np.nan for c in AGG_FEATURE_COLUMNS}
                row.update({
                    'dataset': str(g['dataset'].iloc[0]),
                    'fov_id': str(fov_id),
                    'y_true_signed_um': float(gt_map.get(str(fov_id), np.nan)),
                    'selection_time_ms': float(result.selection_time_ms),
                    'runtime_ms_per_fov': float(result.selection_time_ms),
                    'status': 'no_selected_roi',
                })
                rows.append(row)
                continue
            w = pd.to_numeric(selected.get('roi_importance', 1.0), errors='coerce').fillna(1.0).to_numpy(dtype=float)
            if not np.isfinite(w).any() or float(np.sum(w)) <= 0:
                w = np.ones((len(selected),), dtype=float)
            row = {
                'dataset': str(selected['dataset'].iloc[0]),
                'fov_id': str(fov_id),
                'y_true_signed_um': float(gt_map.get(str(fov_id), np.nan)),
                'num_inputs_used': int(len(selected)),
                'selection_time_ms': float(result.selection_time_ms),
                'runtime_ms_per_fov': float(result.selection_time_ms) + float(feature_ms_per_roi) * float(len(selected)),
                'status': 'ok',
            }
            for col in FEATURE_COLUMNS:
                vals = pd.to_numeric(selected[col], errors='coerce').to_numpy(dtype=float)
                mask = np.isfinite(vals)
                vals = vals[mask]
                ww = w[mask]
                if vals.size == 0:
                    row[f'mean_{col}'] = np.nan
                    row[f'std_{col}'] = np.nan
                else:
                    if float(np.sum(ww)) <= 0:
                        ww = np.ones_like(vals)
                    row[f'mean_{col}'] = float(np.sum(vals * ww) / max(float(np.sum(ww)), 1e-12))
                    row[f'std_{col}'] = float(np.std(vals))
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _build_model(model_name: str):
        if model_name == 'ridge':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('reg', RidgeCV(alphas=np.array([0.1, 1.0, 10.0, 100.0], dtype=float))),
            ])
        if model_name == 'huber':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('reg', HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000)),
            ])
        raise ValueError(f'Unsupported classical model_name={model_name}')

    @staticmethod
    def _evaluate_scope_predictions(fov_df: pd.DataFrame, preds: np.ndarray) -> dict[str, float]:
        gt = pd.to_numeric(fov_df['y_true_signed_um'], errors='coerce').to_numpy(dtype=float)
        pred = np.asarray(preds, dtype=float)
        mask = np.isfinite(gt) & np.isfinite(pred)
        if not np.any(mask):
            return {k: np.nan for k in ['mae_um', 'rmse_um', 'median_abs_error_um', 'p95_abs_error_um']}
        return regression_metrics(gt[mask], pred[mask])

    def _finalize_test_outputs(self, test_fov: pd.DataFrame, cfg: dict[str, Any]) -> BaselinePredictionBundle:
        test_fov = test_fov.copy()
        test_fov['signed_error_um'] = pd.to_numeric(test_fov['y_pred_signed_um'], errors='coerce') - pd.to_numeric(test_fov['y_true_signed_um'], errors='coerce')
        test_fov['abs_error_um'] = pd.to_numeric(test_fov['signed_error_um'], errors='coerce').abs()
        test_fov['uncertain'] = pd.to_numeric(test_fov['y_pred_signed_um'], errors='coerce').isna().astype(int)
        pred_df = test_fov[['fov_id', 'dataset', 'y_true_signed_um', 'y_pred_signed_um', 'abs_error_um', 'signed_error_um', 'num_inputs_used', 'runtime_ms_per_fov', 'status', 'uncertain']].copy()
        fov_std = build_standard_fov_frame(
            pred_df[['fov_id', 'y_pred_signed_um', 'uncertain', 'runtime_ms_per_fov', 'num_inputs_used', 'status']].copy(),
            index_df=test_fov[['fov_id', 'dataset', 'y_true_signed_um']].rename(columns={'y_true_signed_um': 'defocus_um'}),
            baseline_name=self.baseline_name,
            family=self.baseline_family,
            evaluation_scope=self.evaluation_scope,
            input_scope=cfg.get('scope_name', self.input_scope),
            training_required=1,
            available=1,
            notes=f"Best classical focus baseline: {cfg.get('model_name')} on {cfg.get('scope_name')}.",
        )
        # build_standard_fov_frame needs an index-like dataframe with per-fov GT; fill directly afterwards for robustness.
        gt_cols = test_fov[['fov_id', 'dataset', 'y_true_signed_um']].drop_duplicates(subset=['fov_id']).copy()
        fov_std = fov_std.drop(columns=['dataset', 'y_true_signed_um'], errors='ignore').merge(gt_cols, on='fov_id', how='left')
        fov_std['signed_error_um'] = pd.to_numeric(fov_std['y_pred_signed_um'], errors='coerce') - pd.to_numeric(fov_std['y_true_signed_um'], errors='coerce')
        fov_std['abs_error_um'] = pd.to_numeric(fov_std['signed_error_um'], errors='coerce').abs()
        fov_std['runtime_ms_per_fov'] = pd.to_numeric(pred_df['runtime_ms_per_fov'], errors='coerce').to_numpy(dtype=float)
        fov_std['mean_inputs_used_per_fov'] = pd.to_numeric(pred_df['num_inputs_used'], errors='coerce').to_numpy(dtype=float)
        model_mb = model_size_mb(self.model_path)
        fov_std['model_size_mb'] = model_mb
        fov_std['baseline_name'] = self.baseline_name
        fov_std['family'] = self.baseline_family
        fov_std['evaluation_scope'] = self.evaluation_scope
        fov_std['input_scope'] = cfg.get('scope_name', self.input_scope)
        fov_std['training_required'] = 1
        fov_std['available'] = 1
        fov_std['notes'] = f"Best classical focus baseline: {cfg.get('model_name')} on {cfg.get('scope_name')}"

        fov_out = self.paths['inference'] / 'fov_predictions.csv'
        fov_std.to_csv(fov_out, index=False)
        per_bin = near_focus_rows(
            fov_std,
            {
                'method': self.baseline_name,
                'family': self.baseline_family,
                'evaluation_scope': self.evaluation_scope,
                'available': 1,
                'training_required': 1,
                'input_scope': cfg.get('scope_name', self.input_scope),
                'model_size_mb': model_mb,
                'notes': f"Best classical focus baseline: {cfg.get('model_name')} on {cfg.get('scope_name')}",
            },
        )
        per_bin.to_csv(self.paths['metrics'] / 'per_bin.csv', index=False)
        metrics = summarize_fov_frame(
            fov_std,
            {
                'method': self.baseline_name,
                'family': self.baseline_family,
                'evaluation_scope': self.evaluation_scope,
                'available': 1,
                'training_required': 1,
                'input_scope': cfg.get('scope_name', self.input_scope),
                'model_size_mb': model_mb,
                'notes': f"Best classical focus baseline: {cfg.get('model_name')} on {cfg.get('scope_name')}",
            },
        )
        if self.feature_meta_path.is_file():
            with self.feature_meta_path.open('r', encoding='utf-8') as f:
                meta = json.load(f)
            metrics['feature_ms_per_roi'] = float(meta.get('feature_ms_per_roi', np.nan))
        save_json(metrics, self.paths['metrics'] / 'eval.json')
        return BaselinePredictionBundle(
            baseline_name=self.baseline_name,
            family=self.baseline_family,
            evaluation_scope=self.evaluation_scope,
            input_scope=str(cfg.get('scope_name', self.input_scope)),
            training_required=1,
            available=1,
            fov_df=fov_std,
            roi_df=None,
            metrics=metrics,
            runtime=self.runtime_profile(),
            notes=f"Best classical focus baseline: {cfg.get('model_name')} on {cfg.get('scope_name')}",
        )


def main() -> None:
    ap = argparse.ArgumentParser(description='Fit or evaluate the classical focus baseline')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--mode', choices=['fit', 'predict'], default='fit')
    args = ap.parse_args()
    baseline = ClassicalFocusBaseline(track=args.track, seed=args.seed, resume=args.resume)
    if args.mode == 'fit':
        bundle = baseline.fit()
    else:
        bundle = baseline.predict_fov()
    print(f'[DONE] Classical focus baseline FOV predictions: {baseline.paths["inference"] / "fov_predictions.csv"}')
    print(f'[DONE] Classical focus baseline rows: {len(bundle.fov_df)}')


if __name__ == '__main__':
    main()
