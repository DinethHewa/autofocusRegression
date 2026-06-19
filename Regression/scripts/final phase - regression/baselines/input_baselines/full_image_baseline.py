#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent.parent / 'evaluation'
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from baseline_utils import (
    BaselineMethod,
    BaselinePredictionBundle,
    baseline_method_root,
    ensure_baseline_tree,
    load_phase5_index_full,
    run_fixed_policy_baseline,
    summarize_fov_frame,
)
from evaluation_utils import save_json


class FullFieldTiledProxyBaseline(BaselineMethod):
    baseline_name = 'full_field_tiled_proxy'
    baseline_family = 'learned_input'
    input_scope = 'full_field_tiled_proxy'
    evaluation_scope = 'inference_only_proxy'
    training_required = False
    available = True

    def __init__(self, track: str, seed: int = 42, resume: bool = True):
        self.track = str(track)
        self.seed = int(seed)
        self.resume = bool(resume)
        self.root = baseline_method_root(self.track, self.baseline_name)
        self.paths = ensure_baseline_tree(self.root)

    def fit(self, *args, **kwargs):
        return None

    def predict(self, *args, **kwargs) -> pd.DataFrame:
        bundle = self.predict_fov(*args, **kwargs)
        return bundle.fov_df

    def runtime_profile(self, *args, **kwargs) -> dict:
        eval_path = self.paths['metrics'] / 'eval.json'
        if eval_path.is_file():
            with eval_path.open('r', encoding='utf-8') as f:
                payload = json.load(f)
            return {'runtime_ms_per_fov': payload.get('runtime_ms_per_fov', float('nan'))}
        return {}

    def describe(self) -> dict:
        return {
            'baseline_name': self.baseline_name,
            'baseline_family': self.baseline_family,
            'input_scope': self.input_scope,
            'evaluation_scope': self.evaluation_scope,
            'training_required': int(self.training_required),
            'available': int(self.available),
            'note': 'Uses the fixed proposed downstream models on all cached ROI tiles as a transparent full-field tiled proxy, not a true end-to-end full-image network.',
        }

    def predict_fov(self, *args, **kwargs) -> BaselinePredictionBundle:
        fov_out = self.paths['inference'] / 'fov_predictions.csv'
        roi_out = self.paths['inference'] / 'roi_predictions.csv'
        eval_out = self.paths['metrics'] / 'eval.json'
        if self.resume and fov_out.is_file() and eval_out.is_file():
            fov_df = pd.read_csv(fov_out, low_memory=False)
            roi_df = pd.read_csv(roi_out, low_memory=False) if roi_out.is_file() else None
            return BaselinePredictionBundle(
                baseline_name=self.baseline_name,
                family=self.baseline_family,
                evaluation_scope=self.evaluation_scope,
                input_scope=self.input_scope,
                training_required=0,
                available=1,
                fov_df=fov_df,
                roi_df=roi_df,
                metrics={},
                runtime=self.runtime_profile(),
                notes=self.describe()['note'],
            )

        index_df = load_phase5_index_full(self.track)
        bundle = run_fixed_policy_baseline(
            track=self.track,
            index_df=index_df,
            method_name=self.baseline_name,
            family=self.baseline_family,
            evaluation_scope=self.evaluation_scope,
            input_scope=self.input_scope,
            policy_name='all_rois',
            k=None,
            seed=self.seed,
            note=self.describe()['note'],
        )
        bundle.fov_df.to_csv(fov_out, index=False)
        if bundle.roi_df is not None and not bundle.roi_df.empty:
            bundle.roi_df.to_csv(roi_out, index=False)
        metrics = summarize_fov_frame(
            bundle.fov_df,
            {
                'method': self.baseline_name,
                'family': self.baseline_family,
                'evaluation_scope': self.evaluation_scope,
                'available': 1,
                'training_required': 0,
                'input_scope': self.input_scope,
                'model_size_mb': float(bundle.fov_df['model_size_mb'].dropna().iloc[0]) if 'model_size_mb' in bundle.fov_df.columns and bundle.fov_df['model_size_mb'].notna().any() else float('nan'),
                'notes': self.describe()['note'],
            },
        )
        save_json(metrics, eval_out)
        return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description='Run full-field tiled proxy baseline')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    baseline = FullFieldTiledProxyBaseline(track=args.track, seed=args.seed, resume=args.resume)
    bundle = baseline.predict_fov()
    print(f'[DONE] Full-field proxy baseline FOV predictions: {baseline.paths["inference"] / "fov_predictions.csv"}')
    print(f'[DONE] Full-field proxy baseline rows: {len(bundle.fov_df)}')


if __name__ == '__main__':
    main()
