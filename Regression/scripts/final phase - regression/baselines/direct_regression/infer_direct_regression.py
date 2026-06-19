#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from utils_direct_regression import (
    aggregate_fov_predictions,
    ensure_direct_tree,
    load_or_create_direct_splits,
    predict_roi,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Run direct single-stage regression inference')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    ap.add_argument('--split-name', default='test', choices=['train', 'val', 'test', 'all'])
    ap.add_argument('--top-k', type=int, default=7)
    ap.add_argument('--resume', action='store_true')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_paths = ensure_direct_tree(args.track)
    model_path = out_paths['models'] / 'best_model.keras'
    roi_out = out_paths['inference'] / 'roi_predictions.csv'
    fov_out = out_paths['inference'] / 'fov_predictions.csv'
    if args.resume and model_path.is_file() and roi_out.is_file() and fov_out.is_file():
        print(f'[INFO] Resume: direct regression inference outputs already exist in {out_paths["inference"]}; skipping.')
        return
    if not model_path.is_file():
        raise FileNotFoundError(f'Missing trained direct regression model: {model_path}')

    train_df, val_df, test_df, _, index_df = load_or_create_direct_splits(args.track, resume=True)
    if args.split_name == 'train':
        use_df = train_df.copy().reset_index(drop=True)
    elif args.split_name == 'val':
        use_df = val_df.copy().reset_index(drop=True)
    elif args.split_name == 'test':
        use_df = test_df.copy().reset_index(drop=True)
    else:
        use_df = index_df.copy().reset_index(drop=True)

    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    roi_df = use_df.copy().reset_index(drop=True)
    roi_df['y_pred_signed_um'] = predict_roi(model, roi_df, batch_size=128)
    roi_df['signed_error_um'] = pd.to_numeric(roi_df['y_pred_signed_um'], errors='coerce') - pd.to_numeric(roi_df['defocus_um'], errors='coerce')
    roi_df['abs_error_um'] = pd.to_numeric(roi_df['signed_error_um'], errors='coerce').abs()
    roi_df.to_csv(roi_out, index=False)

    fov_df = aggregate_fov_predictions(roi_df, top_k=int(args.top_k), default_method='weighted_mean')
    fov_df['signed_error_um'] = pd.to_numeric(fov_df['y_pred_signed_um'], errors='coerce') - pd.to_numeric(fov_df['y_true_signed_um'], errors='coerce')
    fov_df['abs_error_um'] = pd.to_numeric(fov_df['signed_error_um'], errors='coerce').abs()
    fov_df.to_csv(fov_out, index=False)
    print(f'[DONE] Direct regression ROI predictions: {roi_out}')
    print(f'[DONE] Direct regression FOV predictions: {fov_out}')


if __name__ == '__main__':
    main()
