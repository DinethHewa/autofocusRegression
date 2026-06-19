#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TripletSamplingStats:
    anchors_seen: int = 0
    anchors_skipped_zero: int = 0
    anchors_no_positive: int = 0
    anchors_no_negative: int = 0
    triplets_built: int = 0

    def to_dict(self) -> dict:
        return {
            "anchors_seen": int(self.anchors_seen),
            "anchors_skipped_zero": int(self.anchors_skipped_zero),
            "anchors_no_positive": int(self.anchors_no_positive),
            "anchors_no_negative": int(self.anchors_no_negative),
            "triplets_built": int(self.triplets_built),
        }


class TripletSampler:
    """
    Branch-specific triplet sampler using O(1)-style bin/group index lookups.

    Positive:
      - same branch dataframe
      - same bin_id
      - different group_id
    Negative:
      - different bin_id; prefer semi-hard bins (bin_id +/- 1)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        bin_width_um: float = 0.5,
        exclude_zero_from_triplets: bool = True,
        seed: int = 42,
    ):
        required = ["y_mag_um", "group_id"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"TripletSampler missing required columns: {missing}")

        self.df = df.reset_index(drop=True).copy()
        self.bin_width_um = float(bin_width_um)
        self.exclude_zero_from_triplets = bool(exclude_zero_from_triplets)
        self.rng = np.random.default_rng(seed)
        self.stats = TripletSamplingStats()

        self.df["_idx"] = np.arange(len(self.df), dtype=np.int64)
        self.df["_bin_id"] = np.floor(self.df["y_mag_um"].astype(float) / self.bin_width_um).astype(int)
        if "defocus_um" in self.df.columns:
            self.df["_is_zero"] = np.isclose(self.df["defocus_um"].astype(float), 0.0)
        else:
            self.df["_is_zero"] = np.isclose(self.df["y_mag_um"].astype(float), 0.0)

        self.group_ids = self.df["group_id"].astype(str).to_numpy()
        self.bin_ids = self.df["_bin_id"].to_numpy(dtype=int)
        self.is_zero = self.df["_is_zero"].to_numpy(dtype=bool)

        self.bin_to_indices: dict[int, np.ndarray] = {}
        self.bin_group_to_indices: dict[tuple[int, str], np.ndarray] = {}
        self.bin_group_to_other_indices: dict[tuple[int, str], np.ndarray] = {}

        for b, gdf in self.df.groupby("_bin_id"):
            idxs = gdf["_idx"].to_numpy(dtype=np.int64)
            self.bin_to_indices[int(b)] = idxs

            by_group: dict[str, np.ndarray] = {}
            for gid, gg in gdf.groupby(self.df.loc[gdf.index, "group_id"].astype(str)):
                by_group[str(gid)] = gg["_idx"].to_numpy(dtype=np.int64)
                self.bin_group_to_indices[(int(b), str(gid))] = by_group[str(gid)]

            # Precompute "same bin, different group" pools for O(1)-style lookup.
            all_groups = list(by_group.keys())
            for gid in all_groups:
                others = [by_group[g] for g in all_groups if g != gid]
                if others:
                    self.bin_group_to_other_indices[(int(b), str(gid))] = np.concatenate(others).astype(np.int64)
                else:
                    self.bin_group_to_other_indices[(int(b), str(gid))] = np.empty((0,), dtype=np.int64)

        self.available_bins = sorted(self.bin_to_indices.keys())

    def sample_for_anchors(
        self,
        anchor_indices: np.ndarray,
        top_k_triplets_per_batch: int | None = None,
    ) -> np.ndarray:
        triplets: list[tuple[int, int, int]] = []

        for a_idx in anchor_indices.astype(np.int64).tolist():
            self.stats.anchors_seen += 1

            if self.exclude_zero_from_triplets and self.is_zero[a_idx]:
                self.stats.anchors_skipped_zero += 1
                continue

            a_bin = int(self.bin_ids[a_idx])
            a_group = str(self.group_ids[a_idx])

            pos_pool = self.bin_group_to_other_indices.get((a_bin, a_group), np.empty((0,), dtype=np.int64))
            pos_pool = pos_pool[pos_pool != a_idx]
            if pos_pool.size == 0:
                self.stats.anchors_no_positive += 1
                continue

            # Negatives: prefer semi-hard neighbors first.
            neg_pool = np.empty((0,), dtype=np.int64)
            for nb in [a_bin - 1, a_bin + 1]:
                if nb in self.bin_to_indices:
                    cand = self.bin_to_indices[nb]
                    if cand.size:
                        neg_pool = np.concatenate([neg_pool, cand]) if neg_pool.size else cand

            if neg_pool.size == 0:
                other_bins = [b for b in self.available_bins if b != a_bin]
                if other_bins:
                    pools = [self.bin_to_indices[b] for b in other_bins if self.bin_to_indices[b].size > 0]
                    if pools:
                        neg_pool = np.concatenate(pools)

            if neg_pool.size == 0:
                self.stats.anchors_no_negative += 1
                continue

            p_idx = int(self.rng.choice(pos_pool))
            n_idx = int(self.rng.choice(neg_pool))
            triplets.append((a_idx, p_idx, n_idx))
            self.stats.triplets_built += 1

            if top_k_triplets_per_batch is not None and len(triplets) >= int(top_k_triplets_per_batch):
                break

        if not triplets:
            return np.empty((0, 3), dtype=np.int64)
        return np.asarray(triplets, dtype=np.int64)
