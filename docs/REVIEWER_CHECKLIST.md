# Reviewer Checklist

The Lakeside devkit supports the core workflows needed to inspect the dataset
and reproduce simple reviewer-facing sanity benchmarks.

## Supported Workflows

- Dataset discovery from `{date}_{traverse}_{component}` folders.
- Simple sequence loading with `LakesideSequence`.
- Sequential LiDAR and camera frame iteration.
- Camera loading by sorted frame order. Camera filenames are sequential frame
  IDs, not Unix timestamps, unless a separate camera timestamp file is later
  documented.
- LiDAR-to-metrics timestamp matching using LiDAR filename Unix timestamps and
  `Imu.csv` `timeStamp` values.
- RTK-status-aware filtering from `Imu.csv`.
- RTK and INS trajectory export in TUM format.
- INS pose export in KITTI 3x4 matrix format.
- INS-vs-RTK trajectory evaluation as a sanity benchmark.
- Cross-season LiDAR/RTK place-recognition pair generation and Recall@K
  evaluation.
- Visualization of camera triplets, LiDAR frames, and RTK/INS trajectories.

## Important Claims

- Semantic labels, when present, should be described as algorithmically
  generated semantic labels, not manual ground truth.
- Camera images are sequentially indexed and loaded by sorted frame order. Do
  not claim camera timestamp synchronization unless a camera timestamp file is
  present and documented.
- The included place-recognition dummy baseline is a placeholder that retrieves
  by sequence index order. It is not a learned perception method.
- The INS-vs-RTK benchmark is a sanity benchmark for loading, filtering,
  coordinate conversion, and evaluation. It should not be described as SLAM
  performance.

## Quick Smoke Commands

```bash
python scripts/inspect_dataset.py --root DATA --date 200803 --traverse on-foot
python scripts/list_frames.py --root DATA --date 200803 --traverse on-foot --component lidar
python scripts/rtk_summary.py --root DATA --date 200803 --traverse on-foot
python scripts/evaluate_ins_vs_rtk.py --root DATA --date 200803 --traverse on-foot --min-status 3
```
