# Lakeside Devkit Function Overview

This file summarizes the public classes, functions, scripts, and examples
provided by the Lakeside devkit.

## Core Dataset API

- `LakesideDataset(root)` - Top-level dataset object that discovers available
  Lakeside sequences under a dataset root.
- `LakesideDataset.discover_sequences()` - Scans component folders and returns
  available `(date, traverse)` sequence IDs.
- `LakesideDataset.sequences()` - Returns discovered sequences as
  `LakesideSequence` objects.
- `LakesideDataset.sequence(date, traverse)` - Opens one specific traversal
  sequence.

## Sequence API

- `LakesideSequence(root, date, traverse, use_downsampled_images=False)` - Main
  object for accessing one recording date and traversal type.
- `get_component_path(component)` - Resolves the folder for `cam1`, `cam2`,
  `cam3`, `lidar`, or `metrics`.
- `lidar_files()` - Lists LiDAR `.pcd` files sorted by timestamp filename.
- `camera_files(camera)` - Lists camera images sorted numerically by frame ID.
- `metrics_file()` - Returns the path to `Imu.csv` if present.
- `summary()` - Returns component paths, sensor counts, first/last files, and
  metrics availability.

## Frame Iteration

- `LidarFrame` - Dataclass storing LiDAR path, Unix timestamp, and sequence
  index.
- `CameraFrame` - Dataclass storing image path, camera name, frame ID, and
  sequence index.
- `iter_lidar()` - Iterates through LiDAR frames in timestamp order.
- `iter_camera(camera)` - Iterates through camera frames by sorted frame ID.
- `get_lidar(index)` - Returns one LiDAR frame by sequence index.
- `get_camera(camera, index)` - Returns one camera frame by sequence index.
- `get_lidar_by_timestamp(timestamp)` - Finds the nearest LiDAR frame to a Unix
  timestamp.
- `get_camera_by_frame_id(camera, frame_id)` - Finds a camera frame by frame
  ID, optionally nearest.

## Sensor Loading

- `load_image(camera_frame)` - Loads a camera image as an RGB NumPy array.
- `load_pcd(path, backend="auto")` - Loads PCD XYZ points using the internal
  ASCII parser or optional Open3D.
- `load_pcd_fields(path, backend="auto")` - Loads XYZ and optional fields such
  as `intensity` or `semantic` when available.
- `load_pcd_xyz(path, backend="auto")` - Convenience wrapper returning only XYZ
  points.

## Metrics, IMU, RTK, and INS

- `load_metrics_csv(path)` - Loads `Imu.csv`, with or without a header, into a
  pandas DataFrame.
- `MetricsData` - Wrapper around metrics rows with RTK and INS helper methods.
- `MetricsData.timestamps()` - Returns metrics timestamps.
- `MetricsData.rtk_status_counts()` - Counts each RTK status value.
- `MetricsData.rtk_status_percentages()` - Computes RTK status percentages.
- `MetricsData.filter_by_rtk_status(min_status=3)` - Keeps rows whose
  `RTK_Status` is at least `min_status`.
- `MetricsData.get_nearest_row(timestamp)` - Finds the metrics row nearest to a
  Unix timestamp.
- `MetricsData.get_rtk_positions(min_status=None)` - Extracts RTK position
  columns, optionally filtered by RTK status.
- `MetricsData.get_ins_positions()` - Extracts INS position columns.
- `MetricsData.rtk_to_enu()` - Converts RTK latitude/longitude/altitude to
  local ENU coordinates.
- `MetricsData.ins_to_enu()` - Converts INS latitude/longitude/altitude to
  local ENU coordinates.
- `MetricsData.summary()` - Returns row count, timestamp range, and RTK status
  statistics.

## Synchronization

- `nearest_index(sorted_values, query_value)` - Finds the nearest value index in
  a sorted numeric list.
- `nearest_timestamp(sorted_timestamps, query_timestamp)` - Finds the nearest
  timestamp index.
- `match_lidar_to_metrics(sequence, lidar_frame)` - Matches one LiDAR frame to
  the nearest metrics row.
- `match_lidar_frames_to_metrics(sequence, limit=None)` - Matches multiple
  LiDAR frames to metrics rows.

Camera frames are not timestamp synchronized by this devkit. Camera filenames
are sequential frame IDs and are loaded by sorted frame order.

## Coordinate Transforms

- `latlon_to_enu()` - Converts GPS latitude/longitude/altitude to local ENU
  using `pyproj`.
- `normalize_quat(q)` - Normalizes quaternion `[qx, qy, qz, qw]`.
- `quat_to_rotmat(qx, qy, qz, qw)` - Converts a quaternion to a rotation
  matrix.
- `make_transform(R, t)` - Builds a 4x4 homogeneous pose matrix.
- `invert_transform(T)` - Inverts a 4x4 homogeneous pose matrix.
- `identity_transform()` - Returns an identity transform container.

## Trajectory Evaluation

- `load_tum_trajectory(path)` - Loads a TUM trajectory text file.
- `associate_by_timestamp(gt, pred, max_dt=0.05)` - Associates predicted and
  ground-truth poses by nearest timestamp.
- `compute_position_errors(gt_xyz, pred_xyz)` - Computes Euclidean position
  errors.
- `compute_ate(gt_xyz, pred_xyz)` - Computes absolute trajectory error RMSE.
- `summarize_errors(errors)` - Computes count, RMSE, mean, median, 95th
  percentile, and max.

## Place Recognition

- `make_reference_database(sequence)` - Builds a reference LiDAR/RTK database.
- `make_query_set(sequence)` - Builds a query LiDAR/RTK set.
- `compute_ground_truth_matches()` - Finds positive reference frames within a
  distance threshold.
- `evaluate_retrieval()` - Computes Recall@K for retrieval predictions.
- `dummy_index_retrieval()` - Placeholder baseline that retrieves by sequence
  index order only.

## Scripts

- `inspect_dataset.py` - Prints resolved component paths, sensor counts, and
  first/last files.
- `validate_dataset.py` - Lists discovered sequences under a dataset root.
- `list_frames.py` - Lists LiDAR or camera frame metadata.
- `rtk_summary.py` - Prints RTK status distribution.
- `export_rtk_csv.py` - Exports RTK position rows.
- `sync_lidar_metrics.py` - Prints LiDAR-to-metrics timestamp matches.
- `plot_trajectory.py` - Plots RTK or INS trajectories.
- `view_image_triplet.py` - Saves a cam1/cam2/cam3 triplet by sorted frame
  index.
- `view_lidar.py` - Plots a LiDAR XY scatter from PCD.
- `export_tum.py` - Exports RTK or INS trajectory in TUM format.
- `export_kitti_poses.py` - Exports INS trajectory in KITTI pose format.
- `evaluate_trajectory.py` - Evaluates a predicted TUM trajectory against a
  ground-truth TUM trajectory.
- `evaluate_ins_vs_rtk.py` - Runs the INS-vs-RTK sanity benchmark.
- `make_place_recognition_pairs.py` - Generates cross-season
  place-recognition positive pairs.
- `evaluate_place_recognition.py` - Evaluates retrieval predictions with
  Recall@K.

## Examples

- `examples/minimal_load.py` - Demonstrates sequence loading and frame metadata
  access.
- `examples/rtk_filtering.py` - Demonstrates RTK status filtering and CSV
  export.
- `examples/image_triplet.py` - Demonstrates loading and saving a camera
  triplet preview.

## Lakeside-Specific Behavior

The devkit follows common robotics dataset-devkit patterns: dataset and
sequence objects, frame iteration, sensor loaders, pose transforms,
visualization scripts, and evaluation helpers.

The Lakeside-specific parts are:

- folder names use `{date}_{traverse}_{component}`;
- traverses are `on-foot` and `scooter`;
- components are `cam1`, `cam2`, `cam3`, `lidar`, and `metrics`;
- LiDAR filenames are Unix timestamps;
- camera filenames are sequential frame IDs, not timestamps;
- metrics are loaded from `{date}_{traverse}_metrics/{date}/Imu.csv`;
- RTK/INS columns and RTK status filtering are built around the Lakeside
  `Imu.csv` schema;
- cross-season place-recognition pairs are generated from repeated RTK
  positions along the lakeside route.
