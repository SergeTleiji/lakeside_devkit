# Lakeside Devkit

This devkit provides tools for working with the long-term Lakeside robotics
dataset. The dataset contains repeated traversals of the same lakeside route
across different dates and seasons, with on-foot and scooter runs organized
into camera, LiDAR, and metrics components.

The current devkit supports dataset discovery, sequence loading, sensor-frame
iteration, metrics loading, RTK/INS trajectory export, lightweight
visualization, INS-vs-RTK evaluation, and RTK-based cross-season
place-recognition pair generation.

If you use this dataset or devkit in a paper, add the dataset citation here.

```bibtex
@misc{lakeside_dataset,
  title = {Long-Term Lakeside Robotics Dataset},
  note = {Citation to be added}
}
```

## Installation

From source:

```bash
git clone <repo-url>
pip install -e lakeside_devkit
```

Or, from inside this directory:

```bash
pip install -e .
```

Optional dependencies:

```bash
pip install open3d   # binary or compressed PCD loading
```

`pyproj` is used for GPS-to-ENU conversion and is included in the base package
dependencies.

## Dataset Organization

The dataset root should contain folders named:

```text
{date}_{traverse}_{component}
```

Example:

```text
200803_on-foot_lidar
200803_on-foot_metrics
200803_on-foot_cam1
200803_on-foot_cam2
200803_on-foot_cam3
```

Supported traverses:

- `on-foot`
- `scooter`

Supported components:

- `cam1`
- `cam2`
- `cam3`
- `lidar`
- `metrics`

LiDAR files are timestamp-named PCD files:

```text
{root}/{date}_{traverse}_lidar/{date}/*.pcd
```

Metrics are stored in:

```text
{root}/{date}_{traverse}_metrics/{date}/Imu.csv
```

Camera folders contain sequentially indexed images such as `00000.jpg`,
`00020.jpg`, and `00040.jpg`. Images are loaded by sorted frame order. The
camera filenames are frame IDs, not Unix timestamps, unless a separate camera
timestamp file is provided and documented.

Downsampled camera folders are supported as either root-level component folders:

```text
200803_on-foot_cam2_downsampled
```

or camera subfolders:

```text
200803_on-foot_cam2/cam2_downsampled
```

## Comparing KISS-ICP with RTK

The devkit provides `scripts/compare_kiss_icp_rtk.py` to compare a KISS-ICP trajectory with the corresponding RTK measurements.

Run the following commands from the root of the `lakeside_devkit` repository, with your Python environment activated.

### 1. Install KISS-ICP

Install KISS-ICP version 1.3.0 with its optional dependencies for visualization and dataset loading:

```bash
python3 -m pip install "kiss-icp[all]==1.3.0"
```

### 2. Generate the LiDAR odometry trajectory

Run KISS-ICP on the sequence's PCD directory:

```bash
kiss_icp_pipeline --visualize DATA/201210_on-foot_lidar/201210/
```

Visualization is optional; omit `--visualize` to run without the graphical interface.

After the run completes, the trajectory used in this example is available at:

```text
results/latest/201210_poses_tum.txt
```

### 3. Compare the trajectory with RTK measurements

Run the comparison using the generated trajectory, the same PCD directory, and the corresponding `Imu.csv` file:

```bash
python3 scripts/compare_kiss_icp_rtk.py \
    --skip-rtk-status-2 \
    --icp results/latest/201210_poses_tum.txt \
    --pcd-dir DATA/201210_on-foot_lidar/201210/ \
    --rtk DATA/allmetrics/201210_on-foot_metrics/201210/Imu.csv \
    --rtk-time-column timestamp \
    --output-dir . \
    --segment-m 50 \
    --sample-interval 1 \
    --rtk-fix-rate 5
```

This example excludes RTK status 2 measurements and writes the comparison outputs to the current directory.

For another sequence, update the trajectory, PCD, and RTK paths together. `results/latest` refers to the most recent KISS-ICP run.

## Example Usage

```python
from pathlib import Path

from lakeside import LakesideDataset, LakesideSequence

root = Path("/path/to/lakeside/root")

dataset = LakesideDataset(root)
print(dataset.discover_sequences())

seq = LakesideSequence(
    root,
    date="200803",
    traverse="on-foot",
    use_downsampled_images=True,
)

# Dataset summary
print(seq.summary())

# LiDAR frames are timestamped by filename.
lidar_frame = seq.get_lidar(0)
print(lidar_frame.index, lidar_frame.timestamp, lidar_frame.path)

# Camera frames are sequential frame IDs loaded by sorted frame order.
cam_frame = seq.get_camera("cam1", 0)
print(cam_frame.index, cam_frame.frame_id, cam_frame.path)

# Metrics and RTK status.
metrics = seq.load_metrics()
print(metrics.summary())
print(metrics.rtk_status_counts())

# Nearest LiDAR-to-metrics synchronization.
row = seq.get_metrics_for_lidar(0)
print(row["timeStamp"], row.get("RTK_Status"))
```

Runnable examples:

```bash
python examples/minimal_load.py --root DATA --date 200803 --traverse on-foot --downsampled
python examples/rtk_filtering.py --root DATA --date 200803 --traverse on-foot --out outputs/rtk_fix_rows.csv
python examples/image_triplet.py --root DATA --date 200803 --traverse on-foot --index 0 --downsampled --out outputs/image_triplet.png
```

## Command-Line Tools

Inspect a sequence:

```bash
python scripts/inspect_dataset.py --root DATA --date 200803 --traverse on-foot
```

List LiDAR frames:

```bash
python scripts/list_frames.py \
  --root DATA \
  --date 200803 \
  --traverse on-foot \
  --component lidar \
  --limit 10
```

List camera frames:

```bash
python scripts/list_frames.py \
  --root DATA \
  --date 200803 \
  --traverse on-foot \
  --component cam1 \
  --downsampled \
  --limit 10
```

Summarize RTK status:

```bash
python scripts/rtk_summary.py \
  --root DATA \
  --date 200803 \
  --traverse on-foot
```

Plot a trajectory:

```bash
python scripts/plot_trajectory.py \
  --root DATA \
  --date 200803 \
  --traverse on-foot \
  --source rtk \
  --min-status 3
```

View a camera triplet:

```bash
python scripts/view_image_triplet.py \
  --root DATA \
  --date 200803 \
  --traverse on-foot \
  --index 0 \
  --downsampled
```

View a LiDAR frame:

```bash
python scripts/view_lidar.py \
  --root DATA \
  --date 200803 \
  --traverse on-foot \
  --index 0 \
  --pcd-backend auto
```

Export RTK or INS trajectories in TUM format:

```bash
python scripts/export_tum.py \
  --root DATA \
  --date 200803 \
  --traverse on-foot \
  --source rtk \
  --min-status 3 \
  --out outputs/200803_rtk_tum.txt
```

```bash
python scripts/export_tum.py \
  --root DATA \
  --date 200803 \
  --traverse on-foot \
  --source ins \
  --out outputs/200803_ins_tum.txt
```

Export KITTI-style INS poses:

```bash
python scripts/export_kitti_poses.py \
  --root DATA \
  --date 200803 \
  --traverse on-foot \
  --source ins \
  --out outputs/poses.txt
```

Evaluate INS against RTK:

```bash
python scripts/evaluate_ins_vs_rtk.py \
  --root DATA \
  --date 200803 \
  --traverse on-foot \
  --min-status 3
```

Evaluate two TUM trajectories already expressed in the same coordinate frame:

```bash
python scripts/evaluate_trajectory.py \
  --gt outputs/200803_rtk_tum.txt \
  --pred outputs/200803_ins_tum.txt \
  --max-dt 0.05
```

Generate cross-season LiDAR/RTK place-recognition pairs:

```bash
python scripts/make_place_recognition_pairs.py \
  --root DATA \
  --ref-date 200803 \
  --query-date 210115 \
  --traverse on-foot \
  --distance-threshold 5.0 \
  --out outputs/place_pairs_200803_210115.json
```

Evaluate place-recognition predictions:

```bash
python scripts/evaluate_place_recognition.py \
  --pairs outputs/place_pairs_200803_210115.json \
  --pred outputs/retrieval_predictions.json
```

Prediction JSON format:

```json
{
  "predictions": [
    {
      "query_index": 0,
      "retrieved_ref_indices": [5, 10, 11, 20, 30]
    }
  ]
}
```
## LiDAR odometry example with KISS-ICP

This example was tested on the **201210 on-foot survey (10 December 2020)** using **KISS-ICP 1.3.0**. It processes the released PCD scans directly, using LiDAR geometry and the default KISS-ICP configuration.

### Prepare the data

Download the LiDAR ZIP package for the **201210 on-foot survey** from the [dataset website](https://dream.georgiatech-metz.fr/datasets/backpack/index.html) and extract the individual PCD scans into a directory. The directory used for this run was:

```text
/home/gchahine/Downloads/201210/201210
```

The path passed to the pipeline should point to the directory containing the individual `.pcd` files.

### Install and run

Install the tested KISS-ICP version and Open3D, which is used to load the PCD files:

```bash
pip install kiss-icp==1.3.0 open3d
kiss_icp_pipeline --version
```

The version command should report:

```text
KISS-ICP Version: 1.3.0
```

Run the pipeline with visualization enabled:

```bash
kiss_icp_pipeline --visualize \
  /home/gchahine/Downloads/201210/201210
```

Replace the path with the location of your extracted PCD files. This run uses the default configuration; no custom configuration file is required.

### Expected output

The visualization shows the registered LiDAR point clouds and the estimated trajectory. After processing, the pipeline exports the estimated poses and the run configuration to its results directory.

A recording of the successful run is available here: [KISS-ICP on the 201210 on-foot survey](https://youtu.be/p5gHWgLPDhA).

The recording provides a qualitative illustration of the estimated trajectory and point-cloud registration.

For implementation details, see the [KISS-ICP repository](https://github.com/PRBonn/kiss-icp).
## Notes

LiDAR filenames and `Imu.csv` rows use Unix timestamps, so LiDAR-to-metrics
matching uses nearest timestamp. Camera images are sequentially indexed and
loaded by sorted frame order; the devkit does not fabricate camera timestamps.

Semantic labels, if present in derived files, should be described as
algorithmically generated semantic labels, not manual ground truth.

The INS-vs-RTK benchmark is a loading, filtering, coordinate-conversion, and
evaluation sanity check. It is not a SLAM benchmark. The place-recognition
dummy baseline retrieves by sequence index order only and is included to check
the benchmark format and evaluator.

See [overview.md](overview.md) for a function-by-function overview of the
package.
