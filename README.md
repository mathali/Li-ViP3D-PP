# Li-ViP3D++: Query-Gated Deformable Camera–LiDAR Fusion for End-to-End Perception and Trajectory Prediction
### [Paper](https://arxiv.org/pdf/2601.20720)

Official repository for **Li-ViP3D++: Query-Gated Deformable Camera–LiDAR Fusion for End-to-End Perception and Trajectory Prediction**.

Based on [ViP3D](github.com/Tsinghua-MARS-Lab/ViP3D)

## Installation

#### 1) Create conda environment

```bash
conda create -n vip3d python=3.8
conda activate vip3d
```

#### 2) Install PyTorch (CUDA 11.1)

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 3) Install mmcv and mmdet

```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
pip install mmdet==2.24.1
```

#### 4) Install other packages

```bash
pip install -r requirements.txt
```

#### 5) Install mmdet3d

```bash
cd ~
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
pip install -r requirements/runtime.txt
```

## Prepare dataset

#### 1) Download nuScenes v1.0 and map expansion

Download from [nuScenes](https://www.nuscenes.org/download). Keyframe blobs and radar blobs are sufficient.

#### 2) Directory layout

```
LiViP3D++/
├── mmdet3d/
├── plugin/
├── tools/
├── data/
│   └── nuscenes/
│       ├── maps/
│       ├── samples/
│       ├── v1.0-trainval/
│       └── lidarseg/
```

#### 3) Tracking infos

With nuScenes under `data/nuscenes/` (or set `NUSCENES_ROOT`):

```bash
export NUSCENES_ROOT=data/nuscenes   # optional; configs default to data/nuscenes/
python tools/data_converter/nusc_tracking.py
```

## Training

LiViP3D++ trains with **3 historical frames** and a **ResNet-50** backbone. Image-branch weights are initialized from a pretrained DETR3D detector; LiDAR configs additionally load a PointPillars checkpoint into `LidarEncoder`.

### Checkpoints

Place under `ckpts/`:

| File | Purpose |
|------|---------|
| `detr3d_resnet50.pth` | Image backbone / detection head init ([Google Drive](https://drive.google.com/drive/folders/18q2sQ-J-AxqeCO8FaAWKQ9Fi13PPv_MR?usp=drive_link)) |
| `pp_epoch_060.pth` | `LidarEncoder` init for camera–LiDAR configs (optional but recommended; [Google Drive](https://drive.google.com/drive/u/0/folders/19IFzZNhYcxx0pIHe3mV9psbxDPVYN3oC)) |

Configs read the dataset root from the environment:

```bash
export NUSCENES_ROOT=/path/to/nuscenes
```

### Example configs

| Config | Description |
|--------|-------------|
| `plugin/vip3d/configs/vip3d_resnet50_3frame.py` | Camera-only baseline |
| `plugin/vip3d/configs/livip3d_resnet50_3frame_qgdf.py` | LiViP3D++ camera–LiDAR fusion (`Detr3DCamLidarCrossAttenQGDF`), downscaled images, map-free predictor |

### Launch training

Multi-GPU (recommended):

```bash
conda activate vip3d
cd LiViP3D++
export NUSCENES_ROOT=/path/to/nuscenes

bash tools/dist_train.sh \
  plugin/vip3d/configs/livip3d_resnet50_3frame_qgdf.py \
  4 --work-dir=work_dirs/livip3d_qgdf
```

Single GPU:

```bash
PYTHONPATH=. python tools/train.py \
  plugin/vip3d/configs/livip3d_resnet50_3frame_qgdf.py \
  --gpus 1 --work-dir=work_dirs/livip3d_qgdf
```

`tools/train.py` matches the LiViP3D training entry point: FP16-safe normalization patches (when `fp16` is enabled in the config), distributed device placement, and automatic `LidarEncoder` weight loading from `ckpts/pp_epoch_060.pth` when the model defines a lidar encoder.

Training uses ~24 GB GPU memory per device; ~3.5 days for 15 epochs on 4× A100 GPUs (config-dependent).

## Evaluation

### Tracking (AMOTA)

```bash
conda activate vip3d
export NUSCENES_ROOT=/path/to/nuscenes

PYTHONPATH=. python tools/test.py \
  plugin/vip3d/configs/livip3d_resnet50_3frame_qgdf.py \
  work_dirs/livip3d_qgdf/epoch_24.pth \
  --eval bbox
```

**Standard metrics** (`tools/prediction_eval.py`):

```bash
python tools/prediction_eval.py \
  --result_path work_dirs/livip3d_qgdf/results_nusc.json
```

**Extended metrics** (`tools/prediction_eval_advanced.py`) — adds forecasting mAP on the first future frame and related summaries:

```bash
python tools/prediction_eval_advanced.py \
  --result_path work_dirs/livip3d_qgdf/results_nusc.json \
  --prediction_infos_path ./nuscenes_prediction_infos_val.json
```

Metrics are written to `prediction_metrics.json` next to the result path (see script output for the exact directory).

## License

Code and assets are under the Apache 2.0 license.
