# PQR3D


This is the official PyTorch implementation for our paper:

**PQR3D: Progressive Query Refinement over Reference-Conditioned Temporal Windows for Multi-View 3D Object Detection**


## News

* 2026-09-25: We release the source code and pretrained weights.


## Model Zoo

| Setting  | Pretrain |  epochs| Training Cost | NDS<sub>val</sub> | NDS<sub>test</sub>  | Weights | Log
|----------|:--------:|:-------------:|:-----------------:|:------------------: |:-------:|:-------:|:-------:|
| [r50_nuimg_704x256](configs/r50_nuimg_704x256.py) | [nuImg](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) | 24| 13h (8x4090) | 58.2 | -  | [ckpt](https://drive.google.com/file/d/1MkX2vfkRp8pD7RjJ1X9gDqLAVsoYPnMf/view?usp=share_link) |[log](https://drive.google.com/file/d/1RhHzy9a3WT4UmpMdUT-d0Sby3A95MsfO/view?usp=share_link) |
| [r50_nuimg_704x256_625q](configs/r50_nuimg_704x256_625q_48ep.py) | [nuImg](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) | 48| 22h (8x4090) | 58.2 | -  | [ckpt](https://drive.google.com/file/d/1DzBCKxZm3BqvM8F6UbvSBTkDhn0-QOS_/view?usp=share_link) |[log](https://drive.google.com/file/d/1K5MFWONSF5gcl0dyopYGwG66FWqgFZuU/view?usp=share_link) |
| [r101_nuimg_1408x512](configs/r101_nuimg_1408x512.py) | [nuImg](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r101_fpn_1x_nuim/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth) | 24  | 1d16h (8x4090) | 60.5 | - |  [ckpt](https://drive.google.com/file/d/1BlB0Qz7iTdPoI1EUga6UdqBjCQkDeiTQ/view?usp=share_link) |[log](https://drive.google.com/file/d/1pLEmfZEy_a_oIp0VO0cwqWN0JUNuDOkP/view?usp=share_link) |
| [vov99_dd3d_1600x640](configs/vov99_dd3d_1600x640_trainval_future.py) | [DD3D](https://drive.google.com/file/d/1gQkhWERCzAosBwG5bh2BKkt1k0TJZt-A/view) | 24| 11d7h (8xA100) | - | 68.0  | [ckpt](https://drive.google.com/file/d/1BlB0Qz7iTdPoI1EUga6UdqBjCQkDeiTQ/view?usp=share_link) |[log](https://drive.google.com/file/d/1uOOjl_m-SgG0EyJBButrzD_9hVB0X_Eb/view?usp=share_link) |
|[vit_eva02_1600x640](configs/vit_eva02_1600x640_trainval_future.py) | [EVA02](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_seg_sys_o365.pth) | 24 | 14d16h (8xA100) | - | 71.6  | [ckpt](https://drive.google.com/file/d/1U1fFmqRLkda5HNOFQjprCNzJm8ITvAMI/view?usp=sharing) |[log](https://drive.google.com/file/d/1RwM9w_7-p1vEDHYCMPv04VBIX6_jQkpb/view?usp=share_link) |




## Environment

Install PyTorch 2.0 + CUDA 11.8:

```
conda create -n propbev python=3.8
conda activate propbev
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```


Install other dependencies:

```
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
mim install mmdet3d==1.0.0rc6
pip install setuptools==59.5.0
pip install numpy==1.23.5
```

Install turbojpeg and pillow-simd to speed up data loading (optional but important):

```
sudo apt-get update
sudo apt-get install -y libturbojpeg
pip install pyturbojpeg
pip uninstall pillow
pip install pillow-simd==9.0.0.post1
```

Compile CUDA extensions:

```
cd models/csrc
python setup.py build_ext --inplace
```

## Prepare Dataset

1. Download nuScenes from [https://www.nuscenes.org/nuscenes](https://www.nuscenes.org/nuscenes) and put it in `data/nuscenes`.
2. Download the generated info file from [Google Drive](https://drive.google.com/drive/folders/1UXqE0ysvxlqXudBt7aCZ0JHRhDwGC2uQ?usp=drive_link).
3. Folder structure:

```
data/nuscenes
├── maps
├── nuscenes_infos_test_sweep.pkl
├── nuscenes_infos_train_sweep.pkl
├── nuscenes_infos_train_mini_sweep.pkl
├── nuscenes_infos_val_sweep.pkl
├── nuscenes_infos_val_mini_sweep.pkl
├── samples
├── sweeps
├── v1.0-test
└── v1.0-trainval
```


## Training

Download pretrained weights and put it in directory `pretrain/`:

```
pretrain
├── cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth
├── cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth
```

Train PropBEV with 8 GPUs:

```
torchrun --nproc_per_node 8 train.py --config configs/r50_nuimg_704x256.py
```

Train PropBEV with 4 GPUs (i.e the last four GPUs):

```
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node 4 train.py --config configs/r50_nuimg_704x256.py
```

The batch size for each GPU will be scaled automatically. So there is no need to modify the `batch_size` in config files.

## Evaluation

Single-GPU evaluation:

```
export CUDA_VISIBLE_DEVICES=0
python val.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```

Multi-GPU evaluation:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 val.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```

## Timing

FPS is measured with a single GPU:

```
export CUDA_VISIBLE_DEVICES=0
python timing.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```

## Visualization

Visualize the predicted bbox:

```
python viz_bbox_predictions.py --config configs/r50_nuimg_704x256.py --weights checkpoints/r50_nuimg_704x256.pth
```


## Acknowledgements

Our implementation is based on these excellent open-source projects: [SparseBEV](https://github.com/MCG-NJU/SparseBEV), [Sparse4D](https://github.com/HorizonRobotics/Sparse4D),  [StreamPETR](https://github.com/exiawsh/StreamPETR), [HoP](https://github.com/Sense-X/HoP), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
