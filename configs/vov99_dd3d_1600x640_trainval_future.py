_base_ = ['./r50_nuimg_704x256.py']

num_frames = 15

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_backbone = dict(
    _delete_=True,
    type='VoVNet',
    spec_name='V-99-eSE',
    out_features=['stage2', 'stage3', 'stage4', 'stage5'],
    norm_eval=True,
    frozen_stages=1,
    with_cp=True
)
img_neck=dict(
    _delete_=True,
    type='FPN',
    in_channels=[256, 512, 768, 1024],
    out_channels=256,
    num_outs=5
)
img_norm_cfg = dict(
    _delete_=True,
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False
)

model = dict(
    data_aug=dict(
        img_color_aug=True,
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)
    ),
    img_backbone=img_backbone,
    img_neck=img_neck,
    # ── Temporal propagation ──
    num_propgated=256,
    use_t2=False,  # Disable t-2 keyframe to save memory with large backbone
    t1_slot=1,   # Interleaved: [curr, prev1, next1, prev2, ...] → t-1 at slot 1
    t2_slot=3,   # Interleaved: [curr, prev1, next1, prev2, ...] → t-2 at slot 3
    pts_bbox_head=dict(
        num_query=1600,
        transformer=dict(
            num_levels=5,
            num_points=4,
            num_frames=num_frames
        )
    )
)

ida_aug_conf = {
    'resize_lim': (0.94, 1.25),
    'final_dim': (640, 1600),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweepsFutureInterleave', prev_sweeps_num=7, next_sweeps_num=7),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05],
            flip_dx_ratio=0.2, flip_dy_ratio=0.2),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'], meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape',
        'lidar2img', 'lidar2img_t1', 'lidar2img_t2', 'img_timestamp',
        'ego_pose', 'ego_pose_t1', 'ego_pose_t2'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweepsFutureInterleave', prev_sweeps_num=7, next_sweeps_num=7, test_mode=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img'], meta_keys=(
                'filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape',
                'lidar2img', 'lidar2img_t1', 'lidar2img_t2', 'img_timestamp',
                'ego_pose', 'ego_pose_t1', 'ego_pose_t2'))
        ])
]

data = dict(
    train=dict(
        ann_file=['data/nuscenes/nuscenes_infos_train_sweep.pkl',
                  'data/nuscenes/nuscenes_infos_val_sweep.pkl'],
        pipeline=train_pipeline),
    val=dict(
        ann_file='data/nuscenes/nuscenes_infos_val_sweep.pkl',  # use nuscenes_infos_test_sweep.pkl for submission
        pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01
)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2)
)

# learning policy
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)


# load pretrained weights
load_from = 'pretrain/dd3d_det_final.pth'
revise_keys = None

# evaluate every epoch
eval_config = dict(interval=1)