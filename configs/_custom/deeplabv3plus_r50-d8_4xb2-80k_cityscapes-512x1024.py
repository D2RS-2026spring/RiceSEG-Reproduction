_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 1024)
data_root = 'data/Rice_Blast_Leaf/0121'

model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size),
    decode_head=dict(
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        num_classes=2,),
    test_cfg=dict(mode='slide',crop_size=crop_size, stride=(341, 682))
)

load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize',scale=(3000,1000),keep_ratio=True),
    dict(type='RandomCrop',crop_size=crop_size,cat_max_ratio=0.75),
    dict(type='RandomFlip',prob=0.5), 
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',scale=(3000,1000),keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

# DataLoader
metainfo = dict(
    classes=['background', 'rice_blast_leaf'],
    palette=[[0, 0, 0], [0, 255, 0]]
)
dataset_type = 'BaseSegDataset'

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/images', seg_map_path='train/annotations'),
        metainfo=metainfo,
        pipeline=train_pipeline,
        img_suffix='.jpg',
        seg_map_suffix='.png'
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/annotations'),
        metainfo=metainfo,
        pipeline=test_pipeline,
        img_suffix='.jpg',
        seg_map_suffix='.png'
    )
)
test_dataloader = val_dataloader

# tensorboard
vis_backend = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backend,
    name='visualizer'
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50,log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', 
                    by_epoch=False, interval=400, 
                    max_keep_ckpts=3,
                    save_best='mIoU',
                    rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook',)
)

val_evaluator = dict(type='IoUMetric',iou_metrics=['mIoU','mDice'])
test_evaluator = val_evaluator

train_cfg = dict(type='IterBasedTrainLoop', max_iters=4000, val_interval=400)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)  
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=4000, by_epoch=False)
]

strict = False