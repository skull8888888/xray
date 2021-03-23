_base_ = 'models/cascade_resnest_50.py'
fp16 = dict(loss_scale=512.)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.0, 0.1],
        contrast_limit=[0.0, 0.1],
        p=0.3),
#     dict(
#         type="OneOf",
#         transforms=[
#             dict(type="Blur"),
#             dict(type="GaussNoise"),
#         ],
#         p=0.3,
#     ),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0,
        scale_limit=0.05,
        rotate_limit=3,
#         interpolation=1,
        p=0.5),
]

img_size = 1500
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type='Resize', img_scale=(img_size, img_size), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Pad", size_divisor=32),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize',**img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type='Collect', keys=['img'])
        ])
]
classes = (
    "Aortic enlargement", 
    "Atelectasis", 
    "Calcification", 
    "Cardiomegaly", 
    "Consolidation", 
    "ILD", 
    "Infiltration", 
    "Lung Opacity", 
    "Nodule/Mass", 
    "Other lesion", 
    "Pleural effusion", 
    "Pleural thickening", 
    "Pneumothorax", 
    "Pulmonary fibrosis")

dataset_type = 'CocoDataset'
data_root = 'vinbigdata/images/train/'
batch_size = 8

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.05,
        dataset=dict(
            type='CocoDataset',
            ann_file='fold_3_abnormal_train_org_size.json',
            img_prefix=data_root,
            classes=classes,
            pipeline=train_pipeline)
    ),
#     train=dict(
#         type='CocoDataset',
#         ann_file='fold_3_abnormal_train_org_size.json',
#         img_prefix=data_root,
#         classes=classes,
#         pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        samples_per_gpu=batch_size,
        ann_file='fold_3_abnormal_valid_org_size.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        samples_per_gpu=batch_size,
        ann_file='test_coco_org.json',
        img_prefix='vinbigdata/test/',
        classes=classes,
        pipeline=test_pipeline)
        )

evaluation = dict(interval=1, metric=['bbox'], iou_thrs=[0.4])

optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

runner = dict(type='EpochBasedRunner', max_epochs=12)

checkpoint_config = dict(interval=1)

log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = 'checkpoints_1500_resnest50_fold_3/'
gpu_ids = [0,1]