model = dict(
    # model training and testing settings
    train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict( # The config to generate proposals during training
            nms_across_levels=False,  
            nms_pre=2000,  # The number of boxes before NMS
            nms_post=1000,  # The number of boxes to be kept by NMS, Only work in `GARPNHead`.
            max_per_img=1000,  # The number of boxes to be kept after NMS.
            nms=dict( # Config of nms
                type='nms',  #Type of nms
                iou_threshold=0.6 # NMS threshold
                ),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(type="RandomSampler", num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False,
            ),
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(type="RandomSampler", num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False,
            ),
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(type="RandomSampler", num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False,
            ),
        ],
    ),
    test_cfg = dict(
        rpn=dict( # The config to generate proposals during training
            nms_across_levels=False,  
            nms_pre=2000,  # The number of boxes before NMS
            nms_post=1000,  # The number of boxes to be kept by NMS, Only work in `GARPNHead`.
            max_per_img=1000,  # The number of boxes to be kept after NMS.
            nms=dict( # Config of nms
                type='nms',  #Type of nms
                iou_threshold=0.7 # NMS threshold
                ),
            min_bbox_size=0),
        rcnn=dict(score_thr=0.001, nms=dict(type="nms", iou_threshold=0.7), max_per_img=100),
    )
)
