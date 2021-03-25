python test.py \
--weights runs/train/exp6/weights/best.pt \
--data data/fold_4_vinbigdata.yaml \
--img-size 1024  \
--conf-thres 0.01 \
--iou-thres 0.5 \
--task test \
--augment --save-txt --save-conf
