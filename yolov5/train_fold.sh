IMG_SIZE=$1
BATCH_SIZE=$2
EPOCHS=$3

for i in {0..4}; do

python train.py --img-size ${IMG_SIZE} \
--cfg models/yolov5x6.yaml \
--batch ${BATCH_SIZE} \
--epochs ${EPOCHS} \
--data data/fold_${i}_vinbigdata.yaml \
--weights yolov5x.pt \
--hyp hyp.scratch.yaml 

done

# for i in {0..4}; do

# python test.py \
# --weights runs/train/exp${i}/weights/best.pt \
# --data data/fold_${i}_vinbigdata.yaml \
# --img-size 1500 \
# --batch-size ${BATCH_SIZE} \
# --conf-thres 0.01 \
# --iou-thres 0.5 \
# --task test \
# --augment --save-txt --save-conf
# done