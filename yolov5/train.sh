FOLD=$1
IMG_SIZE=$2

python train.py --img-size ${IMG_SIZE} \
--cfg models/yolov5x.yaml \
--batch 16 \
--epochs 30 \
--data data/fold_${FOLD}_minority_vinbigdata.yaml \
--weights yolov5x.pt \
--hyp finetune.yaml 