wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.1 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "UNet_bs16_30epochs_lr01_SGD_momentum09_wd1e-3_labelsmoothing01_CosineAnnealing_dataaugmentation_focalloss" \