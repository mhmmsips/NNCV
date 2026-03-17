wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.1 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "NEW_Unet_512x512" \
    # --backbone "facebook/dinov2-small" \
    # --decoder "upsample"