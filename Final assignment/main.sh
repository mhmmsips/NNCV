wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 30 \
    --lr 0.1 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "BIGGER_and_UNFROZEN_dinov2-small_upsample_TUNED_4BLOCKS" \
    --backbone "facebook/dinov2-small" \
    --decoder "upsample"