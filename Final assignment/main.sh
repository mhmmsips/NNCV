wandb login

python3 train.py \
    --data-dir /data/Cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --val-split 0.1 \
    --seed 42 \