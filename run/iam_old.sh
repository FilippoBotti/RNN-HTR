export CUDA_VISIBLE_DEVICES=2

python3 train.py  \
--max-lr 1e-3 \
--train-bs 128 \
--val-bs 8 \
--weight-decay 0.5 \
--mask-ratio 0.4 \
--attn-mask-ratio 0.1 \
--max-span-length 8 \
--img-size 512 64 \
--proj 8 \
--dila-ero-max-kernel 2 \
--dila-ero-iter 1 \
--proba 0.5 \
--alpha 1 \
--total-iter 100000 \
--out-dir /home/botti/checkpoints/htr/iam_old \
--exp-name basic \
--architecture transformer \
--head_type linear \
--mamba_scan_type single \
IAM_OLD

# python3 test.py --exp-name iam \
# --max-lr 1e-3 \
# --train-bs 128 \
# --val-bs 8 \
# --weight-decay 0.5 \
# --mask-ratio 0.4 \
# --attn-mask-ratio 0.1 \
# --max-span-length 8 \
# --img-size 512 64 \
# --proj 8 \
# --dila-ero-max-kernel 2 \
# --dila-ero-iter 1 \
# --proba 0.5 \
# --alpha 1 \
# --total-iter 100000 \
# --out-dir /home/botti/checkpoints/htr/iam \
# --exp-name pretrained \
# --architecture transformer \
# --head_type linear \
# --mamba_scan_type single \
# IAM