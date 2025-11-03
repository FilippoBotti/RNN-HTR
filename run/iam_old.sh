export CUDA_VISIBLE_DEVICES=3

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
--out-dir /home/botti/checkpoints/htr/iam_bidimamba \
--exp-name no_sam_bimamba_bidimamba \
--architecture bidimamba \
--head_type bidimamba \
--mamba_scan_type double \
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
# --out-dir /home/botti/checkpoints/htr/iam_old \
# --exp-name bimamba_lr_fisso \
# --architecture bidimamba \
# --head_type bilstm \
# --mamba_scan_type single \
# IAM_OLD