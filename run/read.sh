export CUDA_VISIBLE_DEVICES=0
# python3 train.py --exp-name read \
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
# --out-dir /home/botti/checkpoints/htr/read \
# --exp-name bidimamba_no_sam \
# --architecture bidimamba \
# --head_type bilstm \
# --mamba_scan_type single \
# READ 

python3 test.py --exp-name read \
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
--out-dir /home/botti/checkpoints/htr/read \
--exp-name bidimamba_no_sam \
--architecture bidimamba \
--head_type bilstm \
--mamba_scan_type single \
READ