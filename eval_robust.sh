set -e
set -x

MODEL=$1
CKPT=$2
INPUT=$3

export HF_ENDPOINT=https://hf-mirror.com

python main.py --eval --model ${MODEL} --resume ${CKPT} --data-path ~/imagenet \
--inc_path ~/datasets/OpenDataLab___ImageNet-C/raw \
--insk_path ~/datasets/OpenDataLab___ImageNet-Sketch/raw/sketch \
--ina_path ~/datasets/OpenDataLab___ImageNet-A/raw/imagenet-a \
--inr_path ~/datasets/OpenDataLab___ImageNet-R/raw/imagenet-r \
--batch-size 512 \
--input-size ${INPUT}