# Robustness Evaluation

## Models
| Model | ImageNet-C | ImageNet-A | ImageNet-R | ImageNet-Sketch |
|:-:|:-:|:-:|:-:|:-:|
| [LSNet-T](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_t.pth) | 68.2 | 6.7  | 38.5 | 25.5 |
| [LSNet-S](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_s.pth) | 65.7 | 9.6  | 39.4 | 27.5 |
| [LSNet-B](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_b.pth) | 59.3 | 17.3 | 43.1 | 30.7 |

## Data preparation

Please download and prepare ImageNet-C, ImageNet-A, ImageNet-R, ImageNet-Sketch datasets. 

## Testing
```bash
set -e
set -x

MODEL=lsnet_t
CKPT=pretrain/lsnet_t.pth
INPUT=224

# Optional for mirror
# export HF_ENDPOINT=https://hf-mirror.com

python main.py --eval --model ${MODEL} --resume ${CKPT} --data-path ~/imagenet \
--inc_path ~/datasets/OpenDataLab___ImageNet-C/raw \
--insk_path ~/datasets/OpenDataLab___ImageNet-Sketch/raw/sketch \
--ina_path ~/datasets/OpenDataLab___ImageNet-A/raw/imagenet-a \
--inr_path ~/datasets/OpenDataLab___ImageNet-R/raw/imagenet-r \
--batch-size 512 \
--input-size ${INPUT}
```
