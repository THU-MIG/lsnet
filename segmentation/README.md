# Semantic Segmentation 

Segmentation on ADE20K is implemented based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

## Models
| Model | mIoU | Log |
|:-:|:-:|:-:|
| [LSNet-T](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_t_semfpn.pth) | 40.1 | [lsnet_t_semfpn.json](./logs/lsnet_t_semfpn.json) |
| [LSNet-S](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_s_semfpn.pth) | 41.6 | [lsnet_s_semfpn.json](./logs/lsnet_s_semfpn.json) |
| [LSNet-B](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_b_semfpn.pth) | 43.1 | [lsnet_b_semfpn.json](./logs/lsnet_b_semfpn.json) |

## Requirements
```bash
pip install mmsegmentation==0.30.0
```

## Data preparation

Please prepare ADE20K dataset following [insructions in MMSeg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets). 
The data should appear as: 
```
├── segmentation
│   ├── data
│   │   ├── ade
│   │   │   ├── ADEChallengeData2016
│   │   │   │   ├── annotations
│   │   │   │   │   ├── training
│   │   │   │   │   ├── validation
│   │   │   │   ├── images
│   │   │   │   │   ├── training
│   │   │   │   │   ├── validation

```

## Testing
```bash
./tools/dist_test.sh configs/sem_fpn/fpn_lsnet_b_ade20k_40k.py pretrain/lsnet_b_semfpn.pth 8 --eval mIoU
```

## Training 
Download ImageNet-1K pretrained weights into `./pretrain` 
```bash
./tools/dist_train.sh configs/sem_fpn/fpn_lsnet_b_ade20k_40k.py 8 --seed 0 --deterministic
```

