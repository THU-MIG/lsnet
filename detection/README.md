# Object Detection and Instance Segmentation

Detection and instance segmentation on MS COCO 2017 is implemented based on [MMDetection](https://github.com/open-mmlab/mmdetection).

## Models
Results with RetinaNet
| Model | $AP$ | $AP_{50}$ | $AP_{75}$ | $AP_S$ | $AP_M$ | $AP_L$ | Log |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [LSNet-T](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_t_retinanet.pth) | 34.2 | 54.6 | 35.2 | 17.8 | 37.1 | 48.5 | [lsnet_t_retinanet.json](./logs/lsnet_t_retinanet.json) |
| [LSNet-S](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_s_retinanet.pth) | 36.5 | 57.3 | 38.1 | 20.3 | 39.5 | 51.0 | [lsnet_s_retinanet.json](./logs/lsnet_s_retinanet.json) |
| [LSNet-B](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_b_retinanet.pth) | 39.2 | 60.0 | 41.5 | 22.1 | 43.0 | 52.9 | [lsnet_b_retinanet.json](./logs/lsnet_b_retinanet.json) |

Results with MaskR-CNN
| Model | $AP^b$ | $AP_{50}^b$ | $AP_{75}^b$ | $AP^m$ | $AP_{50}^m$ | $AP_{75}^m$ | Log |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [LSNet-T](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_t_maskrcnn.pth) | 35.0 | 57.0 | 37.3 | 32.7 | 53.8 | 34.3 | [lsnet_t_maskrcnn.json](./logs/lsnet_t_maskrcnn.json)  |
| [LSNet-S](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_s_maskrcnn.pth) | 37.4 | 59.9 | 39.8 | 34.8 | 56.8 | 36.6 | [lsnet_s_maskrcnn.json](./logs/lsnet_s_maskrcnn.json)  |
| [LSNet-B](https://huggingface.co/jameslahm/lsnet/blob/main/lsnet_b_maskrcnn.pth) | 40.8 | 63.4 | 44.0 | 37.8 | 60.5 | 40.1 | [lsnet_b_maskrcnn.json](./logs/lsnet_b_maskrcnn.json)  |

## Installation
```bash
pip install mmcv-full==1.7.2
pip install mmdet==2.28.2
# Please replace line 160 in anaconda3/envs/seg/lib/python3.10/site-packages/mmcv/parallel/distributed.py to module_to_run = self.module
# Please patch mmcv following https://github.com/HarborYuan/mmcv_16/commit/ad1a72fe0cbeead2716706ff618dfa0269d2cf4c
```

## Data preparation

Please prepare COCO 2017 dataset according to the [instructions in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).
The dataset should be organized as 
```
detection
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

## Testing
For RetinaNet
```bash
bash ./dist_test.sh configs/retinanet_lsnet_b_fpn_1x_coco.py pretrain/lsnet_b_retinanet.pth 8 --eval bbox --out results.pkl
```
For Mask R-CNN
```bash
bash ./dist_test.sh configs/mask_rcnn_lsnet_b_fpn_1x_coco.py pretrain/lsnet_b_maskrcnn.pth 8 --eval bbox segm --out results.pkl
```

## Training
Download ImageNet-1K pretrained weights into `./pretrain` 
For RetinaNet
```bash
bash ./dist_train.sh configs/retinanet_lsnet_b_fpn_1x_coco.py 8
```
For Mask R-CNN
```bash
bash ./dist_train.sh configs/mask_rcnn_lsnet_b_fpn_1x_coco.py 8
```
