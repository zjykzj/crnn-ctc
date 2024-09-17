# EVAL

## EMNIST

```shell
$ CUDA_VISIBLE_DEVICES=0 python eval_emnist.py crnn_tiny-emnist.pth ../datasets/emnist/
args: Namespace(not_tiny=False, pretrained='crnn_tiny-emnist.pth', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn_tiny-emnist.pth
crnn_tiny-emnist summary: 22 layers, 427467 parameters, 427467 gradients, 0.1 GFLOPs
Batch:49999 ACC:100.000: 100%|████████████████████████████████████████████████████████| 50000/50000 [02:42<00:00, 308.44it/s]
ACC:98.482
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python eval_emnist.py crnn-emnist.pth ../datasets/emnist/ --not-tiny
args: Namespace(not_tiny=True, pretrained='crnn-emnist.pth', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn-emnist.pth
crnn-emnist summary: 29 layers, 7924363 parameters, 7924363 gradients, 2.2 GFLOPs
Batch:49999 ACC:100.000: 100%|████████████████████████████████████████████████████████| 50000/50000 [03:53<00:00, 213.84it/s]
ACC:98.640
```

## License Plate

### CRNN/CRNN_Tiny

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn_tiny-plate.pth ../datasets/chinese_license_plate/recog/
args: Namespace(not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='crnn_tiny-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn_tiny-plate.pth
crnn_tiny-plate summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████| 4657/4657 [00:36<00:00, 125.99it/s]
ACC:76.590
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn-plate.pth ../datasets/chinese_license_plate/recog/ --not-tiny
args: Namespace(not_tiny=True, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='crnn-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn-plate.pth
crnn-plate summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Load test data: 149002
Batch:4656 ACC:90.000: 100%|█████████████████████████████████████████████████████████████| 4657/4657 [00:52<00:00, 89.33it/s]
ACC:82.311
```

### LPRNet/LPRNetPlus

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_plus-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet
args: Namespace(not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='lprnet_plus-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_plus-plate.pth
lprnet_plus-plate summary: 57 layers, 576988 parameters, 576988 gradients, 0.5 GFLOPs
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████| 4657/4657 [00:31<00:00, 147.36it/s]
ACC:63.449
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --use-origin-block
args: Namespace(not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='lprnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet-plate.pth
lprnet-plate summary: 51 layers, 486236 parameters, 486236 gradients, 0.3 GFLOPs
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████| 4657/4657 [00:29<00:00, 157.88it/s]
ACC:61.096
```

## Specify Which Dataset to Eval

### CRNN_Tiny

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn_tiny-plate.pth ../datasets/chinese_license_plate/recog/ --only-ccpd2019
args: Namespace(not_tiny=False, only_ccpd2019=True, only_ccpd2020=False, only_others=False, pretrained='crnn_tiny-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn_tiny-plate.pth
crnn_tiny-plate summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Load test data: 141982
Batch:4436 ACC:53.333: 100%|████████████████████████████████████████████████████████████| 4437/4437 [00:33<00:00, 132.54it/s]
ACC:75.729
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn_tiny-plate.pth ../datasets/chinese_license_plate/recog/ --only-ccpd2020
args: Namespace(not_tiny=False, only_ccpd2019=False, only_ccpd2020=True, only_others=False, pretrained='crnn_tiny-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn_tiny-plate.pth
crnn_tiny-plate summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Load test data: 5006
Batch:156 ACC:92.857: 100%|████████████████████████████████████████████████████████████████| 157/157 [00:01<00:00, 94.02it/s]
ACC:92.829
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn_tiny-plate.pth ../datasets/chinese_license_plate/recog/ --only-others
args: Namespace(not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=True, pretrained='crnn_tiny-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn_tiny-plate.pth
crnn_tiny-plate summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Load test data: 2014
Batch:62 ACC:100.000: 100%|██████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 66.81it/s]
ACC:96.922
```

### CRNN

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn-plate.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-ccpd2019
args: Namespace(not_tiny=True, only_ccpd2019=True, only_ccpd2020=False, only_others=False, pretrained='crnn-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn-plate.pth
crnn-plate summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Load test data: 141982
Batch:4436 ACC:73.333: 100%|█████████████████████████████████████████████████████████████| 4437/4437 [00:49<00:00, 89.59it/s]
ACC:81.698
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn-plate.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-ccpd2020
args: Namespace(not_tiny=True, only_ccpd2019=False, only_ccpd2020=True, only_others=False, pretrained='crnn-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn-plate.pth
crnn-plate summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Load test data: 5006
Batch:156 ACC:85.714: 100%|████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 69.73it/s]
ACC:93.548
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn-plate.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-others
args: Namespace(not_tiny=True, only_ccpd2019=False, only_ccpd2020=False, only_others=True, pretrained='crnn-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn-plate.pth
crnn-plate summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Load test data: 2014
Batch:62 ACC:96.667: 100%|███████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 50.62it/s]
ACC:97.617
```

### LPRNetPlus

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_plus-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --only-ccpd2019
args: Namespace(not_tiny=False, only_ccpd2019=True, only_ccpd2020=False, only_others=False, pretrained='lprnet_plus-plate.pth',use_lprnet=True, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_plus-plate.pth
lprnet_plus-plate summary: 57 layers, 576988 parameters, 576988 gradients, 0.5 GFLOPs
Load test data: 141982
Batch:4436 ACC:40.000: 100%|████████████████████████████████████████████████████████████| 4437/4437 [00:29<00:00, 148.36it/s]
ACC:62.080
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_plus-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --only-ccpd2020
args: Namespace(not_tiny=False, only_ccpd2019=False, only_ccpd2020=True, only_others=False, pretrained='lprnet_plus-plate.pth',use_lprnet=True, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_plus-plate.pth
lprnet_plus-plate summary: 57 layers, 576988 parameters, 576988 gradients, 0.5 GFLOPs
Load test data: 5006
Batch:156 ACC:78.571: 100%|████████████████████████████████████████████████████████████████| 157/157 [00:01<00:00, 98.43it/s]
ACC:89.213
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_plus-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --only-others
args: Namespace(not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=True, pretrained='lprnet_plus-plate.pth',use_lprnet=True, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_plus-plate.pth
lprnet_plus-plate summary: 57 layers, 576988 parameters, 576988 gradients, 0.5 GFLOPs
Load test data: 2014
Batch:62 ACC:96.667: 100%|███████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 68.93it/s]
ACC:95.631
```

### LPRNet

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --use-origin-block --only-ccpd2019
args: Namespace(not_tiny=False, only_ccpd2019=True, only_ccpd2020=False, only_others=False, pretrained='lprnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet-plate.pth
lprnet-plate summary: 51 layers, 486236 parameters, 486236 gradients, 0.3 GFLOPs
Load test data: 141982
Batch:4436 ACC:40.000: 100%|████████████████████████████████████████████████████████████| 4437/4437 [00:28<00:00, 157.58it/s]
ACC:59.686
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --use-origin-block --only-ccpd2020
args: Namespace(not_tiny=False, only_ccpd2019=False, only_ccpd2020=True, only_others=False, pretrained='lprnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet-plate.pth
lprnet-plate summary: 51 layers, 486236 parameters, 486236 gradients, 0.3 GFLOPs
Load test data: 5006
Batch:156 ACC:71.429: 100%|███████████████████████████████████████████████████████████████| 157/157 [00:01<00:00, 101.93it/s]
ACC:87.335
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --use-origin-block --only-others
args: Namespace(not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=True, pretrained='lprnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet-plate.pth
lprnet-plate summary: 51 layers, 486236 parameters, 486236 gradients, 0.3 GFLOPs
Load test data: 2014
Batch:62 ACC:96.667: 100%|███████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 72.31it/s]
ACC:95.283
```