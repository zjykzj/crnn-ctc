# EVAL

## EMNIST

```shell
$ CUDA_VISIBLE_DEVICES=0 python eval_emnist.py crnn_tiny-emnist.pth ../datasets/emnist/
args: Namespace(not_tiny=False, pretrained='crnn_tiny-emnist.pth', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn_tiny-emnist.pth
crnn_tiny-emnist summary: 22 layers, 427467 parameters, 427467 gradients, 0.1 GFLOPs
Batch:49999 ACC:100.000: 100%|████████████████████████████████████████████████████████| 50000/50000 [02:41<00:00, 310.01it/s]
ACC:98.306
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python eval_emnist.py crnn-emnist.pth ../datasets/emnist/ --not-tiny
args: Namespace(not_tiny=True, pretrained='crnn-emnist.pth', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn-emnist.pth
crnn-emnist summary: 29 layers, 7924363 parameters, 7924363 gradients, 2.2 GFLOPs
Batch:49999 ACC:100.000: 100%|████████████████████████████████████████████████████████| 50000/50000 [03:47<00:00, 219.75it/s]
ACC:98.570
```

## License Plate

### CRNN/CRNN_Tiny

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn_tiny-plate.pth ../datasets/chinese_license_plate/recog/
args: Namespace(add_stnet=False, not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='crnn_tiny-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn_tiny-plate.pth
crnn_tiny-plate summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████| 4657/4657 [00:33<00:00, 138.66it/s]
ACC:76.590
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn-plate.pth ../datasets/chinese_license_plate/recog/ --not-tiny
args: Namespace(add_stnet=False, not_tiny=True, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='crnn-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn-plate.pth
crnn-plate summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Load test data: 149002
Batch:4656 ACC:100.000: 100%|████████████████████████████████████████████████████████████| 4657/4657 [00:52<00:00, 89.13it/s]
ACC:82.147
```

### LPRNet/LPRNetPlus

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_plus-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet
args: Namespace(add_stnet=False, not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='lprnet_plus-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_plus-plate.pth
lprnet_plus-plate summary: 57 layers, 576988 parameters, 576988 gradients, 0.5 GFLOPs
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████| 4657/4657 [00:31<00:00, 147.67it/s]
ACC:63.546
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --use-origin-block
args: Namespace(add_stnet=False, not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='lprnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet-plate.pth
lprnet-plate summary: 51 layers, 486236 parameters, 486236 gradients, 0.3 GFLOPs
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████| 4657/4657 [00:29<00:00, 159.63it/s]
ACC:60.105
```

### LPRNet/LPRNetPlus+STNet

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_plus_stnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --add-stnet
args: Namespace(add_stnet=True, not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='lprnet_plus_stnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_plus_stnet-plate.pth
/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:4223: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
lprnet_plus_stnet-plate summary: 69 layers, 632418 parameters, 632418 gradients, 0.5 GFLOPs
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████| 4657/4657 [00:32<00:00, 141.28it/s]
ACC:72.130
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_stnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --use-origin-block --add-stnet
args: Namespace(add_stnet=True, not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='lprnet_stnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_stnet-plate.pth
/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:4223: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
lprnet_stnet-plate summary: 63 layers, 541666 parameters, 541666 gradients, 0.3 GFLOPs
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████| 4657/4657 [00:31<00:00, 149.95it/s]
ACC:72.261
```

## Specify Which Dataset to Eval

### CRNN_Tiny

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn_tiny-plate.pth ../datasets/chinese_license_plate/recog/ --only-ccpd2019
args: Namespace(add_stnet=False, not_tiny=False, only_ccpd2019=True, only_ccpd2020=False, only_others=False, pretrained='crnn_tiny-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn_tiny-plate.pth
crnn_tiny-plate summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Load test data: 141982
Batch:4436 ACC:53.333: 100%|████████████████████████████████████████████████████████████| 4437/4437 [00:32<00:00, 135.92it/s]
ACC:75.729
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn_tiny-plate.pth ../datasets/chinese_license_plate/recog/ --only-ccpd2020
args: Namespace(add_stnet=False, not_tiny=False, only_ccpd2019=False, only_ccpd2020=True, only_others=False, pretrained='crnn_tiny-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn_tiny-plate.pth
crnn_tiny-plate summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Load test data: 5006
Batch:156 ACC:92.857: 100%|████████████████████████████████████████████████████████████████| 157/157 [00:01<00:00, 95.36it/s]
ACC:92.829
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn_tiny-plate.pth ../datasets/chinese_license_plate/recog/ --only-others
args: Namespace(add_stnet=False, not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=True, pretrained='crnn_tiny-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn_tiny-plate.pth
crnn_tiny-plate summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Load test data: 2014
Batch:62 ACC:100.000: 100%|██████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 71.62it/s]
ACC:96.922
```

### CRNN

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn-plate.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-ccpd2019
args: Namespace(add_stnet=False, not_tiny=True, only_ccpd2019=True, only_ccpd2020=False, only_others=False, pretrained='crnn-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn-plate.pth
crnn-plate summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Load test data: 141982
Batch:4436 ACC:80.000: 100%|█████████████████████████████████████████████████████████████| 4437/4437 [00:49<00:00, 90.04it/s]
ACC:81.512
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn-plate.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-ccpd2020
args: Namespace(add_stnet=False, not_tiny=True, only_ccpd2019=False, only_ccpd2020=True, only_others=False, pretrained='crnn-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn-plate.pth
crnn-plate summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Load test data: 5006
Batch:156 ACC:92.857: 100%|████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 69.77it/s]
ACC:93.787
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn-plate.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-others
args: Namespace(add_stnet=False, not_tiny=True, only_ccpd2019=False, only_ccpd2020=False, only_others=True, pretrained='crnn-plate.pth', use_lprnet=False, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn-plate.pth
crnn-plate summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Load test data: 2014
Batch:62 ACC:96.667: 100%|███████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 53.53it/s]
ACC:98.014
```

### LPRNetPlus

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_plus-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --only-ccpd2019
args: Namespace(add_stnet=False, not_tiny=False, only_ccpd2019=True, only_ccpd2020=False, only_others=False, pretrained='lprnet_plus-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_plus-plate.pth
lprnet_plus-plate summary: 57 layers, 576988 parameters, 576988 gradients, 0.5 GFLOPs
Load test data: 141982
Batch:4436 ACC:46.667: 100%|████████████████████████████████████████████████████████████| 4437/4437 [00:29<00:00, 149.64it/s]
ACC:62.184
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_plus-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --only-ccpd2020
args: Namespace(add_stnet=False, not_tiny=False, only_ccpd2019=False, only_ccpd2020=True, only_others=False, pretrained='lprnet_plus-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_plus-plate.pth
lprnet_plus-plate summary: 57 layers, 576988 parameters, 576988 gradients, 0.5 GFLOPs
Load test data: 5006
Batch:156 ACC:78.571: 100%|████████████████████████████████████████████████████████████████| 157/157 [00:01<00:00, 99.20it/s]
ACC:89.373
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_plus-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --only-others
args: Namespace(add_stnet=False, not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=True, pretrained='lprnet_plus-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_plus-plate.pth
lprnet_plus-plate summary: 57 layers, 576988 parameters, 576988 gradients, 0.5 GFLOPs
Load test data: 2014
Batch:62 ACC:93.333: 100%|███████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 69.70it/s]
ACC:95.233
```

### LPRNet

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --use-origin-block --only-ccpd2019
args: Namespace(add_stnet=False, not_tiny=False, only_ccpd2019=True, only_ccpd2020=False, only_others=False, pretrained='lprnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet-plate.pth
lprnet-plate summary: 51 layers, 486236 parameters, 486236 gradients, 0.3 GFLOPs
Load test data: 141982
Batch:4436 ACC:46.667: 100%|████████████████████████████████████████████████████████████| 4437/4437 [00:27<00:00, 158.81it/s]
ACC:58.597
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --use-origin-block --only-ccpd2020
args: Namespace(add_stnet=False, not_tiny=False, only_ccpd2019=False, only_ccpd2020=True, only_others=False, pretrained='lprnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet-plate.pth
lprnet-plate summary: 51 layers, 486236 parameters, 486236 gradients, 0.3 GFLOPs
Load test data: 5006
Batch:156 ACC:85.714: 100%|███████████████████████████████████████████████████████████████| 157/157 [00:01<00:00, 108.01it/s]
ACC:89.153
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --use-origin-block --only-others
args: Namespace(add_stnet=False, not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=True, pretrained='lprnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet-plate.pth
lprnet-plate summary: 51 layers, 486236 parameters, 486236 gradients, 0.3 GFLOPs
Load test data: 2014
Batch:62 ACC:96.667: 100%|███████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 75.99it/s]
ACC:94.091
```

### LPRNetPlus+STNet

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_plus_stnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --add-stnet --only-ccpd2019
args: Namespace(add_stnet=True, not_tiny=False, only_ccpd2019=True, only_ccpd2020=False, only_others=False, pretrained='lprnet_plus_stnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_plus_stnet-plate.pth
/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:4223: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
lprnet_plus_stnet-plate summary: 69 layers, 632418 parameters, 632418 gradients, 0.5 GFLOPs
Load test data: 141982
Batch:4436 ACC:63.333: 100%|████████████████████████████████████████████████████████████| 4437/4437 [00:31<00:00, 139.76it/s]
ACC:71.125
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_plus_stnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --add-stnet --only-ccpd2020
args: Namespace(add_stnet=True, not_tiny=False, only_ccpd2019=False, only_ccpd2020=True, only_others=False, pretrained='lprnet_plus_stnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_plus_stnet-plate.pth
/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:4223: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
lprnet_plus_stnet-plate summary: 69 layers, 632418 parameters, 632418 gradients, 0.5 GFLOPs
Load test data: 5006
Batch:156 ACC:78.571: 100%|████████████████████████████████████████████████████████████████| 157/157 [00:01<00:00, 95.63it/s]
ACC:90.611
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_plus_stnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --add-stnet --only-others
args: Namespace(add_stnet=True, not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=True, pretrained='lprnet_plus_stnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_plus_stnet-plate.pth
/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:4223: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
lprnet_plus_stnet-plate summary: 69 layers, 632418 parameters, 632418 gradients, 0.5 GFLOPs
Load test data: 2014
Batch:62 ACC:100.000: 100%|██████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 64.06it/s]
ACC:97.071
```

### LPRNet+STNet

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_stnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --use-origin-block --add-stnet --only-ccpd2019
args: Namespace(add_stnet=True, not_tiny=False, only_ccpd2019=True, only_ccpd2020=False, only_others=False, pretrained='lprnet_stnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_stnet-plate.pth
/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:4223: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
lprnet_stnet-plate summary: 63 layers, 541666 parameters, 541666 gradients, 0.3 GFLOPs
Load test data: 141982
Batch:4436 ACC:63.333: 100%|████████████████████████████████████████████████████████████| 4437/4437 [00:29<00:00, 150.29it/s]
ACC:71.291
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_stnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --use-origin-block --add-stnet --only-ccpd2020
args: Namespace(add_stnet=True, not_tiny=False, only_ccpd2019=False, only_ccpd2020=True, only_others=False, pretrained='lprnet_stnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_stnet-plate.pth
/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:4223: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
lprnet_stnet-plate summary: 63 layers, 541666 parameters, 541666 gradients, 0.3 GFLOPs
Load test data: 5006
Batch:156 ACC:64.286: 100%|███████████████████████████████████████████████████████████████| 157/157 [00:01<00:00, 103.83it/s]
ACC:89.832
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py lprnet_stnet-plate.pth ../datasets/chinese_license_plate/recog/ --use-lprnet --use-origin-block --add-stnet --only-others
args: Namespace(add_stnet=True, not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=True, pretrained='lprnet_stnet-plate.pth', use_lprnet=True, use_lstm=False, use_origin_block=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: lprnet_stnet-plate.pth
/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:4223: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
lprnet_stnet-plate summary: 63 layers, 541666 parameters, 541666 gradients, 0.3 GFLOPs
Load test data: 2014
Batch:62 ACC:96.667: 100%|███████████████████████████████████████████████████████████████████| 63/63 [00:00<00:00, 65.71it/s]
ACC:96.773
```