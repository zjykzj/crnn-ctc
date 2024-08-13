# EVAL

## EMNIST

```shell
$ CUDA_VISIBLE_DEVICES=0 python eval_emnist.py crnn_tiny-emnist-b512-e100.pth ../datasets/emnist/
args: Namespace(not_tiny=False, pretrained='crnn_tiny-emnist-b512-e100.pth', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn_tiny-emnist-b512-e100.pth
crnn_tiny-emnist-b512-e100 summary: 22 layers, 427467 parameters, 427467 gradients, 0.1 GFLOPs
Batch:1562 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [00:19<00:00, 80.29it/s]
ACC:98.278
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python eval_emnist.py crnn-emnist-b512-e100.pth ../datasets/emnist/ --not-tiny
args: Namespace(not_tiny=True, pretrained='crnn-emnist-b512-e100.pth', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn-emnist-b512-e100.pth
crnn-emnist-b512-e100 summary: 29 layers, 7924363 parameters, 7924363 gradients, 2.2 GFLOPs
Batch:1562 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [00:17<00:00, 86.84it/s]
ACC:98.718
```

## License Plate

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn_tiny-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/
args: Namespace(not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='crnn_tiny-plate-b512-e100.pth', use_lstm=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn_tiny-plate-b512-e100.pth
crnn_tiny-plate-b512-e100 summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Load test data: 149002
Batch:4656 ACC:90.000: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4657/4657 [01:08<00:00, 67.50it/s]
ACC:76.226
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny
args: Namespace(not_tiny=True, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='crnn-plate-b512-e100.pth', use_lstm=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn-plate-b512-e100.pth
crnn-plate-b512-e100 summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4657/4657 [00:54<00:00, 85.01it/s]
ACC:82.384
```

## Specify Which Dataset to Eval

### CRNN_Tiny

### CRNN

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-ccpd2019
args: Namespace(not_tiny=True, only_ccpd2019=True, only_ccpd2020=False, only_others=False, pretrained='crnn-plate-b512-e100.pth', use_lstm=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn-plate-b512-e100.pth
crnn-plate-b512-e100 summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Load test data: 141982
Batch:4436 ACC:83.333: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4437/4437 [00:49<00:00, 89.24it/s]
ACC:81.761
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-ccpd2020
args: Namespace(not_tiny=True, only_ccpd2019=False, only_ccpd2020=True, only_others=False, pretrained='crnn-plate-b512-e100.pth', use_lstm=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn-plate-b512-e100.pth
crnn-plate-b512-e100 summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Load test data: 5006
Batch:156 ACC:92.857: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 70.23it/s]
ACC:93.728
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-others
args: Namespace(not_tiny=True, only_ccpd2019=False, only_ccpd2020=False, only_others=True, pretrained='crnn-plate-b512-e100.pth', use_lstm=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn-plate-b512-e100.pth
crnn-plate-b512-e100 summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Load test data: 2014
Batch:62 ACC:100.000: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:01<00:00, 54.85it/s]
ACC:98.113
```