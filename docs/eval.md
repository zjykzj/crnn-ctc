# EVAL

## EMNIST

```shell
$ CUDA_VISIBLE_DEVICES=0 python eval_emnist.py runs/crnn_tiny-emnist-b512-e100.pth ../datasets/emnist/
args: Namespace(not_tiny=False, pretrained='runs/crnn_tiny-emnist-b512-e100.pth', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: runs/crnn_tiny-emnist-b512-e100.pth
Batch:1562 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [00:22<00:00, 71.04it/s]
ACC:98.396
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python eval_emnist.py runs/crnn-emnist-b512-e100.pth ../datasets/emnist/ --not-tiny
args: Namespace(not_tiny=True, pretrained='runs/crnn-emnist-b512-e100.pth', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: runs/crnn-emnist-b512-e100.pth
Batch:1562 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [00:22<00:00, 69.08it/s]
ACC:98.546
```

## License Plate

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py ./runs/crnn_tiny-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/
args: Namespace(not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='./runs/crnn_tiny-plate-b512-e100.pth', use_lstm=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: ./runs/crnn_tiny-plate-b512-e100.pth
Load test data: 149002
Batch:4656 ACC:90.000: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 4657/4657 [00:52<00:00, 89.05it/s]
ACC:76.222
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py ./runs/crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny
args: Namespace(not_tiny=True, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='./runs/crnn-plate-b512-e100.pth', use_lstm=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: ./runs/crnn-plate-b512-e100.pth
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 4657/4657 [00:56<00:00, 83.05it/s]
ACC:82.379
```

## Specify Which Dataset to Eval

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py ./runs/crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-ccpd2019
args: Namespace(not_tiny=True, only_ccpd2019=True, only_ccpd2020=False, only_others=False, pretrained='./runs/crnn-plate-b512-e100.pth', use_lstm=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: ./runs/crnn-plate-b512-e100.pth
Load test data: 141982
Batch:4436 ACC:66.667: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 4437/4437 [00:51<00:00, 86.65it/s]
ACC:81.757
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py ./runs/crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-ccpd2020
args: Namespace(not_tiny=True, only_ccpd2019=False, only_ccpd2020=True, only_others=False, pretrained='./runs/crnn-plate-b512-e100.pth', use_lstm=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: ./runs/crnn-plate-b512-e100.pth
Load test data: 5006
Batch:156 ACC:85.714: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:03<00:00, 39.36it/s]
ACC:93.708
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py ./runs/crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny --only-others
args: Namespace(not_tiny=True, only_ccpd2019=False, only_ccpd2020=False, only_others=True, pretrained='./runs/crnn-plate-b512-e100.pth', use_lstm=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: ./runs/crnn-plate-b512-e100.pth
Load test data: 2014
Batch:62 ACC:100.000: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:03<00:00, 20.97it/s]
ACC:98.113
```