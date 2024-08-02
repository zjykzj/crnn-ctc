# EVAL

## EMNIST

```shell
$ python3 eval_emnist.py ./runs/crnn_gru-emnist-b512/crnn_gru-emnist-b512-e100.pth ../datasets/emnist/ --use-gru
args: Namespace(pretrained='./runs/crnn_gru-emnist-b512/crnn_gru-emnist-b512-e100.pth', use_gru=True, val_root='../datasets/emnist/')
Loading CRNN pretrained: ./runs/crnn_gru-emnist-b512/crnn_gru-emnist-b512-e100.pth
Batch:1562 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [00:38<00:00, 40.34it/s]
ACC:98.432
```

```shell
$ python3 eval_emnist.py ./runs/crnn_lstm-emnist-b512/crnn_lstm-emnist-b512-e100.pth ../datasets/emnist/
args: Namespace(pretrained='./runs/crnn_lstm-emnist-b512/crnn_lstm-emnist-b512-e100.pth', use_gru=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: ./runs/crnn_lstm-emnist-b512/crnn_lstm-emnist-b512-e100.pth
Batch:1562 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [00:39<00:00, 39.67it/s]
ACC:98.386
```

## License Plate

```shell
$ python3 eval_plate.py ./runs/crnn_lstm-plate-b256-e100.pth ../datasets/chinese_license_plate/recog/
args: Namespace(only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='./runs/crnn_lstm-plate-b256-e100.pth', use_gru=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: ./runs/crnn_lstm-plate-b256-e100.pth
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4657/4657 [08:19<00:00,  9.33it/s]
ACC:74.252
```

```shell
$ CUDA_VISIBLE_DEVICES=1 python3 eval_plate.py ./runs/crnn_gru-plate-b256-e100.pth ../datasets/chinese_license_plate/recog/ --use-gru
args: Namespace(only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='./runs/crnn_gru-plate-b256-e100.pth', use_gru=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: ./runs/crnn_gru-plate-b256-e100.pth
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4657/4657 [07:27<00:00, 10.41it/s]
ACC:75.649
```

## Specify Which Dataset to Eval

```shell
$ python3 eval_plate.py ./runs/crnn_gru-plate-b256-e100.pth ../datasets/chinese_license_plate/recog/ --use-gru --only-ccpd2019
args: Namespace(only_ccpd2019=True, only_ccpd2020=False, only_others=False, pretrained='./runs/crnn_gru-plate-b256-e100.pth', use_gru=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: ./runs/crnn_gru-plate-b256-e100.pth
Load test data: 141982
Batch:4436 ACC:80.000: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4437/4437 [02:02<00:00, 36.12it/s]
ACC:74.743
```

```shell
$ CUDA_VISIBLE_DEVICES=1 python3 eval_plate.py ./runs/crnn_gru-plate-b256-e100.pth ../datasets/chinese_license_plate/recog/ --use-gru --only-ccpd2020
args: Namespace(only_ccpd2019=False, only_ccpd2020=True, only_others=False, pretrained='./runs/crnn_gru-plate-b256-e100.pth', use_gru=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: ./runs/crnn_gru-plate-b256-e100.pth
Load test data: 5006
Batch:156 ACC:92.857: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 28.41it/s]
ACC:92.369
```

```shell
$ CUDA_VISIBLE_DEVICES=1 python3 eval_plate.py ./runs/crnn_gru-plate-b256-e100.pth ../datasets/chinese_license_plate/recog/ --use-gru --only-others
args: Namespace(only_ccpd2019=False, only_ccpd2020=False, only_others=True, pretrained='./runs/crnn_gru-plate-b256-e100.pth', use_gru=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: ./runs/crnn_gru-plate-b256-e100.pth
Load test data: 2014
Batch:62 ACC:96.667: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:02<00:00, 28.52it/s]
ACC:97.865
```