# PREDICT

## EMNIST

```shell
$ python3 predict_emnist.py runs/crnn_lstm-emnist-b512-e100.pth ../datasets/emnist/ ./runs/predict/emnist/
args: Namespace(pretrained='runs/crnn_lstm-emnist-b512-e100.pth', save_dir='./runs/predict/emnist/', use_gru=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: runs/crnn_lstm-emnist-b512-e100.pth
Label: [5 2 2 7 9] Pred: [5 2 2 7 9]
Label: [5 7 1 4 6] Pred: [5 7 1 4 6]
Label: [4 2 9 7 0] Pred: [4 2 9 7 0]
Label: [1 9 2 9 2] Pred: [1 9 2 9 2]
Label: [4 9 9 8 0] Pred: [4 9 9 8 0]
Label: [7 5 7 9 8] Pred: [7 5 7 9 8]
```

```shell
$ python3 predict_emnist.py runs/crnn_gru-emnist-b512-e100.pth ../datasets/emnist/ ./runs/predict/emnist/ --use-gru
args: Namespace(pretrained='runs/crnn_gru-emnist-b512-e100.pth', save_dir='./runs/predict/emnist/', use_gru=True, val_root='../datasets/emnist/')
Loading CRNN pretrained: runs/crnn_gru-emnist-b512-e100.pth
Label: [3 8 6 3 1] Pred: [3 8 6 3 1]
Label: [7 7 3 8 5] Pred: [7 7 3 8 5]
Label: [6 3 4 9 7] Pred: [6 3 4 9 7]
Label: [5 1 4 9 5] Pred: [5 1 4 9 5]
Label: [6 8 4 1 9] Pred: [6 8 4 1 9]
Label: [9 1 7 3 7] Pred: [9 1 7 3 7]
```

## License Plate

```shell
python predict_plate.py runs/crnn_lstm-plate-b256-e100.pth ./assets/plate/宁A87J92_0.jpg ./runs/predict/plate/
args: Namespace(image_path='./assets/plate/宁A87J92_0.jpg', pretrained='runs/crnn_lstm-plate-b256-e100.pth', save_dir='./runs/predict/plate/', use_gru=False)
Loading CRNN pretrained: runs/crnn_lstm-plate-b256-e100.pth
Pred: 宁A·87J92
Save to ./runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ python predict_plate.py runs/crnn_gru-plate-b256-e100.pth ./assets/plate/宁A87J92_0.jpg ./runs/predict/plate/ --use-gru
args: Namespace(image_path='./assets/plate/宁A87J92_0.jpg', pretrained='runs/crnn_gru-plate-b256-e100.pth', save_dir='./runs/predict/plate/', use_gru=True)
Loading CRNN pretrained: runs/crnn_gru-plate-b256-e100.pth
Pred: 宁A·87J92
Save to ./runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ python predict_plate.py runs/crnn_lstm-plate-b256-e100.pth ./assets/plate/川A3X7J1_0.jpg ./runs/predict/plate/
args: Namespace(image_path='./assets/plate/川A3X7J1_0.jpg', pretrained='runs/crnn_lstm-plate-b256-e100.pth', save_dir='./runs/predict/plate/', use_gru=False)
Loading CRNN pretrained: runs/crnn_lstm-plate-b256-e100.pth
Pred: 川A·3X7J1
Save to ./runs/predict/plate/plate_川A3X7J1_0.jpg
```

```shell
$ python predict_plate.py runs/crnn_gru-plate-b256-e100.pth ./assets/plate/川A3X7J1_0.jpg ./runs/predict/plate/ --use-gru
args: Namespace(image_path='./assets/plate/川A3X7J1_0.jpg', pretrained='runs/crnn_gru-plate-b256-e100.pth', save_dir='./runs/predict/plate/', use_gru=True)
Loading CRNN pretrained: runs/crnn_gru-plate-b256-e100.pth
Pred: 川A·3X7J1
Save to ./runs/predict/plate/plate_川A3X7J1_0.jpg
```