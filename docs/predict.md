# PREDICT

## EMNIST

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_emnist.py runs/crnn_tiny-emnist-b512-e100.pth ../datasets/emnist/ ./runs/predict/emnist/
args: Namespace(not_tiny=False, pretrained='runs/crnn_tiny-emnist-b512-e100.pth', save_dir='./runs/predict/emnist/', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: runs/crnn_tiny-emnist-b512-e100.pth
Label: [8 8 1 1 5] Pred: [8 8 1 1 5]
Label: [4 8 4 7 2] Pred: [4 8 4 7 2]
Label: [7 4 4 4 5] Pred: [7 4 4 4 5]
Label: [9 2 1 5 2] Pred: [9 2 1 5 2]
Label: [3 0 3 2 6] Pred: [3 0 3 2 6]
Label: [9 1 0 3 2] Pred: [9 1 0 3 2]
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_emnist.py runs/crnn-emnist-b512-e100.pth ../datasets/emnist/ ./runs/predict/emnist/ --not-tiny
args: Namespace(not_tiny=True, pretrained='runs/crnn-emnist-b512-e100.pth', save_dir='./runs/predict/emnist/', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: runs/crnn-emnist-b512-e100.pth
Label: [4 5 9 0 1] Pred: [4 5 9 0 1]
Label: [5 9 5 6 1] Pred: [5 9 5 6 1]
Label: [1 6 0 8 3] Pred: [1 6 0 8 3]
Label: [9 1 7 2 4] Pred: [9 1 7 2 4]
Label: [2 8 9 6 3] Pred: [2 8 9 6 3]
Label: [0 6 9 8 2] Pred: [0 6 9 8 2]
```

## License Plate

```shell
$ python predict_plate.py ./runs/crnn_tiny-plate-b512-e100.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/
args: Namespace(image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=False, pretrained='./runs/crnn_tiny-plate-b512-e100.pth', save_dir='runs/predict/plate/', use_lstm=False)
Loading CRNN pretrained: ./runs/crnn_tiny-plate-b512-e100.pth
Pred: 宁A·87J92
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ python predict_plate.py ./runs/crnn-plate-b512-e100.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --not-tiny
args: Namespace(image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=True, pretrained='./runs/crnn-plate-b512-e100.pth', save_dir='runs/predict/plate/', use_lstm=False)
Loading CRNN pretrained: ./runs/crnn-plate-b512-e100.pth
Pred: 宁A·87J92
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ python predict_plate.py ./runs/crnn-plate-b512-e100.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/ --not-tiny
args: Namespace(image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=True, pretrained='./runs/crnn-plate-b512-e100.pth', save_dir='runs/predict/plate/', use_lstm=False)
Loading CRNN pretrained: ./runs/crnn-plate-b512-e100.pth
Pred: 川A·3X7J1
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```

```shell
$ python predict_plate.py ./runs/crnn-plate-b512-e100.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/ --not-tiny
args: Namespace(image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=True, pretrained='./runs/crnn-plate-b512-e100.pth', save_dir='runs/predict/plate/', use_lstm=False)
Loading CRNN pretrained: ./runs/crnn-plate-b512-e100.pth
Pred: 川A·3X7J1
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```