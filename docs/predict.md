# PREDICT

## EMNIST

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_emnist.py crnn_tiny-emnist-b512-e100.pth ../datasets/emnist/ ./runs/predict/emnist/
args: Namespace(not_tiny=False, pretrained='crnn_tiny-emnist-b512-e100.pth', save_dir='./runs/predict/emnist/', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn_tiny-emnist-b512-e100.pth
crnn_tiny-emnist-b512-e100 summary: 22 layers, 427467 parameters, 427467 gradients, 0.1 GFLOPs
Label: [3 8 5 8 5] Pred: [3 8 5 8 5]
Label: [4 8 6 8 0] Pred: [4 8 6 8 0]
Label: [4 6 4 7 0] Pred: [4 6 4 7 0]
Label: [2 3 5 0 7] Pred: [2 3 5 0 7]
Label: [4 7 8 4 6] Pred: [4 7 8 4 6]
Label: [0 1 4 3 6] Pred: [0 1 4 3 6]
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_emnist.py crnn-emnist-b512-e100.pth ../datasets/emnist/ ./runs/predict/emnist/ --not-tiny
args: Namespace(not_tiny=True, pretrained='crnn-emnist-b512-e100.pth', save_dir='./runs/predict/emnist/', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn-emnist-b512-e100.pth
crnn-emnist-b512-e100 summary: 29 layers, 7924363 parameters, 7924363 gradients, 2.2 GFLOPs
Label: [9 5 5 8 1] Pred: [9 5 5 8 1]
Label: [8 9 1 0 3] Pred: [8 9 1 0 3]
Label: [9 9 3 0 4] Pred: [9 9 3 0 4]
Label: [5 4 3 2 3] Pred: [5 4 3 2 3]
Label: [8 8 5 8 9] Pred: [8 8 5 8 9]
Label: [8 9 1 5 1] Pred: [8 9 1 5 1]
```

## License Plate

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn_tiny-plate-b512-e100.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/
args: Namespace(image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=False, pretrained='crnn_tiny-plate-b512-e100.pth', save_dir='runs/predict/plate/', use_lstm=False)
Loading CRNN pretrained: crnn_tiny-plate-b512-e100.pth
crnn_tiny-plate-b512-e100 summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Pred: 宁A·87J92 - Predict time: 8.4 ms
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn-plate-b512-e100.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --not-tiny
args: Namespace(image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=True, pretrained='crnn-plate-b512-e100.pth', save_dir='runs/predict/plate/', use_lstm=False)
Loading CRNN pretrained: crnn-plate-b512-e100.pth
crnn-plate-b512-e100 summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Pred: 宁A·87J92 - Predict time: 5.7 ms
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn_tiny-plate-b512-e100.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/
args: Namespace(image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=False, pretrained='crnn_tiny-plate-b512-e100.pth', save_dir='runs/predict/plate/', use_lstm=False)
Loading CRNN pretrained: crnn_tiny-plate-b512-e100.pth
crnn_tiny-plate-b512-e100 summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Pred: 川A·3X7J1 - Predict time: 8.4 ms
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn-plate-b512-e100.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/ --not-tiny
args: Namespace(image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=True, pretrained='crnn-plate-b512-e100.pth', save_dir='runs/predict/plate/', use_lstm=False)
Loading CRNN pretrained: crnn-plate-b512-e100.pth
crnn-plate-b512-e100 summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Pred: 川A·3X7J1 - Predict time: 6.0 ms
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```