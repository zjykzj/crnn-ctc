# PREDICT

## EMNIST

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_emnist.py crnn_tiny-emnist.pth ../datasets/emnist/ ./runs/predict/emnist/
args: Namespace(not_tiny=False, pretrained='crnn_tiny-emnist.pth', save_dir='./runs/predict/emnist/', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn_tiny-emnist.pth
crnn_tiny-emnist summary: 22 layers, 427467 parameters, 427467 gradients, 0.1 GFLOPs
Label: [3 8 5 8 5] Pred: [3 8 5 8 5]
Label: [4 8 6 8 0] Pred: [4 8 6 8 0]
Label: [4 6 4 7 0] Pred: [4 6 4 7 0]
Label: [2 3 5 0 7] Pred: [2 3 5 0 7]
Label: [4 7 8 4 6] Pred: [4 7 8 4 6]
Label: [0 1 4 3 6] Pred: [0 1 4 3 6]
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_emnist.py crnn-emnist.pth ../datasets/emnist/ ./runs/predict/emnist/ --not-tiny
args: Namespace(not_tiny=True, pretrained='crnn-emnist.pth', save_dir='./runs/predict/emnist/', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn-emnist.pth
crnn-emnist summary: 29 layers, 7924363 parameters, 7924363 gradients, 2.2 GFLOPs
Label: [9 5 5 8 1] Pred: [9 5 5 8 1]
Label: [8 9 1 0 3] Pred: [8 9 1 0 3]
Label: [9 9 3 0 4] Pred: [9 9 3 0 4]
Label: [5 4 3 2 3] Pred: [5 4 3 2 3]
Label: [8 8 5 8 9] Pred: [8 8 5 8 9]
Label: [8 9 1 5 1] Pred: [8 9 1 5 1]
```

## License Plate

### CRNN/CRNN_Tiny

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn_tiny-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/
args: Namespace(image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=False, pretrained='crnn_tiny-plate.pth', save_dir='runs/predict/plate/', use_lprnet=False, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: crnn_tiny-plate.pth
crnn_tiny-plate summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Pred: 宁A·87J92 - Predict time: 3.7 ms
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --not-tiny
args: Namespace(image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=True, pretrained='crnn-plate.pth', save_dir='runs/predict/plate/', use_lprnet=False, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: crnn-plate.pth
crnn-plate summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Pred: 宁A·87J92 - Predict time: 4.4 ms
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn_tiny-plate.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/
args: Namespace(image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=False, pretrained='crnn_tiny-plate.pth', save_dir='runs/predict/plate/', use_lprnet=False, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: crnn_tiny-plate.pth
crnn_tiny-plate summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Pred: 川A·3X7J1 - Predict time: 3.5 ms
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn-plate.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/ --not-tiny
args: Namespace(image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=True, pretrained='crnn-plate.pth', save_dir='runs/predict/plate/', use_lprnet=False, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: crnn-plate.pth
crnn-plate summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Pred: 川A·3X7J1 - Predict time: 4.5 ms
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```

### LPRNet/LPRNetPlus

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py lprnet-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --use-lprnet --use-origin-block
args: Namespace(image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=False, pretrained='lprnet-plate.pth', save_dir='runs/predict/plate/', use_lprnet=True, use_lstm=False, use_origin_block=True)
Loading CRNN pretrained: lprnet-plate.pth
lprnet-plate summary: 51 layers, 486236 parameters, 486236 gradients, 0.3 GFLOPs
Pred: 宁A·87J92 - Predict time: 2.5 ms
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py lprnet_plus-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --use-lprnet
args: Namespace(image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=False, pretrained='lprnet_plus-plate.pth', save_dir='runs/predict/plate/', use_lprnet=True, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: lprnet_plus-plate.pth
lprnet_plus-plate summary: 57 layers, 576988 parameters, 576988 gradients, 0.5 GFLOPs
Pred: 宁A·87J92 - Predict time: 2.8 ms
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py lprnet-plate.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/ --use-lprnet --use-origin-block
args: Namespace(image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=False, pretrained='lprnet-plate.pth', save_dir='runs/predict/plate/', use_lprnet=True, use_lstm=False, use_origin_block=True)
Loading CRNN pretrained: lprnet-plate.pth
lprnet-plate summary: 51 layers, 486236 parameters, 486236 gradients, 0.3 GFLOPs
Pred: 川A·3X7J1 - Predict time: 2.6 ms
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py lprnet_plus-plate.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/ --use-lprnet
args: Namespace(image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=False, pretrained='lprnet_plus-plate.pth', save_dir='runs/predict/plate/', use_lprnet=True, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: lprnet_plus-plate.pth
lprnet_plus-plate summary: 57 layers, 576988 parameters, 576988 gradients, 0.5 GFLOPs
Pred: 川A·3X7J1 - Predict time: 2.7 ms
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```