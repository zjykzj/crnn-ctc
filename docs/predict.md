# PREDICT

## EMNIST

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_emnist.py crnn_tiny-emnist.pth ../datasets/emnist/ ./runs/predict/emnist/
args: Namespace(not_tiny=False, pretrained='crnn_tiny-emnist.pth', save_dir='./runs/predict/emnist/', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn_tiny-emnist.pth
crnn_tiny-emnist summary: 22 layers, 427467 parameters, 427467 gradients, 0.1 GFLOPs
Label: [2 4 5 3 4] Pred: [2 4 5 3 4]
Label: [4 8 3 4 1] Pred: [4 8 3 4 1]
Label: [0 7 5 8 4] Pred: [0 7 5 8 4]
Label: [8 7 5 5 5] Pred: [8 7 5 5 5]
Label: [1 0 5 1 1] Pred: [1 0 5 1 1]
Label: [5 6 7 9 8] Pred: [5 6 7 9 8]
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_emnist.py crnn-emnist.pth ../datasets/emnist/ ./runs/predict/emnist/ --not-tiny
args: Namespace(not_tiny=True, pretrained='crnn-emnist.pth', save_dir='./runs/predict/emnist/', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn-emnist.pth
crnn-emnist summary: 29 layers, 7924363 parameters, 7924363 gradients, 2.2 GFLOPs
Label: [0 4 2 4 7] Pred: [0 4 2 4 7]
Label: [2 0 6 5 4] Pred: [2 0 6 5 4]
Label: [7 3 9 9 5] Pred: [7 3 9 9 5]
Label: [9 6 6 0 9] Pred: [9 6 6 0 9]
Label: [2 3 0 7 6] Pred: [2 3 0 7 6]
Label: [6 5 9 5 2] Pred: [6 5 9 5 2]
```

## License Plate

### CRNN/CRNN_Tiny

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn_tiny-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/
args: Namespace(add_stnet=False, image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=False, pretrained='crnn_tiny-plate.pth', save_dir='runs/predict/plate/', use_lprnet=False, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: crnn_tiny-plate.pth
crnn_tiny-plate summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Pred: 宁A·87J92 - Predict time: 3.7 ms
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --not-tiny
args: Namespace(add_stnet=False, image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=True, pretrained='crnn-plate.pth', save_dir='runs/predict/plate/', use_lprnet=False, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: crnn-plate.pth
crnn-plate summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Pred: 宁A·87J92 - Predict time: 5.4 ms
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn_tiny-plate.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/
args: Namespace(add_stnet=False, image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=False, pretrained='crnn_tiny-plate.pth', save_dir='runs/predict/plate/', use_lprnet=False, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: crnn_tiny-plate.pth
crnn_tiny-plate summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Pred: 川A·3X7J1 - Predict time: 3.7 ms
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn-plate.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/ --not-tiny
args: Namespace(add_stnet=False, image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=True, pretrained='crnn-plate.pth', save_dir='runs/predict/plate/', use_lprnet=False, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: crnn-plate.pth
crnn-plate summary: 29 layers, 15083854 parameters, 15083854 gradients, 4.0 GFLOPs
Pred: 川A·3X7J1 - Predict time: 4.7 ms
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```

### LPRNet/LPRNetPlus

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py lprnet-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --use-lprnet --use-origin-block
args: Namespace(add_stnet=False, image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=False, pretrained='lprnet-plate.pth', save_dir='runs/predict/plate/', use_lprnet=True, use_lstm=False, use_origin_block=True)
Loading CRNN pretrained: lprnet-plate.pth
lprnet-plate summary: 51 layers, 486236 parameters, 486236 gradients, 0.3 GFLOPs
Pred: 宁A·87J92 - Predict time: 2.6 ms
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py lprnet_plus-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --use-lprnet
args: Namespace(add_stnet=False, image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=False, pretrained='lprnet_plus-plate.pth', save_dir='runs/predict/plate/', use_lprnet=True, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: lprnet_plus-plate.pth
lprnet_plus-plate summary: 57 layers, 576988 parameters, 576988 gradients, 0.5 GFLOPs
Pred: 宁A·87J92 - Predict time: 2.7 ms
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py lprnet-plate.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/ --use-lprnet --use-origin-block
args: Namespace(add_stnet=False, image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=False, pretrained='lprnet-plate.pth', save_dir='runs/predict/plate/', use_lprnet=True, use_lstm=False, use_origin_block=True)
Loading CRNN pretrained: lprnet-plate.pth
lprnet-plate summary: 51 layers, 486236 parameters, 486236 gradients, 0.3 GFLOPs
Pred: 川A·3X7J1 - Predict time: 2.5 ms
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py lprnet_plus-plate.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/ --use-lprnet
args: Namespace(add_stnet=False, image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=False, pretrained='lprnet_plus-plate.pth', save_dir='runs/predict/plate/', use_lprnet=True, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: lprnet_plus-plate.pth
lprnet_plus-plate summary: 57 layers, 576988 parameters, 576988 gradients, 0.5 GFLOPs
Pred: 川A·3X7J1 - Predict time: 2.7 ms
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```

### LPRNet/LPRNetPlus+STNet

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py lprnet_plus_stnet-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --use-lprnet --add-stnet
args: Namespace(add_stnet=True, image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=False, pretrained='lprnet_plus_stnet-plate.pth', save_dir='runs/predict/plate/', use_lprnet=True, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: lprnet_plus_stnet-plate.pth
/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:4223: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
lprnet_plus_stnet-plate summary: 69 layers, 632418 parameters, 632418 gradients, 0.5 GFLOPs
Pred: 宁A·87J92 - Predict time: 3.0 ms
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py lprnet_stnet-plate.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --use-lprnet --use-origin-block --add-stnet
args: Namespace(add_stnet=True, image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=False, pretrained='lprnet_stnet-plate.pth', save_dir='runs/predict/plate/', use_lprnet=True, use_lstm=False, use_origin_block=True)
Loading CRNN pretrained: lprnet_stnet-plate.pth
/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:4223: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
lprnet_stnet-plate summary: 63 layers, 541666 parameters, 541666 gradients, 0.3 GFLOPs
Pred: 宁A·87J92 - Predict time: 3.0 ms
Save to runs/predict/plate/plate_宁A87J92_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py lprnet_plus_stnet-plate.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/ --use-lprnet --add-stnet
args: Namespace(add_stnet=True, image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=False, pretrained='lprnet_plus_stnet-plate.pth', save_dir='runs/predict/plate/', use_lprnet=True, use_lstm=False, use_origin_block=False)
Loading CRNN pretrained: lprnet_plus_stnet-plate.pth
/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:4223: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
lprnet_plus_stnet-plate summary: 69 layers, 632418 parameters, 632418 gradients, 0.5 GFLOPs
Pred: 川A·3X7J1 - Predict time: 3.2 ms
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py lprnet_stnet-plate.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/ --use-lprnet --use-origin-block --add-stnet
args: Namespace(add_stnet=True, image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=False, pretrained='lprnet_stnet-plate.pth', save_dir='runs/predict/plate/', use_lprnet=True, use_lstm=False, use_origin_block=True)
Loading CRNN pretrained: lprnet_stnet-plate.pth
/opt/conda/lib/python3.8/site-packages/torch/nn/functional.py:4223: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
lprnet_stnet-plate summary: 63 layers, 541666 parameters, 541666 gradients, 0.3 GFLOPs
Pred: 川A·3X7J1 - Predict time: 2.9 ms
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```