<!-- <div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div> -->

<div align="center"><a title="" href="https://github.com/zjykzj/crnn-ctc"><img align="center" src="assets/icons/crnn-ctc.svg" alt=""></a></div>

<p align="center">
  «crnn-ctc» implemented CRNN+CTC
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
</p>

|   **Model**   | **ARCH** | **Model Size (MB)** | **EMNIST Accuracy (%)** | **Training Data** | **Testing Data** |
|:-------------:|:--------:|:-------------------:|:-----------------------:|:-----------------:|:----------------:|
|   **CRNN**    | CONV+GRU |         31          |         98.546          |      100,000      |      5,000       |
| **CRNN_Tiny** | CONV+GRU |         1.7         |         98.396          |      100,000      |      5,000       |

|   **Model**   | **ARCH** | **Model Size (MB)** | **ChineseLicensePlate Accuracy (%)** | **Training Data** | **Testing Data** |
|:-------------:|:--------:|:-------------------:|:------------------------------------:|:-----------------:|:----------------:|
|   **CRNN**    | CONV+GRU |         58          |                82.379                |      269,621      |     149,002      |
| **CRNN_Tiny** | CONV+GRU |          4          |                76.222                |      269,621      |     149,002      |

## Table of Contents

- [Table of Contents](#table-of-contents)
- [News🚀🚀🚀](#news)
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
  - [Train](#train)
  - [Eval](#eval)
  - [Predict](#predict)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## News🚀🚀🚀

| Version                                                          | Release Date | Major Updates                                                                          |
|------------------------------------------------------------------|--------------|----------------------------------------------------------------------------------------|
| [v0.3.0](https://github.com/zjykzj/crnn-ctc/releases/tag/v0.3.0) | 2024/08/03   | Implement models CRNN_LSTM and CRNN_GRU on datasets EMNIST and ChineseLicensePlate.    |
| [v0.2.0](https://github.com/zjykzj/crnn-ctc/releases/tag/v0.2.0) | 2023/10/11   | Support training/evaluation/prediction of CRNN+CTC based on license plate.             |
| [v0.1.0](https://github.com/zjykzj/crnn-ctc/releases/tag/v0.1.0) | 2023/10/10   | Support training/evaluation/prediction of CRNN+CTC based on EMNIST digital characters. |

## Background

This warehouse aims to better understand and apply CRNN+CTC, and currently achieves digital recognition and license plate recognition

## Installation

```shell
pip install -r requirements.txt
```

## Usage

### Train

* ChineseLicensePlate: [Baidu Drive](https://pan.baidu.com/s/1fQh0E9c6Z4satvrEthKevg)(ad7l)

```shell
# EMNIST
$ python3 train_emnist.py ../datasets/emnist/ ./runs/crnn-emnist-b512/ --batch-size 512 --device 0 --not-tiny
# Plate
$ python3 train_plate.py ../datasets/chinese_license_plate/recog/ ./runs/crnn-plate-b512/ --batch-size 512 --device 0 --not-tiny
```

### Eval

```shell
# EMNIST
$ CUDA_VISIBLE_DEVICES=0 python eval_emnist.py runs/crnn-emnist-b512-e100.pth ../datasets/emnist/ --not-tiny
args: Namespace(not_tiny=True, pretrained='runs/crnn-emnist-b512-e100.pth', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: runs/crnn-emnist-b512-e100.pth
Batch:1562 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [00:22<00:00, 69.08it/s]
ACC:98.546
# Plate
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py ./runs/crnn-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/ --not-tiny
args: Namespace(not_tiny=True, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='./runs/crnn-plate-b512-e100.pth', use_lstm=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: ./runs/crnn-plate-b512-e100.pth
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 4657/4657 [00:56<00:00, 83.05it/s]
ACC:82.379
```

### Predict

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

![](assets/predict/emnist/predict_emnist.jpg)

```shell
$ python predict_plate.py ./runs/crnn-plate-b512-e100.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/ --not-tiny
args: Namespace(image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=True, pretrained='./runs/crnn-plate-b512-e100.pth', save_dir='runs/predict/plate/', use_lstm=False)
Loading CRNN pretrained: ./runs/crnn-plate-b512-e100.pth
Pred: 宁A·87J92
Save to runs/predict/plate/plate_宁A87J92_0.jpg
$ python predict_plate.py ./runs/crnn-plate-b512-e100.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/ --not-tiny
args: Namespace(image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=True, pretrained='./runs/crnn-plate-b512-e100.pth', save_dir='runs/predict/plate/', use_lstm=False)
Loading CRNN pretrained: ./runs/crnn-plate-b512-e100.pth
Pred: 川A·3X7J1
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```

<p align="left"><img src="assets/predict/plate/plate_宁A87J92_0.jpg" height="240"\>  <img src="assets/predict/plate/plate_川A3X7J1_0.jpg" height="240"\></p>

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [rinabuoy/crnn-ctc-loss-pytorch](https://github.com/rinabuoy/crnn-ctc-loss-pytorch.git)
* [we0091234/crnn_plate_recognition](https://github.com/we0091234/crnn_plate_recognition.git)
* [zjykzj/LPDet](https://github.com/zjykzj/LPDet)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/crnn-ctc/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2023 zjykzj