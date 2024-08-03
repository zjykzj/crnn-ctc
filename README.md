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

| **Model** | **ARCH**  | **Model Size (MB)** | **EMNIST Accuracy (%)** | **Training Data** | **Testing Data** |
|:---------:|:---------:|:-------------------:|:-----------------------:|:-----------------:|:----------------:|
| **CRNN**  | CONV+LSTM |         34          |         98.432          |      100,000      |      5,000       |
| **CRNN**  | CONV+GRU  |         31          |         98.386          |      100,000      |      5,000       |

| **Model** | **ARCH**  | **Model Size (MB)** | **ChineseLicensePlate Accuracy (%)** | **Training Data** | **Testing Data** |
|:---------:|:---------:|:-------------------:|:------------------------------------:|:-----------------:|:----------------:|
| **CRNN**  | CONV+LSTM |         70          |                74.252                |      269,621      |     149,002      |
| **CRNN**  | CONV+GRU  |         58          |                75.649                |      269,621      |     149,002      |

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Latest News](#latest-news)
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

## Latest News

* ***[2023/10/11][v0.2.0](https://github.com/zjykzj/crnn-ctc/releases/tag/v0.2.0). Support training/evaluation/prediction of CRNN+CTC based on license plate.***
* ***[2023/10/10][v0.1.0](https://github.com/zjykzj/crnn-ctc/releases/tag/v0.1.0). Support training/evaluation/prediction of CRNN+CTC based on EMNIST digital characters.***

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
$ python3 train_emnist.py ../datasets/emnist/ ./runs/crnn_gru-emnist-b512/ --batch-size 512 --device 0 --use-gru
# Plate
$ python3 train_plate.py ../datasets/chinese_license_plate/recog/ ./runs/crnn_gru-plate-b256/ --batch-size 256 --device 0 --use-gru
```

### Eval

```shell
# EMNIST
$ python3 eval_emnist.py ./runs/crnn_gru-emnist-b512/crnn_gru-emnist-b512-e100.pth ../datasets/emnist/ --use-gru
args: Namespace(pretrained='./runs/crnn_gru-emnist-b512/crnn_gru-emnist-b512-e100.pth', use_gru=True, val_root='../datasets/emnist/')
Loading CRNN pretrained: ./runs/crnn_gru-emnist-b512/crnn_gru-emnist-b512-e100.pth
Batch:1562 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [00:38<00:00, 40.34it/s]
ACC:98.432
# Plate
$ CUDA_VISIBLE_DEVICES=1 python3 eval_plate.py ./runs/crnn_gru-plate-b256-e100.pth ../datasets/chinese_license_plate/recog/ --use-gru
args: Namespace(only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='./runs/crnn_gru-plate-b256-e100.pth', use_gru=True, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: ./runs/crnn_gru-plate-b256-e100.pth
Load test data: 149002
Batch:4656 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4657/4657 [07:27<00:00, 10.41it/s]
ACC:75.649
```

### Predict

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

![](assets/predict/emnist/predict_emnist.jpg)

```shell
$ python predict_plate.py runs/crnn_gru-plate-b256-e100.pth ./assets/plate/宁A87J92_0.jpg ./runs/predict/plate/ --use-gru
args: Namespace(image_path='./assets/plate/宁A87J92_0.jpg', pretrained='runs/crnn_gru-plate-b256-e100.pth', save_dir='./runs/predict/plate/', use_gru=True)
Loading CRNN pretrained: runs/crnn_gru-plate-b256-e100.pth
Pred: 宁A·87J92
Save to ./runs/predict/plate/plate_宁A87J92_0.jpg
$ python predict_plate.py runs/crnn_gru-plate-b256-e100.pth ./assets/plate/川A3X7J1_0.jpg ./runs/predict/plate/ --use-gru
args: Namespace(image_path='./assets/plate/川A3X7J1_0.jpg', pretrained='runs/crnn_gru-plate-b256-e100.pth', save_dir='./runs/predict/plate/', use_gru=True)
Loading CRNN pretrained: runs/crnn_gru-plate-b256-e100.pth
Pred: 川A·3X7J1
Save to ./runs/predict/plate/plate_川A3X7J1_0.jpg
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