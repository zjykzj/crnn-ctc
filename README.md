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

...

## Background

This warehouse aims to better understand and apply CRNN+CTC, and currently achieves digital recognition and license plate recognition

## Installation

```shell
pip install -r requirements.txt
```

## Usage

### Train

```shell
python -m torch.distributed.run --nproc_per_node 4 --master_port 32512 train.py --device 0,1,2,3 ../datasets/EMNIST/ runs/emnist/
```

### Eval

```shell
$ python eval.py runs/emnist/CRNN-e100.pth ../datasets/EMNIST/
args: Namespace(pretrained='runs/emnist/CRNN-e100.pth', val_root='../datasets/EMNIST/')
Loading CRNN pretrained: runs/emnist/CRNN-e100.pth
Batch:62 ACC:93.750: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [00:02<00:00, 31.24it/s]
ACC:94.550
```

### Predict

```shell
$ python predict.py runs/emnist/CRNN-e100.pth ../datasets/EMNIST/ runs/
args: Namespace(pretrained='runs/emnist/CRNN-e100.pth', save_dir='runs/', val_root='../datasets/EMNIST/')
Loading CRNN pretrained: runs/emnist/CRNN-e100.pth
Label: [8 5 6 3 0] Pred: [8 5 6 3 0]
Label: [9 5 9 5 6] Pred: [9 5 9 5 6]
Label: [1 6 0 2 7] Pred: [1 6 0 2 7]
Label: [9 3 2 3 3] Pred: [9 3 2 3 3]
Label: [5 3 5 3 5] Pred: [5 3 6 3 5]
Label: [4 4 9 6 4] Pred: [4 4 9 6 4]
```

![](assets/predict.jpg)

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [rinabuoy/crnn-ctc-loss-pytorch](https://github.com/rinabuoy/crnn-ctc-loss-pytorch.git)
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