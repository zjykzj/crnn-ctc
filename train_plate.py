# -*- coding: utf-8 -*-

"""
@date: 2023/10/9 下午3:08
@file: train.py
@author: zj
@description:

Usage - Single-GPU training using CRNN:
    $ python3 train_plate.py ../datasets/chinese_license_plate/recog/ ./runs/crnn_tiny-plate-b512/ --batch-size 512 --device 0
    $ python3 train_plate.py ../datasets/chinese_license_plate/recog/ ./runs/crnn-plate-b512/ --batch-size 512 --device 0 --not-tiny

Usage - Single-GPU training using LPRNet:
    $ python3 train_plate.py ../datasets/chinese_license_plate/recog/ ./runs/lprnetv2-plate-b512/ --batch-size 512 --device 0 --use-lprnet
    $ python3 train_plate.py ../datasets/chinese_license_plate/recog/ ./runs/lprnet-plate-b512/ --batch-size 512 --device 0 --use-lprnet --use-origin-block

"""

import argparse
import os.path
import time

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed

from utils.model.crnn import CRNN
from utils.model.lprnet import LPRNet
from utils.loss import CTCLoss
from utils.evaluator import Evaluator
from utils.torchutil import select_device
from utils.ddputil import smart_DDP
from utils.logger import LOGGER
from utils.general import init_seeds
from utils.dataset.plate import PlateDataset, PLATE_CHARS

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('data', metavar='DIR', type=str, help='path to chinese_license_plate dataset')
    parser.add_argument('output', metavar='OUTPUT', type=str, help='path to output')

    parser.add_argument('--batch-size', type=int, default=512, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--use-lstm', action='store_true', help='use nn.LSTM instead of nn.GRU')
    parser.add_argument('--not-tiny', action='store_true', help='use this flag to specify non-tiny mode')

    parser.add_argument("--use-lprnet", action='store_true', help='use LPRNet instead of CRNN')
    parser.add_argument("--use-origin-block", action='store_true', help='use origin small_basic_block impl')

    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    args = parser.parse_args()
    LOGGER.info(f"args: {args}")
    return args


def adjust_learning_rate(lr, warmup_epoch, optimizer, epoch: int, step: int, len_epoch: int) -> None:
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    # Warmup
    lr = lr * float(1 + step + epoch * len_epoch) / (warmup_epoch * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(opt, device):
    data_root, batch_size, not_tiny, use_lstm, use_lprnet, use_origin_block, output = \
        opt.data, opt.batch_size, opt.not_tiny, opt.use_lstm, opt.use_lprnet, opt.use_origin_block, opt.output
    if RANK in {-1, 0} and not os.path.exists(output):
        os.makedirs(output)

    LOGGER.info("=> Create Model")
    if use_lprnet:
        # (W, H)
        input_shape = (94, 24)
        model = LPRNet(in_channel=3, num_classes=len(PLATE_CHARS), use_origin_block=use_origin_block).to(device)
        if use_origin_block:
            model_prefix = 'lprnet'
        else:
            model_prefix = "lprnetv2"
    else:
        input_shape = (168, 48)
        model = CRNN(in_channel=3, num_classes=len(PLATE_CHARS), cnn_input_height=input_shape[1], is_tiny=not not_tiny,
                     use_gru=not use_lstm).to(device)
        if not_tiny:
            model_prefix = 'crnn'
        else:
            model_prefix = "crnn_tiny"

    blank_label = 0
    criterion = CTCLoss(blank_label=blank_label).to(device)

    learn_rate = 0.001 * WORLD_SIZE
    weight_decay = 1e-5
    LOGGER.info(f"Final learning rate: {learn_rate}, weight decay: {weight_decay}")
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70, 90])

    LOGGER.info("=> Load data")
    train_dataset = PlateDataset(data_root, is_train=True, input_shape=input_shape)
    sampler = None if LOCAL_RANK == -1 else distributed.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True and sampler is None,
                                  sampler=sampler,
                                  num_workers=4,
                                  drop_last=True,
                                  pin_memory=True)
    if RANK in {-1, 0}:
        val_dataset = PlateDataset(data_root, is_train=False, input_shape=input_shape)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False,
                                    pin_memory=True)

        LOGGER.info("=> Load evaluator")
        evaluator = Evaluator(blank_label=blank_label)

    LOGGER.info("=> Start training")
    t0 = time.time()
    amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # DDP mode
    cuda = device.type != 'cpu'
    if cuda and RANK != -1:
        model = smart_DDP(model)

    epochs = 100
    start_epoch = 1
    warmup_epoch = 5
    for epoch in range(start_epoch, epochs + start_epoch):
        # epoch: start from 1
        model.train()
        if RANK != -1:
            train_dataloader.sampler.set_epoch(epoch)

        pbar = train_dataloader
        if LOCAL_RANK in {-1, 0}:
            pbar = tqdm(pbar)
        optimizer.zero_grad()
        for idx, (images, targets) in enumerate(pbar):
            batch_size = len(images)

            targets = train_dataset.convert(targets)
            target_lengths = torch.IntTensor([len(t) for t in targets]).to(device)
            targets = torch.concat(targets).to(device)

            with torch.cuda.amp.autocast(amp):
                outputs = model(images.to(device))
                loss = criterion(outputs, targets, target_lengths)
            scaler.scale(loss).backward()

            if epoch <= warmup_epoch:
                adjust_learning_rate(learn_rate, warmup_epoch, optimizer, epoch - 1, idx, len(train_dataloader))

            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()

            if RANK in {-1, 0}:
                lr = optimizer.param_groups[0]["lr"]
                info = f"Epoch:{epoch} Batch:{idx} LR:{lr:.6f} Loss:{loss:.6f}"
                pbar.set_description(info)

        if RANK in {-1, 0} and epoch % 5 == 0 and epoch > 0:
            model.eval()
            if not_tiny:
                save_path = os.path.join(output, f"{model_prefix}-plate-b{batch_size}-e{epoch}.pth")
            else:
                save_path = os.path.join(output, f"{model_prefix}_tiny-plate-b{batch_size}-e{epoch}.pth")
            LOGGER.info(f"Save to {save_path}")
            torch.save(model.state_dict(), save_path)

            evaluator.reset()
            pbar = tqdm(val_dataloader)
            for idx, (images, targets) in enumerate(pbar):
                images = images.to(device)
                targets = val_dataset.convert(targets)
                with torch.no_grad():
                    outputs = model(images).cpu()

                acc = evaluator.update(outputs, targets)
                info = f"Batch:{idx} ACC:{acc * 100:.3f}"
                pbar.set_description(info)
            acc = evaluator.result()
            LOGGER.info(f"ACC: {acc * 100:.3f}")
        scheduler.step()
        torch.cuda.empty_cache()
    LOGGER.info(f'\n{epochs} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')


def main(opt):
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with LPDet Multi-GPU DDP training'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    init_seeds(opt.seed + 1 + RANK, deterministic=False)
    # LOGGER.info(f"LOCAL_RANK: {LOCAL_RANK} RANK: {RANK} WORLD_SIZE: {WORLD_SIZE}")
    train(opt, device)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
