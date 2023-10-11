# -*- coding: utf-8 -*-

"""
@date: 2023/10/9 下午3:08
@file: train.py
@author: zj
@description:

Usage - Single-GPU training:
    $ python train_emist.py ../crnn-ctc-loss-pytorch/EMNIST/ runs/emnist/

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 32512 train_emist.py --device 0,1,2,3 ../datasets/EMNIST/ runs/emnist_ddp/

"""

import argparse
import os.path
import time

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed

from utils.model.crnn_gru import CRNN
from utils.loss import CTCLoss
from utils.evaluator import Evaluator
from utils.torchutil import select_device
from utils.ddputil import smart_DDP
from utils.logger import LOGGER
from utils.general import init_seeds
from utils.dataset.emnist import EMNISTDataset

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('data_root', metavar='DIR', type=str, help='path to EMNIST dataset')
    parser.add_argument('output', metavar='OUTPUT', type=str, help='path to output')

    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs, -1 for autobatch')

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
    data_root, batch_size, output = opt.data_root, opt.batch_size, opt.output
    if RANK in {-1, 0} and not os.path.exists(output):
        os.makedirs(output)

    LOGGER.info("=> Create Model")
    model = CRNN(in_channel=1, num_classes=11, cnn_output_height=4).to(device)
    blank_label = 10
    criterion = CTCLoss(blank_label=blank_label).to(device)

    learn_rate = 0.001 * WORLD_SIZE
    weight_decay = 1e-5
    LOGGER.info(f"Final learning rate: {learn_rate}, weight decay: {weight_decay}")
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70, 90])

    LOGGER.info("=> Load data")
    train_dataset = EMNISTDataset(data_root, is_train=True, num_of_sequences=10000, digits_per_sequence=5)
    sampler = None if LOCAL_RANK == -1 else distributed.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True and sampler is None,
                                  sampler=sampler,
                                  num_workers=4,
                                  drop_last=False,
                                  pin_memory=True)
    if RANK in {-1, 0}:
        val_dataset = EMNISTDataset(data_root, is_train=False, num_of_sequences=2000, digits_per_sequence=5)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False,
                                    pin_memory=True)

        LOGGER.info("=> Load evaluator")
        emnist_evaluator = Evaluator(blank_label=blank_label)

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
            images = images.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast(amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
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
            save_path = os.path.join(output, f"crnn-emnist-e{epoch}.pth")
            LOGGER.info(f"Save to {save_path}")
            torch.save(model.state_dict(), save_path)

            emnist_evaluator.reset()
            pbar = tqdm(val_dataloader)
            for idx, (images, targets) in enumerate(pbar):
                images = images.to(device)
                with torch.no_grad():
                    outputs = model(images).cpu()

                acc = emnist_evaluator.update(outputs, targets)
                info = f"Batch:{idx} ACC:{acc * 100:.3f}"
                pbar.set_description(info)
            acc = emnist_evaluator.result()
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
