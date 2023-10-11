# -*- coding: utf-8 -*-

"""
@date: 2023/10/9 下午3:08
@file: train.py
@author: zj
@description:

Usage - Single-GPU training:
    $ python train.py ../crnn-ctc-loss-pytorch/EMNIST/ runs/

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 32512 train.py --device 0,1,2,3 ../datasets/EMNIST/ runs/emnist/

"""

import argparse
import os.path
import time

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed

# from utils.model.crnn_gru import CRNN
from utils.model.crnn_conv import CRNN
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
    parser.add_argument('train_root', metavar='DIR', type=str, help='path to plate train dataset')
    parser.add_argument('val_root', metavar='DIR', type=str, help='path to plate val dataset')
    parser.add_argument('output', metavar='OUTPUT', type=str, help='path to output')

    parser.add_argument('--batch-size', type=int, default=256, help='total batch size for all GPUs, -1 for autobatch')

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
    train_root, val_root, batch_size, output = opt.train_root, opt.val_root, opt.batch_size, opt.output
    if RANK in {-1, 0} and not os.path.exists(output):
        os.makedirs(output)

    LOGGER.info("=> Create Model")
    # model = CRNN(in_channel=3, num_classes=len(PLATE_CHARS), cnn_output_height=9).to(device)
    cfg = [16, 16, 32, 32, 'M', 64, 64, 'M', 96, 96, 'M', 128, 256]
    model = CRNN(num_classes=len(PLATE_CHARS), cfg=cfg).to(device)
    blank_label = 0
    criterion = CTCLoss(blank_label=blank_label).to(device)
    # criterion = CTCLoss(blank_label=blank_label)
    # criterion = torch.nn.CTCLoss()

    learn_rate = 0.001 * WORLD_SIZE
    weight_decay = 0.
    LOGGER.info(f"Final learning rate: {learn_rate}, weight decay: {weight_decay}")
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70, 90])

    LOGGER.info("=> Load data")
    train_dataset = PlateDataset(train_root, is_train=True, img_w=168, img_h=48)
    sampler = None if LOCAL_RANK == -1 else distributed.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True and sampler is None,
                                  sampler=sampler,
                                  num_workers=4,
                                  drop_last=True,
                                  pin_memory=True)
    if RANK in {-1, 0}:
        val_dataset = PlateDataset(val_root, is_train=False, img_w=168, img_h=48)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False,
                                    pin_memory=True)

        LOGGER.info("=> Load evaluator")
        evaluator = Evaluator(blank_label=blank_label)

    LOGGER.info("=> Start training")
    t0 = time.time()
    amp = True
    # scaler = torch.cuda.amp.GradScaler(enabled=amp)

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

            # targets = targets.to(device)
            targets = train_dataset.convert(targets)
            target_lengths = torch.IntTensor([len(t) for t in targets])
            targets = torch.concat(targets)

            # with torch.cuda.amp.autocast(amp):
            outputs = model(images.to(device), export=False).cpu()
            # preds_size = torch.IntTensor([outputs.size(0)] * batch_size)  # timestep * batchsize
            loss = criterion(outputs, targets, target_lengths)
            # loss = criterion(outputs, targets, preds_size, target_lengths)
            # scaler.scale(loss).backward()
            loss.backward()

            if epoch <= warmup_epoch:
                adjust_learning_rate(learn_rate, warmup_epoch, optimizer, epoch - 1, idx, len(train_dataloader))

            # scaler.step(optimizer)  # optimizer.step
            # scaler.update()
            optimizer.step()
            optimizer.zero_grad()

            if RANK in {-1, 0}:
                lr = optimizer.param_groups[0]["lr"]
                info = f"Epoch:{epoch} Batch:{idx} LR:{lr:.6f} Loss:{loss:.6f}"
                pbar.set_description(info)

        if RANK in {-1, 0} and epoch % 1 == 0 and epoch > 0:
            model.eval()
            save_path = os.path.join(output, f"crnn-plate-e{epoch}.pth")
            LOGGER.info(f"Save to {save_path}")
            torch.save(model.state_dict(), save_path)

            evaluator.reset()
            pbar = tqdm(val_dataloader)
            for idx, (images, targets) in enumerate(pbar):
                images = images.to(device)
                targets = val_dataset.convert(targets)
                with torch.no_grad():
                    outputs = model(images, export=True).cpu()

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
