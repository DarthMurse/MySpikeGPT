import os
import torch 
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from spikingjelly.activation_based import functional
import functools

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from src.model import MySpikeGPT 
from src.utils import *

class TrainConfig:
    def __init__(self,
                 model,
                 dataloader,
                 optimizer,
                 lr_scheduler,
                 model_args=args
                 ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer 
        self.lr_scheduler = lr_scheduler
        self.args = model_args 

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_one_epoch(train_config, rank, world_size):
    model = train_config.model
    dataloader = train_config.dataloader
    optimizer = train_config.optimizer
    lr_scheduler = train_config.lr_scheduler
    args = train_config.args
    loss_fn = nn.CrossEntropyLoss()
    min_loss = 999999
    i = 0
    device = train_config.args.device

    model.train()
    
    for i, (x, y) in enumerate(dataloader):
        ddp_loss = torch.zeros(2).to(rank)
        x = x.to(device)
        y = y.to(device)    # [B, S]
        pred = model(x)     # [B, S, vocab]
        loss = loss_fn(pred.flatten(0, 1), y.flatten(0, 1))

        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1
            
        model.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        model.reset()

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(f"batch {i*world_size*args.batch_size}/{len(dataloader)} loss: {ddp_loss[0] / ddp_loss[1]}")

        '''
        if  i % 100 == 0 and ddp_loss[0] / ddp_loss[1] < min_loss and rank == 0:
            dist.barrier()
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler': lr_scheduler.state_dict()}, 'model/model.pth')
            min_loss = ddp_loss[0] / ddp_loss[1]
        '''

def fsdp_main(rank, world_size):
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=bool, default=False)
    command_args = parser.parse_args()

    setup(rank, world_size)
    train_set = EnwikiDataset("enwik8", "char_book.json", split="train")
    train_sampler = DistributedSampler(train_set, rank=rank, num_replicas=world_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=world_size, pin_memory=True)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)

    model = MySpikeGPT().to(rank)
    model = FSDP(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    total_itr = args.epoch * (len(train_loader) // (world_size * args.batch_size) + 1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_itr)
    
    if command_args.resume:
        checkpoint = torch.load('model/model.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    config = TrainConfig(model, train_loader, optimizer, lr_scheduler)

    print(f"Trainig starts! Device: {args.device}")
    for i in range(args.epoch):
        train_one_epoch(config, rank, world_size)
    print("Training complete!")
    cleanup()

if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
             args=([WORLD_SIZE]),
             nprocs=WORLD_SIZE)
