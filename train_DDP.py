import torch 
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os
import subprocess
from tqdm import tqdm

from src.model import MySpikeGPT 
from src.utils import *

PAD_ID = 77
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

def train_one_epoch(train_config, epoch_num, mloss=999999):
    model = train_config.model
    dataloader = train_config.dataloader
    optimizer = train_config.optimizer
    lr_scheduler = train_config.lr_scheduler
    args = train_config.args
    loss_fn = nn.CrossEntropyLoss()
    min_loss = mloss
    device = train_config.args.device
    scaler = torch.cuda.amp.GradScaler()

    for i, (x, y) in enumerate(dataloader):
        batch_size = x.shape[0]
        x = x.to(device)
        y = y.to(device)

        for k in range(batch_size):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = []
                for j in tqdm(range(args.ctx_len), desc=f"Epoch {epoch_num}, batch {i*args.batch_size}/{len(dataloader)}"):
                    if x[k, j] != PAD_ID:
                        pred.append(model.forward(x[k], j+1).unsqueeze(0))
                        functional.reset_net(model.module)
            if len(pred) != 0:
                pred = torch.concat(pred, dim=0)
                n = pred.shape[0]
                loss = loss_fn(pred, y[k, :n])
                model.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                print(f"Epoch {epoch_num}, batch {i*args.batch_size}/{len(dataloader)}, loss: {loss}")

        if  i % 10 == 0 and loss < min_loss:
            torch.save({'model': model.module.state_dict(), 'optim': optimizer.state_dict(), 'scheduler': lr_scheduler.state_dict(), 'loss': min_loss.item()}, 'model/model.pth')
            min_loss = loss
            print(f"Checkpoint saved with loss: {min_loss.item()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=False, type=bool)
    command_arg = parser.parse_args()

    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    if "MASTER_PORT" in os.environ:
        pass 
    else:
        os.environ["MASTER_PORT"] = "29500"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
    os.environ["RANK"] = str(proc_id)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)

    train_set = EnwikiDataset("enwik8", "char_book.json", split="train", regenerate=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    model = MySpikeGPT().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    tokenizer = MyTokenizer("char_book.json", args.ctx_len)
    train_loader = DataLoader(train_set, 4*args.batch_size, sampler=train_sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    total_itr = args.epoch * (len(train_loader) // args.batch_size + 1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_itr)
    
    min_loss = 9999999
    if command_arg.resume:
        checkpoint = torch.load("model/model.pth")
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        min_loss = checkpoint['loss']
    config = TrainConfig(model, train_loader, optimizer, lr_scheduler)

    for i in range(args.epoch):
        config.dataloader.sampler.set_epoch(i)
        train_one_epoch(config, i+1, min_loss)
    print("Training complete!")

if __name__ == "__main__":
    main()
