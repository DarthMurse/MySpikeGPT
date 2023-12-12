import os
import torch 
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from spikingjelly.activation_based import functional
import functools
from datasets import load_dataset
from accelerate import Accelerator

#from src.SpikeGPT import GPT, GPTConfig
from src.utils import *

class TrainConfig:
    def __init__(self,
                 model,
                 train_loader,
                 valid_loader,
                 optimizer,
                 lr_scheduler,
                 model_name,
                 model_args=args
                 ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer 
        self.lr_scheduler = lr_scheduler
        self.model_name = model_name
        self.args = model_args 

def train_one_epoch(train_config, resume=False):
    model = train_config.model
    train_loader = train_config.train_loader
    valid_loader = train_config.valid_loader
    optimizer = train_config.optimizer
    lr_scheduler = train_config.lr_scheduler
    model_name = train_config.model_name
    args = train_config.args

    loss_fn = nn.CrossEntropyLoss()
    accelerator = Accelerator()
    model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, valid_loader, lr_scheduler)
    loss_curve = []
    valid_loss_curve = []

    if resume:
        accelerator.load_state(model_name)
    
    for i, (x, y) in enumerate(train_loader):  
        model.train()
        loss = model(x, y)     # [B, S, vocab]
            
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            count = 5
            for j, (vx, vy) in enumerate(valid_loader):
                if j < count:
                    vpred = model(vx)
                    valid_loss += loss_fn(vpred, vy[:, -1])
                else:
                    break
            valid_loss /= count

        print(f"batch {i}/{len(train_loader)} loss: {loss.item()} valid_loss:{valid_loss.item()}")
        loss_curve.append(loss.item())
        valid_loss_curve.append(valid_loss.item())
        
        if  i % 100 == 0 and i != 0:
            accelerator.save_state(model_name+'_multiple'+str(i))
            torch.save(loss_curve, 'model/loss_curve')
            torch.save(valid_loss_curve, 'model/valid_loss_curve')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="ANN")
    command_args = parser.parse_args()
    
    if command_args.model_type == "SNN":
        from src.model import MySpikeGPT
        model_name = 'model/SNN_model'
    elif command_args.model_type == "ANN":
        from src.ANNModel import MySpikeGPT
        model_name = 'model/ANN_model'

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="20B_tokenizer.json")
    train_set = WikitextDataset(tokenizer, split="train")
    valid_set = WikitextDataset(tokenizer, split='valid')
    #train_set = EnwikiDataset(split="train")
    #valid_set = EnwikiDataset(split="valid")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True)
    model = MySpikeGPT()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00)
    total_itr = args.epoch * (len(train_loader) // args.batch_size + 1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_itr)
        
    config = TrainConfig(model, train_loader, valid_loader, optimizer, lr_scheduler, model_name)

    print(f"Trainig starts! Device: {args.device}")
    for i in range(args.epoch):
        train_one_epoch(config, command_args.resume)
    print("Training complete!")

if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    main()
