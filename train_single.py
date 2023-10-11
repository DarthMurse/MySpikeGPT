import torch 
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from spikingjelly.activation_based import functional
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

def train_one_epoch(train_config, epoch_num):
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
    
    for i, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch_num}, batch {i*args.batch_size}/{len(dataloader)}"):
        x = x.to(device)
        y = y.to(device)    # [B, S]
        pred = model(x)     # [B, S, vocab]
        loss = loss_fn(pred.flatten(0, 1), y.flatten(0, 1))
            
        model.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        model.reset()
        print(f"  loss: {loss.item()}")

        if  i % 10 == 0 and loss.item() < min_loss:
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler': lr_scheduler.state_dict()}, 'model/model.pth')
            min_loss = loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=bool, default=False)
    command_args = parser.parse_args()

    train_set = EnwikiDataset("enwik8", "char_book.json", split="train")
    model = MySpikeGPT().to(args.device)
    tokenizer = MyTokenizer("char_book.json", args.ctx_len)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    total_itr = args.epoch * (len(train_loader) // args.batch_size + 1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_itr)
    
    if command_args.resume:
        checkpoint = torch.load('model/model.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    config = TrainConfig(model, train_loader, optimizer, lr_scheduler)

    print(f"Trainig starts! Device: {args.device}")
    for i in range(args.epoch):
        train_one_epoch(config, i+1)
    print("Training complete!")

if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    main()
