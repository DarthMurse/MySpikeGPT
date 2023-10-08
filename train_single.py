import torch 
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
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

def train_one_epoch(train_config, epoch_num):
    model = train_config.model
    dataloader = train_config.dataloader
    optimizer = train_config.optimizer
    lr_scheduler = train_config.lr_scheduler
    args = train_config.args
    loss_fn = nn.CrossEntropyLoss()
    min_loss = 999999
    device = train_config.args.device
    scaler = torch.cuda.amp.GradScaler()

    model.train()

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
                        model.reset()
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
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler': lr_scheduler.state_dict()}, 'model/model.pth')
            min_loss = loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=bool, default=False)
    command_args = parser.parse_args()

    train_set = EnwikiDataset("enwik8", "char_book.json", split="train", regenerate=False)
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

    for i in range(args.epoch):
        train_one_epoch(config, i+1)
    print("Training complete!")

if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    main()
