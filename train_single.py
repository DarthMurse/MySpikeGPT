import torch 
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
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

def train_one_epoch(train_config, epoch_num):
    model = train_config.model
    dataloader = train_config.dataloader
    optimizer = train_config.optimizer
    lr_scheduler = train_config.lr_scheduler
    args = train_config.args
    loss_fn = nn.CrossEntropyLoss()
    min_loss = 999999

    for i, (x, y) in enumerate(dataloader):
        loss = 0
        batch_size = x.shape[0]
        for j in tqdm(range(args.ctx_len), desc=f"Epoch {epoch_num}, batch {i*args.batch_size}/{len(dataloader)}"):
            for k in range(batch_size):
                if x[k, i] != PAD_ID:
                    pred = model.forward(x[k].unsqueeze(0), j+1)
                    loss += loss_fn(pred[0], y[k, j])
                    model.reset()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        print(f"Epoch {epoch_num}, batch {i*args.batch_size}/{len(dataloader)}, loss: {loss}")
        if  i % 10 == 0 and loss < min_loss:
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler': lr_scheduler.state_dict()}, 'model/model.pth')
            min_loss = loss

def main():
    train_set = EnwikiDataset("enwik8", "char_book.json", split="train", regenerate=False)
    model = MySpikeGPT()
    tokenizer = MyTokenizer("char_book.json", args.ctx_len)
    train_loader = DataLoader(train_set, 4*args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    total_itr = args.epoch * (len(train_loader) // args.batch_size + 1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_itr)
    config = TrainConfig(model, train_loader, optimizer, lr_scheduler)

    for i in range(args.epoch):
        train_one_epoch(config, i+1)
    print("Training complete!")

if __name__ == "__main__":
    main()
