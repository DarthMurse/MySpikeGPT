import torch 
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
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

def train_one_epoch(train_config):
    pass

def main():
    dataset = EnwikiDataset("enwik8", "char_book.json", split="test", regenerate=False)
    model = MySpikeGPT()
    tokenizer = MyTokenizer("char_book.json", args.ctx_len)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True)
    output = model.forward(dataset[0][0].unsqueeze(0), 1)
    #loss = criterion(output, dataset[0][1])

if __name__ == "__main__":
    main()
