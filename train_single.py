import torch 
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from spikingjelly.activation_based import functional

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
        self.args = model_args 
        self.model_name = model_name

def train_one_epoch(train_config, epoch_num):
    model = train_config.model
    train_loader = train_config.train_loader
    valid_loader = train_config.valid_loader
    optimizer = train_config.optimizer
    lr_scheduler = train_config.lr_scheduler
    model_name = train_config.model_name
    args = train_config.args
    loss_fn = nn.CrossEntropyLoss()
    i = 0
    device = train_config.args.device
    loss_curve = []
    valid_loss_curve = []
    
    for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch_num}, batch {i}/{len(train_loader)}"):
        model.train() 
        x = x.to(device)
        y = y.to(device)    # [B, S]
        pred = model(x)     # [B, S, vocab]
        loss = loss_fn(pred.flatten(0, 1), y.flatten(0, 1))
            
        model.zero_grad()
        loss.backward()
        #for name, param in model.named_parameters():
        #    if param.grad is None:
        #        print(name)

        optimizer.step()
        lr_scheduler.step()
        if model_name == 'model/model.pth':
            model.reset()

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            count = 1
            for j, (vx, vy) in enumerate(valid_loader):
                if j < count:
                    vx, vy = vx.to(device), vy.to(device)
                    vpred = model(vx)
                    valid_loss += loss_fn(vpred.flatten(0, 1), vy.flatten(0, 1))
                else:
                    break
            valid_loss /= count

        print(f" loss: {loss.item()} valid_loss:{valid_loss.item()}")
        
        if i % 500 == 0:
            loss_curve.append(loss.item())
            valid_loss_curve.append(valid_loss.item())

        if  i % 5000 == 0 and i != 0:
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'scheduler': lr_scheduler.state_dict()}, model_name+str(i))
            torch.save(loss_curve, 'model/loss_curve_ann_to_convert')
            torch.save(valid_loss_curve, 'model/valid_loss_curve_ann_to_convert')

        #if i % 100 == 0:
        #    for param in model.parameters():
        #        print(f"param grad mean: {param.grad.flatten().mean()}, std: {param.grad.flatten().std()}")
        #        print(param.shape)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="SNN")
    command_args = parser.parse_args()

    if command_args.model_type == "SNN":
        from src.model import MySpikeGPT
        model_name = "model/model.pth"
    elif command_args.model_type == "ANN":
        from src.ANNModel import MySpikeGPT
        model_name = "model/ANN_model_single"

    model = MySpikeGPT().to(args.device)
    #model = model = GPT(GPTConfig(args.vocab_size, args.ctx_len, model_type='RWKV',
    #                      n_layer=args.n_layers, n_embd=args.embed)).to(args.device)
    #print(model.parameters)
    #for param in model.parameters():
    #    print(param.shape)
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')
    train_set = WikitextDataset(tokenizer, split='train')
    valid_set = WikitextDataset(tokenizer, split='valid')
    
    #train_set = EnwikiDataset(split="train")
    #valid_set = EnwikiDataset(split="valid")

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, 1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    total_itr = args.epoch * (len(train_loader) // args.batch_size + 1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_itr)
    
    if command_args.resume:
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    config = TrainConfig(model, train_loader, valid_loader, optimizer, lr_scheduler, model_name)

    print(f"Trainig starts! Device: {args.device}")
    for i in range(args.epoch):
        train_one_epoch(config, i+1)
    print("Training complete!")

if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    main()
