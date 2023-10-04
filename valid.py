import torch 
from torch import nn 
import torch.nn.functional as F
from src.model import MySpikeGPT 
from src.utils import * 
import math
from tqdm import tqdm

def BPC(model, valid_set, tokenizer, model_args=args):
    model.eval()
    count = 0
    loss = 0
    eps = 1e-5
    for seq_num in tqdm(range(len(valid_set))):
        seq = valid_set[seq_num][0].unsqueeze(0)
        for i in tqdm(range(model_args.ctx_len)):
            if seq[0, i] != tokenizer.pad_id:
                prob = model.forward(seq, i+1)
                if i < model_args.ctx_len - 1:
                    loss += math.log2(prob[0, seq[0, i+1]] + eps)
                count += 1
                model.reset()
            else:
                break
    loss /= count
    return loss 

if __name__ == "__main__":
    model = MySpikeGPT()
    checkpoint = torch.load("model/model.pth")
    model.load_state_dict(checkpoint['model'])
    valid_set = EnwikiDataset("enwik8", "char_book.json", split="valid")
    tokenizer = MyTokenizer("char_book.json", args.ctx_len)
    loss = BPC(model, valid_set, tokenizer)
    print(loss)
