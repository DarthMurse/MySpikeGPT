import torch 
from torch import nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys

from src.utils import * 
import math
from tqdm import tqdm
import argparse

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def BPC(model, valid_set, tokenizer, model_args=args):
    model.eval()
    valid_loader = DataLoader(valid_set, batch_size=model_args.batch_size, shuffle=True)
    total_count = 100
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    temperature = 1.0
    top_p = 0.9

    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(valid_loader), total=total_count, desc=f"total_count: {total_count}, current_loss: {total_loss}"):
            if  i >= total_count:
                break 
        
            x, y = x.to(model_args.device), y.to(model_args.device)
            pred = model(x)
            total_loss += loss_fn(pred, y).item()

        loss = total_loss / total_count
        print(f"Average loss: {loss}")

        print("Now print a sentence using temperature top-p sampling: ")
        initial_input = valid_set[2000][0].to(model_args.device)
        print("input: " + tokenizer.decode(initial_input))
        print('--------------------------------------------------------------')
        output = initial_input
        idx = 1
        i = 0
        while idx != 0:
            pred = model(initial_input.unsqueeze(0))
            pred = pred[0]
            #print(pred)
            pred = torch.softmax(pred / temperature, dim=-1)
            idx = sample_top_p(pred, top_p)
            output = torch.cat((output, torch.tensor([idx], dtype=torch.long, device=model_args.device)))
            initial_input = output[i+1:]
            i += 1
        print("output: " + tokenizer.decode(output))

    return loss 

if __name__ == "__main__":
    model_args = args

    from src.ANNModel import MySpikeGPT
    model_name = 'ANN_models/'+sys.argv[1]+'/model.pth'

    model = MySpikeGPT(model_args).to(model_args.device)
    checkpoint = torch.load(model_name, map_location='cpu')
    model.load_state_dict(checkpoint)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')
    valid_set = WikitextDataset(tokenizer, split="test")
    #tokenizer = MyTokenizer("char_book.json", model_args.ctx_len)
    #valid_set = EnwikiDataset(split="test")
    loss = BPC(model, valid_set, tokenizer)
