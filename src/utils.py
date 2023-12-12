from torch.utils.data import Dataset
import torch
import os 
from typing import List, Tuple
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk

from .args import *

class MyTokenizer:
    def __init__(self, json_file, ctx_len):
        with open(json_file, 'r') as f:
            dic = json.load(f)
        self.encode_dict = dict()
        self.decode_dict = dict()
        self.unk_id = 77 # Unknown character

        # Load character to token dictionary
        for key in dic.keys():
            self.encode_dict[dic[key]] = int(key) 
            self.decode_dict[int(key)] = dic[key]
        self.max_len = ctx_len

    # Encode str to List[int]
    def encode(self, text: str):
        output = []
        for char in text:
            if 'A' <= char <= 'Z':
                output.append(self.encode_dict[char.lower()])   # There is no capital letters in the dictionary
            elif char not in self.encode_dict.keys():
                output.append(self.unk_id)
            else:
                output.append(self.encode_dict[char])
        output = torch.tensor(output)
        return output.to(torch.long)

    # Decode torch.tensor to str
    def decode(self, tokens: torch.Tensor):
        result = ""
        for i in range(len(tokens)):
            result = result + self.decode_dict[tokens[i].item()]
        return result

class EnwikiDataset(Dataset):
    def __init__(self, 
                 dataset_name: str = "enwik8",
                 tokenizer_name: str = "char_book.json", 
                 ctx_len: int = args.ctx_len,
                 split: str = "train",
                 ): 
        super().__init__()
        self.name = dataset_name
        self.tokenizer = MyTokenizer(tokenizer_name, ctx_len)
        self.max_len = ctx_len
        self.root_path = "datasets/"

        # Using splitted datafile 
        if os.path.exists(self.root_path+self.name+"."+split):
            with open(self.root_path+self.name+'.'+split, 'r') as f:
                print("Using preprocessed dataset ..."+self.name+"."+split)
                self.text = f.read()
        # Make splitted datafile
        else:
            with open(self.root_path+self.name, 'r') as f:
                print("Loading original dataset ... "+self.name+"."+split)
                self.text = f.read()

            n = len(self.text)
            if split == 'train':
                self.text = self.text[: int(0.9 * n)]
            elif split == 'valid':
                self.text = self.text[int(0.9 * n): int(0.95 * n)]
            elif split == 'test':
                self.text = self.text[int(0.95 * n): ]
            else:
                print("Invalid split name {split}! Please check again.")
            # Save to disk
            with open(self.root_path+self.name+'.'+split, 'w') as f:
                f.write(self.text)

    def __len__(self): 
        return (len(self.text) - self.max_len - 1)

    def __getitem__(self, idx: int):
        x_text = self.text[idx: idx + self.max_len]
        y_text = self.text[idx + 1: idx + self.max_len + 1]
        x = self.tokenizer.encode(x_text)
        y = self.tokenizer.encode(y_text)
        return x, y

class WikitextDataset(Dataset):
    def __init__(self, tokenizer, name="wiki", ctx_len=args.ctx_len, split="train"):
        self.max_len = ctx_len
        self.tokenizer = tokenizer

        if os.path.exists('datasets/'+name+'.preprocessed.'+split):
            print("Using preprocessed dataset ..."+name+"."+split)
            self.tokens = torch.load('datasets/'+name+'.preprocessed.'+split)
        else:
            print("Loading original dataset ... "+name+"."+split)
            with open('datasets/'+name+'.'+split, 'r') as f:
                text = f.read()
            for i in range(len(text)):
                if text[i] == ' ' and i+1 < len(text) and text[i+1] == '\n' and i >= 1 and text[i-1] == '\n':
                    text = text[:i] + '<|endoftext|>' + text[i+1:]
            self.tokens = self.tokenizer.encode(text)
            torch.save(self.tokens, 'datasets/'+name+'.preprocessed.'+split)

    def __len__(self):
        return (len(self.tokens) - self.max_len - 1)
    
    def __getitem__(self, idx):
        x = self.tokens[idx: idx+self.max_len]
        y = self.tokens[idx+1: idx+self.max_len+1]
        x, y = torch.tensor(x), torch.tensor(y)
        return x, y

class LambadaDataset(Dataset):
    def __init__(self, tokenizer, ctx_len=args.ctx_len, split="train"):
        self.dataset = load_from_disk("datasets/lambada/"+split)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

if __name__ == "__main__":
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="20B_tokenizer.json")

    #train_set = WikitextDataset(tokenizer, split='train')
    #valid_set = WikitextDataset(tokenizer, split='valid')
    #test_set = WikitextDataset(tokenizer, split='test')
    train_set = EnwikiDataset(split="train")
    valid_set = EnwikiDataset(split="valid")
    test_set = EnwikiDataset(split="test")
    
    #print(f"test_set[0]: x and y, length: {test_set[0][0].shape[0]}, ctx_len: {args.ctx_len}")
    #print(test_set[0][0])
    #print(test_set[0][1])

    #train_set = LambadaDataset("validation")
