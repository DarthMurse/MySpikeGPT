from torch.utils.data import Dataset
import torch
from transformers import PreTrainedTokenizerFast
import os 
from typing import List, Tuple

from .args import *

def get_tokenizer(file_name):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=file_name)
    tokenizer.pad_token = "<|padding|>"
    tokenizer.eos_token = "<|endoftext|>"
    return tokenizer

class TextDataset(Dataset):
    def __init__(self, 
                 dataset_name: str,
                 tokenizer_name: str, 
                 ctx_len: int = args.ctx_len,
                 regenerate: bool = True
                 ): 
        super().__init__()
        self.name = dataset_name
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_name)
        self.tokenizer.pad_token = "<|padding|>"
        self.tokenizer.eos_token = "<|endoftext|>"
        self.max_len = ctx_len
        self.root_path = "datasets/"

        if os.path.exists(self.root_path+dataset_name+".tokenized") and not regenerate:
            print("Using preprocessed dataset ..."+self.name)
            self.tokens = torch.load(self.root_path+dataset_name+".tokenized")
            self.input_ids = torch.tensor(self.tokens['input_ids'])
        else:
            with open(self.root_path+self.name, 'r') as f:
                print("Loading original dataset ... "+self.name)
                text = f.readlines()

            self.text = []

            for line in text:
                if len(line) <= 100:
                    continue 
                self.text.append(line+"<|endoftext|>")

            self.tokens = self.tokenizer(self.text, padding='max_length', truncation=True, max_length=ctx_len)
            self.input_ids = torch.tensor(self.tokens['input_ids'])
            torch.save(self.tokens, self.root_path+dataset_name+".tokenized")

    def __len__(self): 
        return len(self.tokens)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], torch.cat((self.input_ids[idx][1:], torch.tensor([1])))

class MyTokenizer:
    def __init__(self, json_file, ctx_len):
        with open(json_file, 'r') as f:
            dic = json.load(f)
        self.encode_dict = dict()
        self.decode_dict = dict()
        self.unk_id = 78
        self.pad_id = 77
        for key in dic.keys():
            self.encode_dict[dic[key]] = int(key) 
            self.decode_dict[int(key)] = dic[key]
        self.max_len = ctx_len

    def encode(self, text: str):
        result = []
        text = text.lower()
        output = torch.zeros([self.max_len])
        for char in text:
            if char in self.encode_dict.keys():
                result.append(self.encode_dict[char])
            else:
                result.append(self.unk_id)
        if len(result) <= self.max_len:
            output[:len(result)] = torch.tensor(result)
            output[len(result):] = self.pad_id
        else:
            output = torch.tensor(result)[:self.max_len]
        return output.to(torch.int32)

    def decode(self, tokens: torch.Tensor):
        result = ""
        for i in range(tokens.shape[0]):
            if (tokens[i] != self.pad_id):
                result += self.decode_dict[tokens[i].item()]
            else:
                break
        return result

    def batch_encode(self, text: List[str]):
        n = len(text)
        result = torch.zeros([n, self.max_len])
        for i in range(n):
            result[i] = self.encode(text[i])
        return result.to(torch.int32)

    def batch_decode(self, text: torch.Tensor):
        result = []
        for i in range(text.shape[0]):
            result.append(self.decode(text[i]))
        return result

class EnwikiDataset(Dataset):
    def __init__(self, 
                 dataset_name: str = "enwik8",
                 tokenizer_name: str = "char_book.json", 
                 ctx_len: int = args.ctx_len,
                 split: str = "train",
                 regenerate: bool = False
                 ): 
        super().__init__()
        self.name = dataset_name
        self.tokenizer = MyTokenizer(tokenizer_name, ctx_len)
        self.max_len = ctx_len
        self.root_path = "datasets/"

        if os.path.exists(self.root_path+dataset_name+"."+split+".tokenized") and not regenerate:
            print("Using preprocessed dataset ..."+self.name+"."+split)
            self.tokens = torch.load(self.root_path+dataset_name+"."+split+".tokenized")
        else:
            with open(self.root_path+self.name, 'r') as f:
                print("Loading original dataset ... "+self.name+"."+split)
                text = f.readlines()

            self.text = []

            for line in text:
                if line.isspace():
                    continue 
                self.text.append(line+"<|end|>")

            n = len(self.text)
            if split == "train":
                text = text[:int(0.9*n)]
            elif split == "valid":
                text = text[int(0.9*n):int(0.95*n)]
            else:
                text = text[int(0.95*n):]

            self.tokens = self.tokenizer.batch_encode(self.text)
            self.input_ids = torch.tensor(self.tokens)
            torch.save(self.tokens, self.root_path+dataset_name+"."+split+".tokenized")

    def __len__(self): 
        return len(self.tokens)

    def __getitem__(self, idx: int):
        return self.tokens[idx], torch.cat((self.tokens[idx][1:], torch.tensor([self.tokenizer.pad_id])))

if __name__ == "__main__":
    dataset = TextDataset('datasets/wiki.train.tokens', '20B_tokenizer.json', 512)
