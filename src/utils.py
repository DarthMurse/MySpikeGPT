from torch.utils.data import Dataset
import torch
import os 
from typing import List, Tuple

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

    # Decode List[int] to str
    def decode(self, tokens: torch.Tensor):
        result = ""
        for i in range(len(tokens)):
            result = result + self.decode_dict[tokens[i]]
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

if __name__ == "__main__":
    train_set = EnwikiDataset(split="train")
    valid_set = EnwikiDataset(split="valid")
    test_set = EnwikiDataset(split="test")
    print(f"test_set[0]: x and y, length: {test_set[0][0].shape[0]}, ctx_len: {args.ctx_len}")
    print(test_set[0][0])
    print(test_set[0][1])
