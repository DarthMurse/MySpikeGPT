from torch.utils.data import Dataset
from torch.nn import functional as F
import torch
import os 
from typing import List, Tuple
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk
from tqdm import tqdm

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
        self.split = split
        self.ctx_len = ctx_len
        self.tokenizer = tokenizer
        self.tokens = []
        if self.split == "train":
            if os.path.exists("datasets/lambada.preprocessed.train"):
                self.tokens = torch.load("datasets/lambada.preprocessed.train")
            else:
                for i in tqdm(range(len(self.dataset))):
                    if i == 0:
                        seq = self.dataset[0]['text']
                    else:
                        seq = "<|endoftext|>" + self.dataset[i]['text']
                #print(seq.encode("utf-8"))
                    self.tokens.extend(self.tokenizer.encode(seq))
                torch.save(self.tokens, "datasets/lambada.preprocessed.train")
        print("Loading complete!")

    def __len__(self):
        if self.split == "train":
            return (len(self.tokens) - self.ctx_len - 1)
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        if self.split == "train":
            x = self.tokens[idx: idx+self.ctx_len]
            y = self.tokens[idx+1: idx+self.ctx_len+1]
            x, y = torch.tensor(x), torch.tensor(y)
            return x, y
        else:
            token_seq = self.tokenizer.encode(self.dataset[idx]['text'])
            x = token_seq[:-1]
            y = token_seq[-1]
            x, y = torch.tensor(x), torch.tensor(y).unsqueeze(0)
            length = x.shape[0]
            if length < self.ctx_len:
                x = F.pad(x, (0, self.ctx_len - length), 'constant', 2)
                y = F.pad(y, (0, 1), 'constant', length)
            return x, y

if __name__ == "__main__":
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="lambada.json")
    '''
    dataset = load_from_disk("datasets/lambada/train")
    text = "[SEP]".join(dataset["text"])
    training_corpus = (
        text[i : i + 1000]
        for i in range(0, len(text), 1000)
    )

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "PAD", '[MASK]'])
    #tokenizer.pre_tokenizer = Whitespace()
    #files = [f"datasets/wiki103.{split}" for split in ["test", "train", "valid"]]
    tokenizer.train_from_iterator(training_corpus, trainer)
    tokenizer.save("lambada.json")
    '''
    #train_set = WikitextDataset(tokenizer, name="wiki103", split='train')
    #valid_set = WikitextDataset(tokenizer, name="wiki103", split='valid')
    #test_set = WikitextDataset(tokenizer, name="wiki103", split='test')
    #train_set = EnwikiDataset(split="train")
    #valid_set = EnwikiDataset(split="valid")
    #test_set = EnwikiDataset(split="test")
    
    #print(f"test_set[0]: x and y, length: {test_set[0][0].shape[0]}, ctx_len: {args.ctx_len}")
    #print(test_set[0][0])
    #print(test_set[0][1])

    #train_set = LambadaDataset(tokenizer, split="train")
    #print(train_set[0])
    test_set = LambadaDataset(tokenizer, split="test")
    id = 2
    print(test_set[id])
    print(tokenizer.decode(test_set[id][0]))
    print(tokenizer.decode(test_set[id][1][0]))
