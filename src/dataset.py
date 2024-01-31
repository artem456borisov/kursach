import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Any

device = 'cuda'

class MERA_dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, kwargs: Dict[str, Any]):
        self.df = df
        self.prompt = df['instruction']
        self.inputs = df['inputs']
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.label, label_names = pd.factorize(np.array(df['outputs']))
        
        
    def __getitem__(self, idx):
        prompt = self.prompt[idx].format(**self.inputs[idx])
        label = self.label[idx]
        prompt = self.tokenizer(prompt, **self.kwargs).to(device)
        prompt = {k:v.flatten() for k,v in prompt.items()}
        label = torch.tensor([label]).to(device)
        item = {**prompt}
        return item
    def __len__(self):
        return len(self.df)
    
class Collator():
    def __init__(self, tokenizer, tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
    def __call__(self, batch):
        q, asw = zip(*batch)
        q = self.tokenizer(q, **self.tokenizer_kwargs)
        asw = torch.Tensor(asw)
        return q, asw
        