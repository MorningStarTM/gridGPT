import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np






class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config['n_emb'], config['head_size'], bias=False)
        self.query = nn.Linear(config['n_emb'], config['head_size'], bias=False)
        self.value = nn.Linear(config['n_emb'], config['head_size'], bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config['block_size'], config['block_size'])))
        self.dropout = nn.Dropout(config['dropout'])


    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5   # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out




class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config['head_size']) for _ in range(config['num_heads'])])
        self.proj = nn.Linear(config['head_size'] * config['num_heads'], config['n_emb'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']),
            nn.ReLU(),
            nn.Linear(4 * config['n_embd'], config['n_embd']),
            nn.Dropout(config['dropout'])
        )

    def forward(self, x):
        return self.net(x)
    


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config['n_embd'] // config['n_head']
        self.selfAttention = MultiHeadAttention(config['n_head'], head_size)
        self.ffwd = FeedForward(config['n_embd'])
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])

    def forward(self, x):
        y = self.selfAttention(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
    

