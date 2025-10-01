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
    


class gridGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {
            'vocab_size': 256,
            'block_size': 64,
            'n_embd': 128,
            'n_head': 4,
            'head_size': 32,
            'num_layers': 4,
            'dropout': 0.1
        }

        self.prev_state_embedding = nn.Linear(self.config['state_dim'], self.config['n_embd'])
        self.action_embedding = nn.Embedding(self.config['action_size'], self.config['n_embd'])
        self.next_state_embedding = nn.Linear(self.config['state_dim'], self.config['n_embd'])
        self.trajectory_embedding = nn.Linear(self.config['fusion_embed_dim'], self.config['n_embd'])

        self.trajectory_positional_embedding = nn.Linear(self.config['n_embd'], self.config['n_embd']*2)

        self.ln_f = nn.LayerNorm(self.config['n_embd'])
        self.lm_head = nn.Linear(self.config['n_embd'], self.config['action_size'])

        self.apply(self._init_weights)

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self, prev_state, action, next_state):
        """
        Args:
            prev_state:  [B, state_dim]   - previous state features
            action:      [B] or [B, 1]    - previous action (LongTensor indices)
            next_state:  [B, state_dim]   - current state features (S_t)

        Returns:
            logits: [B, action_size]  - action logits for the NEXT action (A_t)
        """
        # 1) per-part embeddings → [B, n_embd] each
        e_prev = self.prev_state_embedding(prev_state)      # [B, n_embd]
        e_act  = self.action_embedding(action.squeeze(-1) if action.ndim == 2 else action)  # [B, n_embd]
        e_next = self.next_state_embedding(next_state)      # [B, n_embd]

        # 2) fuse by concatenation → [B, 3*n_embd], then project to [B, n_embd]
        fused_in = torch.cat([e_prev, e_act, e_next], dim=-1)  # [B, 3*n_embd]
        # sanity check: the fusion input must match trajectory_embedding.in_features
        if fused_in.size(-1) != self.trajectory_embedding.in_features:
            raise ValueError(
                f"fusion_embed_dim mismatch: got {fused_in.size(-1)} but "
                f"trajectory_embedding expects {self.trajectory_embedding.in_features}. "
                f"Set config['fusion_embed_dim'] = 3 * n_embd or adjust your embeddings."
            )
        token = self.trajectory_embedding(fused_in)  # [B, n_embd]

        # 3) lightweight mixing using your 'trajectory_positional_embedding' (produces 2*n_embd)
        #    interpret as FiLM-style scale/shift on the fused token
        mix = self.trajectory_positional_embedding(token)   # [B, 2*n_embd]
        scale, shift = mix.chunk(2, dim=-1)                 # each [B, n_embd]
        token = token * torch.sigmoid(scale) + shift        # [B, n_embd]

        # 4) norm → head
        token = self.ln_f(token)                            # [B, n_embd]
        logits = self.lm_head(token)                        # [B, action_size]

        return logits

