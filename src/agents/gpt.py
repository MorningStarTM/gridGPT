import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from src.utils.logger import logger
from torch.distributions import Categorical
from src.utils.converter import ActionConverter, MADiscActionConverter




class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def __len__(self):
        return len(self.rewards)


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config['n_embd'], config['head_size'], bias=False)
        self.query = nn.Linear(config['n_embd'], config['head_size'], bias=False)
        self.value = nn.Linear(config['n_embd'], config['head_size'], bias=False)
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
        self.heads = nn.ModuleList([Head(config) for _ in range(config['n_head'])])
        self.proj = nn.Linear(config['head_size'] * config['n_head'], config['n_embd'])
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
        self.selfAttention = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
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
            'action_size': 178,
            'block_size': 64,
            'state_dim': 10,
            'n_embd': 128,
            'fusion_embed_dim': 3 * 128,  # 3 * n_embd
            'n_head': 4,
            'head_size': 32,
            'n_layers': 4,
            'dropout': 0.1,
            'context_len': 16,       # window length L (slot indices 0..L-1)
            'max_timestep': 10000,    # max absolute env step used for embeddings
        }

        # ===== embeddings =====
        self.prev_state_embedding = nn.Linear(self.config['state_dim'], self.config['n_embd'])
        self.action_embedding     = nn.Embedding(self.config['action_size'], self.config['n_embd'])
        self.next_state_embedding = nn.Linear(self.config['state_dim'], self.config['n_embd'])
        self.trajectory_embedding = nn.Linear(self.config['fusion_embed_dim'], self.config['n_embd'])

        # ===== HYBRID POSITIONAL ENCODING (NEW) =====
        # learned slot embedding (relative position inside the window) and absolute time embedding
        self.idx_embedding  = nn.Embedding(self.config['context_len'], self.config['n_embd'])
        self.time_embedding = nn.Embedding(self.config['max_timestep'], self.config['n_embd'])
        # FiLM-style scale/shift produced from the positional vector
        self.trajectory_positional_embedding = nn.Linear(self.config['n_embd'], self.config['n_embd'] * 2)

        # heads 
        self.ln_f   = nn.LayerNorm(self.config['n_embd'])
        self.lm_head= nn.Linear(self.config['n_embd'], self.config['action_size'])
        self.value_head = nn.Linear(self.config['n_embd'], 1)

        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config['n_layers'])])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, prev_state, action, next_state, slot_idx=None, timestep=None, action_mask_last=None, return_value=False):
        """
        Args:
            prev_state: [B, state_dim]    previous state S_{t-1}
            action:     [B] or [B,1]      previous action A_{t-1} (Long)
            next_state: [B, state_dim]    current state S_t
            slot_idx:   [B] (Long, 0..L-1) OPTIONAL window slot index for this token
            timestep:   [B] (Long, 0..max_timestep-1) OPTIONAL absolute env step

        Returns:
            logits: [B, action_size]  action logits for A_t
        """
        # content embeddings → fuse into a token
        B, L, _ = prev_state.shape

        # 1) content embeddings per slot
        e_prev = self.prev_state_embedding(prev_state)        # [B, L, n_embd]
        e_act  = self.action_embedding(action)           # [B, L, n_embd]
        e_next = self.next_state_embedding(next_state)        # [B, L, n_embd]

        fused  = torch.cat([e_prev, e_act, e_next], dim=-1)    # [B, L, 3*n_embd]
        tokens = self.trajectory_embedding(fused)              # [B, L, n_embd]  == t_k

        # 2) hybrid positions per slot and add
        pos = self.idx_embedding(slot_idx) + self.time_embedding(timestep)  # [B, L, n_embd]
        x = tokens + pos                                                     # [B, L, n_embd]

        # 3) transformer over time (now T=L, not 1)
        x = self.blocks(x)                           # [B, L, n_embd]
        h_last = x[:, -1, :]                         # [B, n_embd] decision slot

        # 4) norm → head at last slot
        h_last = self.ln_f(h_last)                   # [B, n_embd]
        logits = self.lm_head(h_last)                # [B, action_size]
        if action_mask_last is not None:
            logits = logits.masked_fill(~action_mask_last.bool(), float('-inf'))
        
        if not return_value:
            return logits
        
        value = self.value_head(h_last).squeeze(-1)  # [B]
        return logits, value


    def save(self, checkpoint_path, filename="gpt_checkpoint.pth"):
        if checkpoint_path:
            os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        save_path = os.path.join(checkpoint_path, filename)
        torch.save(checkpoint, save_path)
        logger.info(f"[SAVE] Checkpoint saved to {save_path}")


    def load(self, checkpoint_path, filename="gpt_checkpoint.pth"):
        file = os.path.join(checkpoint_path, filename)
        checkpoint = torch.load(file, map_location=self.device)

        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"[LOAD] Checkpoint loaded from {file}")
        return True
    

    def act(self,
            prev_states,     # [B,L,state_dim]
            prev_actions,    # [B,L] (Long)
            next_states,     # [B,L,state_dim]
            slot_idx,        # [B,L] (Long 0..L-1)
            timestep,        # [B,L] (Long)
            action_mask_last=None,   # [B, action_size] or None
            deterministic: bool=False):
        logits, value = self.forward(
            prev_states, prev_actions, next_states,
            slot_idx=slot_idx, timestep=timestep,
            action_mask_last=action_mask_last,
            return_value=True
        )  # logits:[B,A], value:[B]

        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)            # [B]
            action_logprob = dist.log_prob(action)    # [B]
        else:
            action = dist.sample()                    # [B]
            action_logprob = dist.log_prob(action)    # [B]

        return action, action_logprob, value


    

class gridGPTAgent:
    def __init__(self, env, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.has_continuous_action_space = self.config['has_continuous_action_space']

        self.ac = ActionConverter(env)
        action_dim = self.ac.n
        logger.info(f"Using ActionConverter with action size: {action_dim}")


        self.gamma = self.config['gamma']
        self.eps_clip = self.config['eps_clip']
        self.K_epochs = self.config['K_epochs']
        
        self.buffer = RolloutBuffer()

        self.policy     = gridGPT().to(self.device)
        self.policy_old = gridGPT().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.AdamW(self.policy.parameters(), lr=self.config['lr_actor'])

        self.MseLoss = nn.MSELoss()


    def select_action(self, prev_state, prev_action, next_state,
                      slot_idx=None, timestep=None, action_mask_last=None, deterministic=False):
        # move inputs to device and correct dtypes
        prev_state  = prev_state.to(self.device)
        next_state  = next_state.to(self.device)
        prev_action = prev_action.to(self.device).long()
        if slot_idx is not None:   slot_idx   = slot_idx.to(self.device).long()
        if timestep is not None:   timestep   = timestep.to(self.device).long()
        if action_mask_last is not None:
            action_mask_last = action_mask_last.to(self.device).bool()

        with torch.no_grad():
            # assumes you added gridGPT.act(...) returning (action, logprob, value)
            action, action_logprob, state_val = self.policy_old.act(
                prev_state, prev_action, next_state,
                slot_idx, timestep,
                action_mask_last=action_mask_last,
                deterministic=deterministic
            )

        action_id = int(action[0].item()) if action.dim() else int(action.item())
        grid_action = self.ac.act(action_id)
        return action_id, grid_action, action_logprob, state_val