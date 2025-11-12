import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from collections import deque
from src.utils.logger import logger
from torch.distributions import Categorical, kl_divergence
from src.utils.converter import ActionConverter, MADiscActionConverter
from src.utils.reply_buffer import ReplayBuffer




class RolloutBuffer:
    def __init__(self):
        self.seq_prev_states   = []
        self.seq_actions       = []
        self.seq_next_states   = []
        self.seq_slot_idx      = []
        self.seq_timesteps     = []


        self.actions           = []
        self.teacher_probs     = []
        self.states            = []
        self.logprobs          = []
        self.rewards           = []
        self.state_values      = []
        self.is_terminals      = []

        self.last_states       = []   # for teacher on s_t only (next_states[..., -1, :])
    
    def clear(self):
        del self.seq_prev_states[:]
        del self.seq_actions[:]
        del self.seq_next_states[:]
        del self.seq_slot_idx[:]
        del self.seq_timesteps[:]

        del self.actions[:]
        del self.teacher_probs[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.last_states[:]



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
            'state_dim': 493,
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


    def save(self, optimizer, checkpoint_path, filename="gpt_checkpoint.pth"):
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


    def load(self, optimizer, checkpoint_path, filename="gpt_checkpoint.pth"):
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

        return action, action_logprob, value, logits
    

    def evaluate(self,
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
            dist_entropy = dist.entropy()
        else:
            action = dist.sample()                    # [B]
            action_logprob = dist.log_prob(action)    # [B]
            dist_entropy = dist.entropy()
        
        return action_logprob, value, dist_entropy, logits


    


class gridGPTAC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        self.logprobs = []
        self.student_logits = []
        self.state_values = []
        self.rewards = []

        self.kl_coef = 0.1                 # weight for distillation term
        self.distill_temperature = 1.0

        self.optimizer = optim.Adam(self.parameters(), lr=self.config['lr'], betas=self.config['betas'])
        self.to(self.device)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, prev_state, action, next_state, slot_idx=None, timestep=None, action_mask_last=None, return_value=False):
        # 0) sanitize inputs
        prev_state = self._safe(prev_state, "prev_state")
        next_state = self._safe(next_state, "next_state")

        B, L, S = prev_state.shape
        assert next_state.shape == (B, L, S)
        assert action.shape == (B, L)

        # 1) index guards for embeddings
        self._check_idx(action, self.config['action_size'], "action")
        self._check_idx(slot_idx, self.config['context_len'], "slot_idx")
        self._check_idx(timestep, self.config['max_timestep'], "timestep")

        # 2) content embeddings
        e_prev = self.prev_state_embedding(prev_state)        # [B,L,E]
        e_act  = self.action_embedding(action)                # [B,L,E]
        e_next = self.next_state_embedding(next_state)        # [B,L,E]
        e_prev = self._safe(e_prev, "e_prev")
        e_act  = self._safe(e_act,  "e_act")
        e_next = self._safe(e_next, "e_next")

        fused  = torch.cat([e_prev, e_act, e_next], dim=-1)   # [B,L,3E]
        tokens = self.trajectory_embedding(fused)             # [B,L,E]
        tokens = self._safe(tokens, "tokens")

        # 3) hybrid positions
        pos = self.idx_embedding(slot_idx) + self.time_embedding(timestep)  # [B,L,E]
        pos = self._safe(pos, "pos")
        x = tokens + pos
        x = self._safe(x, "x(tokens+pos)")

        # 4) transformer
        x = self.blocks(x)                                    # [B,L,E]
        x = self._safe(x, "x(blocks)")
        h_last = x[:, -1, :]                                  # [B,E]

        # 5) heads
        h_last = self.ln_f(h_last)
        h_last = self._safe(h_last, "h_last(ln_f)")

        logits = self.lm_head(h_last)                         # [B,A]
        logits = self._safe(logits, "logits(pre-mask)")

        if action_mask_last is not None:
            logits = logits.masked_fill(~action_mask_last.bool(), float('-inf'))

        # If everything got masked or something went off, rescue to finite logits
        if not torch.isfinite(logits).all() or torch.isneginf(logits).all():
            # Optional one-time noisy log
            # print("[RESCUE] logits had non-finite or all -inf; replacing with zeros.")
            logits = torch.zeros_like(logits)

        if not return_value:
            return logits

        value = self.value_head(h_last).squeeze(-1)           # [B]
        value = self._safe(value, "value")

        dist = Categorical(logits=logits)                     # now guaranteed finite
        action = dist.sample()

        self.logprobs.append(dist.log_prob(action))
        self.state_values.append(value)
        self.student_logits.append(logits.detach())           # keep detached logits

        return action.item()
    


    def calculateLossICM(self, gamma=0.99, value_coef=0.5, entropy_coef=0.01):
        # discounted returns
        returns = []
        g = 0.0
        for r in reversed(self.rewards):
            g = r + gamma * g
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
    
        # stabilize return normalization
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    
        values = torch.stack(self.state_values).to(self.device).squeeze(-1)
        logprobs = torch.stack(self.logprobs).to(self.device)
    
        advantages = returns - values.detach()
        # advantage normalization helps a LOT with small/medium LR
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    
        policy_loss = -(logprobs * advantages).mean()
        value_loss  = F.smooth_l1_loss(values, returns)
    
        # crude entropy from logprobs; better is dist.entropy() if you also stored dist params
        entropy = -(logprobs.exp() * logprobs).mean()
    
        return policy_loss + value_coef * value_loss - entropy_coef * entropy
    
    

    def kl_distill_loss(self, student_logits, teacher_logits, alpha: float = 0.8, T: float = 1.0):
        """
        student_logits : Tensor [B, A]  (actor's raw logits)
        teacher_logits : Tensor [B, A]  (gridGPT's raw logits)
        Returns: alpha * T^2 * KL(student || teacher)
        """
        s_logp_T = F.log_softmax(student_logits / T, dim=-1)
        with torch.no_grad():
            t_p_T = F.softmax(teacher_logits / T, dim=-1)
        kl = F.kl_div(s_logp_T, t_p_T, reduction='batchmean')
        return alpha * (T * T) * kl
    

    def calculateLossUpdated(self, gamma=0.99, value_coef=0.5, entropy_coef=0.01):
        # discounted returns
        returns = []
        g = 0.0
        for r in reversed(self.rewards):
            g = r + gamma * g
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
    
        # stabilize return normalization
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    
        values = torch.stack(self.state_values).to(self.device).squeeze(-1)
        logprobs = torch.stack(self.logprobs).to(self.device)
    
        advantages = returns - values.detach()
        # advantage normalization helps a LOT with small/medium LR
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    
        policy_loss = -(logprobs * advantages).mean()
        value_loss  = F.smooth_l1_loss(values, returns)
    
        # crude entropy from logprobs; better is dist.entropy() if you also stored dist params
        entropy = -(logprobs.exp() * logprobs).mean()
    
        return policy_loss + value_coef * value_loss - entropy_coef * entropy
    

    def calculateLoss(self, teacher_logits, gamma=0.99):
        if not (self.logprobs and self.state_values and self.rewards):
            logger.error("Warning: Empty memory buffers!")
            return torch.tensor(0.0, device=self.device)
        

        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
       
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward.unsqueeze(0))
            loss += (action_loss + value_loss)   

        student_seq = torch.stack(self.student_logits, dim=0).to(self.device)
        kl_loss = self.kl_distill_loss(student_seq, teacher_logits)

        return loss + kl_loss

    def _safe(self, t: torch.Tensor, name: str):
        if not torch.isfinite(t).all():
            bad = ~torch.isfinite(t)
            # Optional one-time noisy log
            # print(f"[SAFE] Repaired non-finite values in {name}: {bad.sum().item()} elements")
            t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
        return t

    def _check_idx(self, idx: torch.Tensor, size: int, name: str):
        if idx.min().item() < 0 or idx.max().item() >= size:
            raise RuntimeError(f"{name} out of range [0,{size-1}]: min={idx.min().item()} max={idx.max().item()}")

    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        del self.student_logits[:]


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
        logger.info(f"[LOAD] Checkpoint loaded from {file}")
        return True
    


class gridGPTNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {
            'action_size': 178,
            'block_size': 64,
            'state_dim': 493,
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
    


class GridNetwork(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(GridNetwork, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=512)
        self.layer_2 = torch.nn.Linear(in_features=512, out_features=256)
        self.layer_3 = torch.nn.Linear(in_features=256, out_features=256)
        self.output_layer = torch.nn.Linear(in_features=256, out_features=output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_activation(self.output_layer(layer_3_output))
        return output
    

class GridGPTDiscreteAgent:
    def __init__(self, gpt_config):
        self.config = gpt_config
        self.state_dim = gpt_config['state_dim']
        self.action_dim = gpt_config['action_size']

        self.critic_local = GridNetwork(input_dimension=self.state_dim,
                                    output_dimension=self.action_dim)
        self.critic_local2 = GridNetwork(input_dimension=self.state_dim,
                                     output_dimension=self.action_dim)
        self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.config['learning_rate'])
        self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.config['learning_rate'])

        self.critic_target = GridNetwork(input_dimension=self.state_dim,
                                     output_dimension=self.action_dim)
        self.critic_target2 = GridNetwork(input_dimension=self.state_dim,
                                      output_dimension=self.action_dim)

        self.soft_update_target_networks(tau=1.)

        self.actor_local = gridGPT()
        self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.config['learning_rate'])

        self.replay_buffer = ReplayBuffer(self.environment)

        self.target_entropy = 0.98 * -np.log(1 / self.config['action_size'])
        self.log_alpha = torch.tensor(np.log(self.config['alpha_initial']), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.config['learning_rate'])


        L = self.config['context_len']          # e.g., 16
        S = self.state_dim
        self._ctx_prev  = deque(maxlen=L)       # prev_state  (S,)
        self._ctx_act   = deque(maxlen=L)       # action id   ()
        self._ctx_next  = deque(maxlen=L)       # next_state  (S,)
        self._ctx_step  = deque(maxlen=L)       # timestep    ()
        self._t = 0



    def get_next_action(self, state, evaluation_episode=False):
        if evaluation_episode:
            discrete_action = self.get_action_deterministically(state)
        else:
            discrete_action = self.get_action_nondeterministically(state)
        return discrete_action
    
    def get_action_nondeterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = np.random.choice(range(self.action_dim), p=action_probabilities)
        return discrete_action

    def get_action_deterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = np.argmax(action_probabilities)
        return discrete_action
    

    def train_on_transition(self, state, discrete_action, next_state, reward, done):
        # update GPT rolling context BEFORE training
        self._ctx_prev.append(np.asarray(state, dtype=np.float32))
        self._ctx_act.append(int(discrete_action))
        self._ctx_next.append(np.asarray(next_state, dtype=np.float32))
        self._ctx_step.append(int(self._t))
        self._t += 1

        transition = (state, discrete_action, reward, next_state, done)
        self.train_networks(transition)



    def train_networks(self, transition):
        # Set all the gradients stored in the optimisers to zero.
        self.critic_optimiser.zero_grad()
        self.critic_optimiser2.zero_grad()
        self.actor_optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()
        # Calculate the loss for this transition.
        self.replay_buffer.add_transition(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        if self.replay_buffer.get_size() >= self.config['replay_buffer_batch_size']:
            # get minibatch of 100 transitions from replay buffer
            minibatch = self.replay_buffer.sample_minibatch(self.config['replay_buffer_batch_size'])
            minibatch_separated = list(map(list, zip(*minibatch)))

            # unravel transitions to get states, actions, rewards and next states
            states_tensor = torch.tensor(np.array(minibatch_separated[0]))
            actions_tensor = torch.tensor(np.array(minibatch_separated[1]))
            rewards_tensor = torch.tensor(np.array(minibatch_separated[2])).float()
            next_states_tensor = torch.tensor(np.array(minibatch_separated[3]))
            done_tensor = torch.tensor(np.array(minibatch_separated[4]))

            critic_loss, critic2_loss = \
                self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor)

            critic_loss.backward()
            critic2_loss.backward()
            self.critic_optimiser.step()
            self.critic_optimiser2.step()

            actor_loss, log_action_probabilities = self.actor_loss(states_tensor)

            actor_loss.backward()
            self.actor_optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probabilities)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.soft_update_target_networks()

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor):
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
            next_q_values_target = self.critic_target.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)
            soft_state_values = (action_probabilities * (
                    torch.min(next_q_values_target, next_q_values_target2) - self.alpha * log_action_probabilities
            )).sum(dim=1)

            next_q_values = rewards_tensor + ~done_tensor * self.DISCOUNT_RATE*soft_state_values

        soft_q_values = self.critic_local(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        soft_q_values2 = self.critic_local2(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        critic_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic_square_error, critic2_square_error)]
        self.replay_buffer.update_weights(weight_update)
        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic_loss, critic2_loss

    def actor_loss(self, states_tensor,):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.critic_local(states_tensor)
        q_values_local2 = self.critic_local2(states_tensor)
        inside_term = self.alpha * log_action_probabilities - torch.min(q_values_local, q_values_local2)
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss

    def get_action_info(self, states_tensor):
        """
        Training-time convenience to return probs/log-probs.
        We DON'T have true sequences in your current buffer, so we build a
        lightweight 'dummy' sequence by tiling each state across L and using zeros.
        This lets you keep the rest of SACD code unchanged until you switch
        to a proper sequence replay.
        """
        device = next(self.actor_local.parameters()).device
        B = states_tensor.shape[0]
        L = self.config['context_len']
        S = self.state_dim

        # Build dummy sequences: tile the *current* state across the window.
        states_tensor = states_tensor.to(device).float()                                  # [B,S]
        prev_state = states_tensor.view(B, 1, S).repeat(1, L, 1)                          # [B,L,S]
        next_state = states_tensor.view(B, 1, S).repeat(1, L, 1)                          # [B,L,S]
        action_seq = torch.zeros(B, L, dtype=torch.long, device=device)                   # [B,L] (dummy past actions = 0)
        slot_idx   = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)  # [B,L]
        timestep   = torch.zeros(B, L, dtype=torch.long, device=device)                   # [B,L]
        action_mask_last = None

        logits = self.actor_local.forward(prev_state, action_seq, next_state,
                                        slot_idx=slot_idx, timestep=timestep,
                                        action_mask_last=action_mask_last)              # [B,A]
        probs = F.softmax(logits, dim=-1)
        # Numerically-safe log probs
        log_probs = torch.log(torch.clamp(probs, min=1e-8))
        return probs, log_probs

    def get_action_probabilities(self, state):
        """
        Uses the rolling GPT context to produce probs for the NEXT action.
        'state' is only used to sanity-append into context if you wish (we already updated in train loop).
        """
        self.actor_local.eval()
        with torch.no_grad():
            (ps, a, ns, idx, t, m) = self._pack_gpt_inputs_from_context(device=next(self.actor_local.parameters()).device)
            logits = self.actor_local.forward(ps, a, ns, slot_idx=idx, timestep=t, action_mask_last=m)  # [1, A]
            probs  = F.softmax(logits, dim=-1).squeeze(0)                                               # [A]
        self.actor_local.train()
        return probs.cpu().numpy()

    def soft_update_target_networks(self, tau=0.01):
        self.soft_update(self.critic_target, self.critic_local, tau)
        self.soft_update(self.critic_target2, self.critic_local2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def predict_q_values(self, state):
        q_values = self.critic_local(state)
        q_values2 = self.critic_local2(state)
        return torch.min(q_values, q_values2)
    

    def reset_context(self):
        self._ctx_prev.clear(); self._ctx_act.clear(); self._ctx_next.clear(); self._ctx_step.clear()
        for _ in range(self.config['context_len']):
            self._ctx_prev.append(np.zeros(self.state_dim, dtype=np.float32))
            self._ctx_act.append(0)
            self._ctx_next.append(np.zeros(self.state_dim, dtype=np.float32))
            self._ctx_step.append(0)
        self._t = 0

    
    def _pack_gpt_inputs_from_context(self, device=None):
        """
        Build 1-seq batch [B=1, L, ...] for gridGPT from the rolling deque.
        """
        L = self.config['context_len']
        S = self.state_dim
        A = self.config['action_size']

        prev_state = torch.tensor(np.stack(list(self._ctx_prev), axis=0), dtype=torch.float32, device=device).unsqueeze(0)     # [1,L,S]
        action     = torch.tensor(np.array(list(self._ctx_act), dtype=np.int64), device=device).unsqueeze(0)                   # [1,L]
        next_state = torch.tensor(np.stack(list(self._ctx_next), axis=0), dtype=torch.float32, device=device).unsqueeze(0)     # [1,L,S]

        slot_idx   = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0)                                              # [1,L]
        timestep   = torch.tensor(np.array(list(self._ctx_step), dtype=np.int64), device=device).unsqueeze(0)                  # [1,L]

        # optional mask — allow all actions at last slot by default
        action_mask_last = None
        return prev_state, action, next_state, slot_idx, timestep, action_mask_last




