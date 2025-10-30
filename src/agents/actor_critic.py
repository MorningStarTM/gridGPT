import torch 
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import grid2op
from src.utils.converter import ActionConverter
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
import os
from src.utils.logger import logger
import torch.optim as optim




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine = nn.Sequential(
            nn.Linear(493, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),                 # ← normalize hidden to tame scales
            nn.ReLU(),
        )
        self.action_layer = nn.Linear(256, 178)
        self.value_layer  = nn.Linear(256, 1)

        self.logprobs, self.state_values, self.rewards = [], [], []

        # Optional: safer initializations
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _sanitize(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x.clamp_(-1e6, 1e6)
        return x

    def forward(self, state_np):
        x = torch.from_numpy(state_np).float().to(self.value_layer.weight.device)
        x = self._sanitize(x)

        h = self.affine(x)                       # includes LayerNorm + ReLU
        h = torch.nan_to_num(h)                  # belt & suspenders

        logits = self.action_layer(h)
        logits = torch.nan_to_num(logits)        # if any NaN slipped through
        logits = logits - logits.max()           # stable softmax
        probs  = torch.softmax(logits, dim=-1)

        # final guard
        if not torch.isfinite(probs).all():
            # Zero-out non-finites and renormalize as an emergency fallback
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            s = probs.sum()
            probs = (probs + 1e-12) / (s + 1e-12)

        dist   = Categorical(probs=probs)
        action = dist.sample()

        self.logprobs.append(dist.log_prob(action))
        self.state_values.append(self.value_layer(h).squeeze(-1))

        return action.item(), logits

    
    def calculateLoss(self, gamma=0.99, value_coef=0.5, entropy_coef=0.01):
        # discounted returns
        returns = []
        g = 0.0
        for r in reversed(self.rewards):
            g = r + gamma * g
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
    
        # stabilize return normalization
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    
        values = torch.stack(self.state_values).to(device).squeeze(-1)
        logprobs = torch.stack(self.logprobs).to(device)
    
        advantages = returns - values.detach()
        # advantage normalization helps a LOT with small/medium LR
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    
        policy_loss = -(logprobs * advantages).mean()
        value_loss  = F.smooth_l1_loss(values, returns)
    
        # crude entropy from logprobs; better is dist.entropy() if you also stored dist params
        entropy = -(logprobs.exp() * logprobs).mean()
    
        return policy_loss + value_coef * value_loss - entropy_coef * entropy

    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

    def save_checkpoint(self, optimizer:optim=None, filename="actor_critic_checkpoint.pth"):
        """Save model + optimizer for exact training resumption."""
        os.makedirs("models", exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        save_path = os.path.join("models", filename)
        torch.save(checkpoint, save_path)
        print(f"[SAVE] Checkpoint saved to {save_path}")


    def load_checkpoint(self, folder_name=None, filename="actor_critic_checkpoint.pth", optimizer:optim=None, load_optimizer=True):
        """Load model + optimizer state."""
        if folder_name is not None:
            file_path = os.path.join(folder_name, filename)
        else:
            file_path = os.path.join("models", filename)
        if not os.path.exists(file_path):
            print(f"[LOAD] No checkpoint found at {file_path}")
            return False

        checkpoint = torch.load(file_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[LOAD] Checkpoint loaded from {file_path}")
        return True