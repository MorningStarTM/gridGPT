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




class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = self.config['input_dim']

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.policy = nn.Linear(256, self.config['action_dim'])
        self.value = nn.Linear(256, 1)

        self.logprobs = []
        self.state_values = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=self.config['lr'], betas=self.config['betas'])
        self.to(self.device)


    def forward(self, state):
        state = torch.tensor(state, device=self.device)
        state = F.relu(self.network(state))

        state_value = self.value(state)

        action_probs = F.softmax(self.policy(state), dim=-1)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item(), action_probs
    

    def calculateLoss(self, gamma=0.99):
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
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

    def save_checkpoint(self, filename="actor_critic_checkpoint.pth"):
        """Save model + optimizer for exact training resumption."""
        os.makedirs(self.config['save_path'], exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        save_path = os.path.join(self.config['save_path'], filename)
        torch.save(checkpoint, save_path)
        logger.info(f"[SAVE] Checkpoint saved to {save_path}")


    def load_checkpoint(self, folder_name=None, filename="actor_critic_checkpoint.pth", load_optimizer=True):
        """Load model + optimizer state."""
        if folder_name is not None:
            file_path = os.path.join(folder_name, filename)
        else:
            file_path = os.path.join(self.config['save_path'], filename)
        if not os.path.exists(file_path):
            logger.error(f"[LOAD] No checkpoint found at {file_path}")
            return False

        checkpoint = torch.load(file_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"[LOAD] Checkpoint loaded from {file_path}")
        return True
    


        