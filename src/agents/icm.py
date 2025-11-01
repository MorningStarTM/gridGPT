import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from src.utils.logger import logger


class Memory:
    def __init__(self):
        self.states_ = []
        self.pred_states = []
        self.actions_pred = []
        self.actions = []

    def remember(self, state_, pred_state, actions, pred_actions):
        self.actions.append(actions)
        self.states_.append(state_)
        self.pred_states.append(pred_state)
        self.actions_pred.append(pred_actions)

    def clear_memory(self):
        self.states_ = []
        self.pred_states = []
        self.actions_pred = []
        self.actions = []

    def sample_memory(self):
        return self.states_, self.pred_states, self.actions, self.actions_pred
    

    
    
class ICM(nn.Module):
    def __init__(self, config) -> None:
        super(ICM, self).__init__()
        self.config = config
        self.batch_size = self.config['batch_size']

        
        self.state = nn.Linear(self.config['input_dim'], 512)
        self.state_ = nn.Linear(self.config['input_dim'], 512)

        # inverse Model
        self.inverse_model = nn.Sequential(
                            nn.Linear(1024, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, self.config['action_dim'])
        )


        # forward model
        self.forward_model = nn.Sequential(
                        nn.Linear(513, 512),
                        nn.ReLU(),
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512)
        )


        self.optimizer = optim.Adam(self.parameters(), lr=self.config['icm_lr'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device=self.device)
        self.memory = Memory()

    def _ensure_batch(self, x):
        # state embedding might be [512]; make it [1,512] for concat
        return x.unsqueeze(0) if x.dim() == 1 else x

    
    
    def forward(self, action, state, next_state):
        state = torch.tensor(state.to_vect(), dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state.to_vect(), dtype=torch.float, device=self.device)
        state = self.state(state)
        state_ = self.state_(next_state)

        action_ = self.inverse_model(torch.cat([state, state_], dim=-1))
        action_probs = F.softmax(action_, dim=-1)
        action_distribution = Categorical(action_probs)
        action_pred = action_distribution.sample()
        
        action = torch.tensor(action, dtype=torch.long, device=self.device).unsqueeze(0)
        pred_next_state = self.forward_model(torch.cat([state, action], dim=-1))

        return state_, pred_next_state, action_pred, action_
    

    def calc_batch_loss(self, state_, pred_state, action_idx, action_logits):

        # add batch dim if single sample arrived
        if state_.dim() == 1:       state_ = state_.unsqueeze(0)
        if pred_state.dim() == 1:   pred_state = pred_state.unsqueeze(0)
        if action_logits.dim() == 1: action_logits = action_logits.unsqueeze(0)

        # ---- forward loss in feature space
        Lf = self.config['beta'] * F.mse_loss(pred_state, state_, reduction='mean')

        # ---- inverse loss (logits vs integer indices)  <-- NO unsqueeze, NO re-wrapping
        if isinstance(action_idx, int):
            action_idx = torch.tensor([action_idx], device=self.device, dtype=torch.long)
        else:
            action_idx = action_idx.to(self.device).long().view(-1)

        Li = (1.0 - self.config['beta']) * F.cross_entropy(action_logits, action_idx, reduction='mean')

        # ---- intrinsic reward (no grad), vector per sample

        return Li, Lf


    def calc_loss(self, state_, pred_state, action=None, action_pred=None):

        with torch.no_grad():
            intrinsic_reward = self.config['alpha'] * ((state_ - pred_state).pow(2)).mean(dim=0)
        return intrinsic_reward #Li, Lf
    

    def learn(self):
        states_, pred_states, actions, actions_pred = self.memory.sample_memory()

        states_ = torch.squeeze(torch.stack(states_, dim=0)).to(self.device)
        pred_states = torch.squeeze(torch.stack(pred_states, dim=0)).to(self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).view(-1)
        actions_pred = torch.squeeze(torch.stack(actions_pred, dim=0)).to(self.device)

        # one shot loss & backward (no loops)
        Li, Lf = self.calc_batch_loss(states_, pred_states, actions, actions_pred)
        loss = Li + Lf

        #self.optimizer.zero_grad()
        #loss.backward()
        #self.optimizer.step()

        #logger.info(f"Average Loss: {float(loss.item()):.3f}")
        return loss


    def train(self):
        states_, pred_states, actions, actions_pred = self.memory.sample_memory()

        states_ = torch.squeeze(torch.stack(states_, dim=0)).to(self.device)
        pred_states = torch.squeeze(torch.stack(pred_states, dim=0)).to(self.device)
        #actions = torch.squeeze(torch.stack(actions, dim=0)).float().detach().to(self.device)
        actions = torch.stack([torch.tensor(a, dtype=torch.long) for a in actions], dim=0)
        actions_pred = torch.squeeze(torch.stack(actions_pred, dim=0)).to(self.device)

        logger.info(f"states : {states_.shape}, pred_states : {pred_states.shape}, actions : {actions.shape}, actions_pred : {actions_pred.shape}")

        # Initialize total loss
        total_loss = 0.0

        # Process data in batches
        num_records = states_.shape[0]
        for start_idx in range(0, num_records, self.config['batch_size']):
            # Define batch indices
            end_idx = start_idx + self.config['batch_size']
            state_batch = states_[start_idx:end_idx]
            pred_state_batch = pred_states[start_idx:end_idx]
            action_batch = actions[start_idx:end_idx]
            action_pred_batch = actions_pred[start_idx:end_idx]

            # Compute loss for the batch
            intrinsic_reward, Li, Lf = self.calc_batch_loss(
                state_batch, pred_state_batch, action_batch, action_pred_batch
            )
            batch_loss = Li + Lf
            print(Li, Lf)

            # Backpropagation and optimizer step
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # Accumulate loss
            total_loss += batch_loss.item()

        # Print average loss for the epoch
        avg_loss = total_loss / (num_records / self.config['batch_size'])
        logger.info(f"Average Loss: {avg_loss:.3f}")


    def save_checkpoint(self, filename="icm_checkpoint.pth"):
        os.makedirs(self.config['save_path'], exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        save_path = os.path.join(self.config['save_path'], filename)
        torch.save(checkpoint, save_path)
        logger.info(f"ICM model saved to {save_path}")

    def load_checkpoint(self, folder_name=None, filename="icm_checkpoint.pth"):
        if folder_name is not None:
            file_path = os.path.join(folder_name, filename)
        else:
            file_path = os.path.join(self.config['save_path'], filename)
        if not os.path.exists(file_path):
            logger.error(f"[LOAD] No checkpoint found at {file_path}")
            return False
        
        checkpoint = torch.load(file_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        logger.info(f"ICM model loaded from {filename}")






