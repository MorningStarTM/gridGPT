import grid2op
from grid2op.Reward import L2RPNSandBoxScore
from lightsim2grid import LightSimBackend
from grid2op import Environment
from src.utils.converter import ActionConverter
from src.agents.icm import ICM
import torch.optim as optim
from grid2op.Exceptions import *
from tqdm import tqdm
from src.agents.gpt import gridGPTAC
from src.utils.utils import save_episode_rewards
from src.utils.logger import logger
import random
import inspect
import numpy as np
import pandas as pd
import torch
import os




import os, numpy as np, torch
import torch.optim as optim
from datetime import datetime
from loguru import logger

class ICMTrainer:
    def __init__(self, env: Environment, gpt_config, config) -> None:
        self.env        = env
        self.config     = config
        self.gpt_config = gpt_config
        self.danger = 0.9
        self.thermal_limit = self.env._thermal_limit_a


        # student (sequence GPT-AC)
        self.agent = gridGPTAC(gpt_config)
        self.agent.load(checkpoint_path=self.gpt_config['model_folder'],
                        filename=self.gpt_config['filename'])
        logger.info(f"Loaded Agent from {self.gpt_config['model_folder']}\\{self.gpt_config['filename']}")

        # helpers
        self.converter = ActionConverter(env=self.env)
        self.icm       = ICM(config)
        self.actor_optimizer = optim.Adam(self.agent.parameters(),
                                          lr=self.config['lr'],
                                          betas=self.config['betas'])
        self.icm_optimizer   = self.icm.optimizer

        # tracking
        self.best_survival_step = 0
        self.episode_rewards    = []
        self.suruvival_steps    = []

        # sequence hyperparams from the student policy config
        pcfg = self.agent.config if hasattr(self.agent, "config") else self.gpt_config
        self.L         = int(pcfg['context_len'])
        self.state_dim = int(pcfg['state_dim'])
        self.action_sz = int(pcfg['action_size'])
        self.tmax      = int(pcfg.get('max_timestep', 4096))

        # device
        self.device = getattr(self.agent, "device",
                              torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # logging dirs
        self.env_name   = str(self.config.get('ENV_NAME', 'l2rpn_case14_sandbox'))
        self.reward_dir = os.path.join("Rewards", self.env_name, "icm_seq")
        os.makedirs(self.reward_dir, exist_ok=True)

    
    def is_safe(self, obs):
        
        for ratio, limit in zip(obs.rho, self.thermal_limit):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= self.danger - 0.05) or ratio >= self.danger:
                return False
        return True

    def fit(self):
        logger.info("""=======================================================
                            ICM + Sequence StudentGPT Training
                       =======================================================""")
        running_reward = 0.0
        time_step = 0

        for i_episode in range(int(self.config['episodes'])):
            state = self.env.reset()
            done = False
            ep_reward = 0.0

            # --- init sliding windows (B=1) ---
            window_prev_states = torch.zeros(self.L, self.state_dim, device=self.device)   # [L,S]
            window_next_states = torch.zeros(self.L, self.state_dim, device=self.device)   # [L,S]
            window_actions     = torch.zeros(self.L, dtype=torch.long, device=self.device) # [L]
            slot_idx_full      = torch.arange(self.L, device=self.device).unsqueeze(0)     # [1,L]
            action_mask_last   = torch.ones(1, self.action_sz, dtype=torch.bool, device=self.device)

            s_prev = None
            a_prev = 0  # no-op id (use what your model expects)

            for t in range(1, int(self.config['max_ep_len']) + 1):
                # vectorize current state
                s_np = state.to_vect()
                s_t  = torch.from_numpy(s_np).to(self.device).float()

                # roll windows
                window_prev_states = torch.roll(window_prev_states, shifts=-1, dims=0)
                window_next_states = torch.roll(window_next_states, shifts=-1, dims=0)
                window_actions     = torch.roll(window_actions,     shifts=-1, dims=0)

                # tail slot values
                if t == 1 or s_prev is None:
                    prev_slot, last_action = torch.zeros(self.state_dim, device=self.device), 0
                else:
                    prev_slot, last_action = s_prev, int(a_prev)

                window_prev_states[-1] = prev_slot
                window_next_states[-1] = s_t
                window_actions[-1]     = last_action

                # batchify to [1,L,...]
                prev_b = window_prev_states.unsqueeze(0) # [1,L,S]
                next_b = window_next_states.unsqueeze(0) # [1,L,S]
                act_b  = window_actions.unsqueeze(0)     # [1,L]

                # time indices [1,L]
                base  = (t - 1) - (self.L - 1)
                times = torch.clamp(torch.arange(base, base + self.L, device=self.device),
                                    0, self.tmax - 1)
                tstep_b = times.unsqueeze(0)             # [1,L]

                is_safe = self.is_safe(state)
                if not is_safe:
                # ---- student: sample action (also fills memory for AC loss) ----
                    action_id = self.agent(prev_b, act_b, next_b,
                                        slot_idx=slot_idx_full,
                                        timestep=tstep_b,
                                        action_mask_last=action_mask_last,
                                        return_value=True)    # -> int

                    env_action = self.converter.act(int(action_id))
                
                else:
                    env_action = self.env.action_space({})

                # ---- step env ----
                next_state, extr_reward, done, info = self.env.step(env_action)

                # ---- ICM forward pass & intrinsic reward ----
                # If your ICM needs vectors instead of env obs, pass s_prev/s_t tensors or raw np vectors.
                state_, pred_next_state, action_pred, action_hat = self.icm(action_id, state, next_state)

                icm_forward_loss = self.icm.calc_loss(state_=state_, pred_state=pred_next_state)
                # treat forward loss as intrinsic reward (common choice), weight it
                eta = float(self.config.get('intrinsic_reward_weight', 0.1))
                intrinsic_reward = float(icm_forward_loss.detach().item())  # scalar

                total_reward = float(extr_reward) + eta * intrinsic_reward

                if not is_safe:
                    # ---- push reward to student memory ----
                    self.agent.rewards.append(total_reward)

                # ---- store experience for ICM training buffer (your API) ----
                self.icm.memory.remember(state_=state_,
                                         pred_state=pred_next_state,
                                         actions=int(action_id),
                                         pred_actions=action_hat)

                # bookkeeping
                ep_reward += total_reward
                time_step += 1
                s_prev = s_t
                a_prev = int(action_id)
                state  = next_state

                if done:
                    self.best_survival_step = max(self.best_survival_step, t)
                    self.suruvival_steps.append(t)
                    break

            # ---- episode end: optimize student + ICM ----
            self.actor_optimizer.zero_grad()
            self.icm_optimizer.zero_grad()

            # ICM loss (your implementation mixes forward/inverse and returns scalar tensor)
            icm_loss = self.icm.learn()

            # Student AC+KL (ICM version): uses internal memory the forward filled
            gamma = float(self.config.get('gamma', 0.99))
            policy_loss = self.agent.calculateLossICM(gamma)

            total_loss = policy_loss + icm_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.icm.parameters(),   1.0)

            self.actor_optimizer.step()
            self.icm_optimizer.step()

            # clear episode memory
            self.agent.clearMemory()
            self.icm.memory.clear_memory()

            self.episode_rewards.append(ep_reward)

            # periodic save
            if i_episode != 0 and (i_episode % 1000 == 0):
                self.agent.save(checkpoint_path=self.gpt_config['save_folder'],
                                filename=f"icm_gridGPT.pt")
                self.icm.save_checkpoint(filename="final_icm.pt")

            # log
            running_reward += ep_reward
            if i_episode % 20 == 0:
                avg20 = running_reward / max(1, (1 if i_episode == 0 else 20))
                logger.info(f"Episode {i_episode}\tlen: {t}\tep_reward: {ep_reward:.3f}\tavg20: {avg20:.3f}")
                running_reward = 0.0

        # wrap-up: save rewards
        os.makedirs(os.path.join("ICM", "episode_reward"), exist_ok=True)
        np.save(os.path.join("ICM", "episode_reward", "final_actor_critic_reward.npy"),
                np.array(self.episode_rewards, dtype=np.float32))
        np.save(os.path.join(self.reward_dir, "survival_steps.npy"), np.array(self.survival_steps))
        logger.info("Reward saved at ICM\\episode_reward")





    def train(self, start=0, end=10):
        num_episodes = len(self.env.chronics_handler.subpaths)
        train_step = 0
        for episode_id in range(start, end):

            print(f"Episode ID : {episode_id}")
            self.env.set_id(episode_id)
            obs = self.env.reset()
            done = False

            for i in tqdm(range(self.env.max_episode_duration()), desc=f"Episode {episode_id}", leave=True):
                train_step += 1
                try:
                    action = self.agent(obs.to_vect()) 
                    obs_, env_reward, done, _ = self.env.step(self.converter.act(action))
                    state_, pred_next_state, action_pred, action_ = self.icm(action, obs, obs_)

                    intrinsic_reward = self.icm.calc_loss(state_=state_, pred_state=pred_next_state)

                    self.icm.memory.remember(state_=state_, pred_state=pred_next_state, actions=action, pred_actions=action_)

                    total_reward = env_reward + intrinsic_reward * 0.001
                    self.agent.rewards.append(total_reward)

                    obs = obs_

                    if done:
                        self.env.set_id(episode_id)
                        
                        obs = self.env.reset()
                        done = False
                        reward = self.env.reward_range[0]

                        self.env.fast_forward_chronics(i - 1)
                        action = self.agent(obs.to_vect()) 
                        obs_, reward, done, _ = self.env.step(self.converter.act(action))
                        self.agent.rewards.append(reward)

                    
                    if train_step == 1024:
                        logger.info(f"\n\n###########################################\n Updating at {i}.....\n\n#####################################################")
                        self.actor_optimizer.zero_grad()
                        self.icm_optimizer.zero_grad()

                        icm_loss = self.icm.learn()
                        policy_loss = self.agent.calculateLoss(self.config['gamma'])
                        total_loss = icm_loss + policy_loss

                        total_loss.backward()
                        self.actor_optimizer.step()
                        self.icm_optimizer.step()

                        self.agent.clearMemory()
                        self.icm.memory.clear_memory()
                        train_step = 0



                except NoForecastAvailable as e:
                    logger.info(f"Grid2OpException encountered at step {i} in episode {episode_id}: {e}")
                    self.env.set_id(episode_id)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(i-1)
                    continue

                except Grid2OpException as e:
                    logger.info(f"Grid2OpException encountered at step {i} in episode {episode_id}: {e}")
                    self.env.set_id(episode_id)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(i-1)
                    continue 


            if episode_id!=0 and episode_id % 5 == 0:
                print(f"\n\n#############################################\n Saving the Agent \n\n#############################################\n\n")
                self.agent.save(checkpoint_path=self.config['folder'],filename=f"icm_actor_critic_{episode_id}.pt")
                self.icm.save_checkpoint(filename=f"icm_{episode_id}.pt")
                
