import torch
import os
import numpy as np
from src.agents.gpt import gridGPTAgent
from torch.utils.data import Dataset, DataLoader
from src.utils.logger import logger
from collections import Counter
from datetime import datetime
from grid2op import Environment
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
from src.utils.custom_reward import LossReward, MarginReward
from grid2op.Exceptions import *
from src.agents.ppo import PPO



class OnlineBC:
    def __init__(self, agent:gridGPTAgent, teacher:PPO,  env, config):
        self.env = env
        self.agent = agent
        self.teacher = teacher
        self.best_score = 0.0
        self.score_history = []
        self.config = config
        self.episode_rewards = []  # Stores total reward per episode
        self.step_rewards = []     # Stores every single reward at each timestep
        self.thermal_limit = self.env._thermal_limit_a
        self.danger = 0.9
        self.survival_steps = []


    
        logger.info(""""============================================================================================
                                                Agent Training Started               
                    ===========================================================================================""")

        self.log_dir = "model_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.log_dir = self.log_dir + '/' + self.config['ENV_NAME'] + '/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        run_num = 0
        current_num_files = next(os.walk(self.log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        self.log_f_name = self.log_dir + '/gridGPT_student_' + self.config['ENV_NAME'] + "_log_" + str(run_num) + ".csv"

        logger.info("current logging run number for " + self.config['ENV_NAME'] + " : ", run_num)
        logger.info("logging at : " + self.log_f_name)

        self.directory = "models"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.directory = self.directory + '/' + self.config['ENV_NAME'] + '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        self.directory = self.directory + '/' + 'gridGPT_student' + '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        logger.info(f"directory for saving models : {self.directory}")

        self.reward_folder = 'Rewards'
        if not os.path.exists(self.reward_folder):
            os.makedirs(self.reward_folder)

        self.reward_folder = self.reward_folder + '/' + self.config['ENV_NAME'] + '/'
        if not os.path.exists(self.reward_folder):
            os.makedirs(self.reward_folder) 
        self.reward_folder = self.reward_folder + '/' + 'gridGPT_student_' + '/'
        if not os.path.exists(self.reward_folder):
            os.makedirs(self.reward_folder) 
        logger.info(f"directory for saving rewards : {self.reward_folder}")


    def is_safe(self, obs):
        
        for ratio, limit in zip(obs.rho, self.thermal_limit):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= self.danger - 0.05) or ratio >= self.danger:
                return False
        return True
    

    def train(self):
        logger.info("""================================================================================
                    =========================    Online Behaviour cloning   =========================
                    ================================================================================""")
        start_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        # logging file
        log_f = open(self.log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0
        
        while time_step <= self.config['max_training_timesteps']:

            state = self.env.reset()
            current_ep_reward = 0

            # ====== sliding window buffers (B=1) ======
            L          = self.agent.policy.config['context_len']   # 16
            state_dim  = self.agent.policy.config['state_dim']     # 10
            action_sz  = self.agent.policy.config['action_size']
            tmax       = self.agent.policy.config['max_timestep']

            device = self.agent.device

            window_prev_states = torch.zeros(L, state_dim, device=device)
            window_next_states = torch.zeros(L, state_dim, device=device)
            window_actions     = torch.zeros(L, dtype=torch.long, device=device)

            slot_idx_full    = torch.arange(L, device=device).unsqueeze(0)           # [1, L]
            action_mask_last = torch.ones(1, action_sz, dtype=torch.bool, device=device)



            for t in range(1, self.config['max_ep_len']+1):
                s_t = torch.from_numpy(state.to_vect()).to(device).float()
                # roll the window left by 1
                window_prev_states = torch.roll(window_prev_states, shifts=-1, dims=0)
                window_next_states = torch.roll(window_next_states, shifts=-1, dims=0)
                window_actions     = torch.roll(window_actions,     shifts=-1, dims=0)

                # fill last slot with [prev_state, prev_action, next_state]
                if t == 1:
                    prev_slot   = torch.zeros(state_dim, device=device)
                    last_action = 0  # or your do-nothing id
                else:
                    prev_slot   = s_prev.clone()
                    last_action = int(a_prev)

                window_prev_states[-1] = prev_slot
                window_next_states[-1] = s_t
                window_actions[-1]     = last_action

                # build batched tensors for model
                prev_states_b = window_prev_states.unsqueeze(0)  # [1, L, state_dim]
                next_states_b = window_next_states.unsqueeze(0)  # [1, L, state_dim]
                actions_b     = window_actions.unsqueeze(0)      # [1, L]

                # absolute timesteps per slot (left-padded with zeros at the beginning)
                base = (t-1) - (L - 1)
                times = torch.clamp(torch.arange(base, base + L, device=device), 0, tmax - 1)
                timestep_b = times.unsqueeze(0)  # [1, L]

                # ---- safety check ----
                is_safe = self.is_safe(state)

                if not is_safe:
                    # student (gridGPT) picks discrete action id
                    action_id, grid_action, logprob, value = self.agent.select_action(
                        prev_states_b, actions_b, next_states_b,
                        slot_idx_full, timestep_b, action_mask_last,
                        deterministic=False  # or True for greedy
                    )

                    # teacher (PPO) action id (behavior target) — optional if you need KL/BC targets
                    teacher_action_idx, _, teacher_logprob, _, _ = self.teacher.select_action(state.to_vect())

                else:
                    # do-nothing (when safe) — keep the grid stable
                    grid_action = self.env.action_space()  # 'do-nothing' action
                    action_id, logprob, value = -1, None, None


                next_state, reward, done, info = self.env.step(grid_action)
                current_ep_reward += reward
                time_step += 1

                s_prev = s_t.clone()
                s_t = torch.from_numpy(next_state.to_vect()).to(device).float()
                a_prev = 0 if action_id == -1 else action_id
                

                if not is_safe:
                    ps = prev_states_b[0].detach().to("cpu")     # [L, state_dim]
                    asq = actions_b[0].detach().to("cpu")        # [L]
                    ns = next_states_b[0].detach().to("cpu")     # [L, state_dim]
                    si = slot_idx_full[0].detach().to("cpu")     # [L]
                    ts = timestep_b[0].detach().to("cpu")        # [L]
                    am = action_mask_last[0].detach().to("cpu")  # [A] (or [L, A] if per-slot)

                    self.agent.buffer.seq_prev_states.append(ps)
                    self.agent.buffer.seq_actions.append(asq)
                    self.agent.buffer.seq_next_states.append(ns)
                    self.agent.buffer.seq_slot_idx.append(si)
                    self.agent.buffer.seq_timesteps.append(ts)
                    #self.agent.buffer.seq_action_masks.append(am)

                    
                    if isinstance(state.to_vect(), torch.Tensor):
                        self.agent.buffer.states.append(state.to_vect().to(self.agent.device))
                    else:
                        self.agent.buffer.states.append(torch.FloatTensor(state.to_vect()).to(self.agent.device))
                    self.agent.buffer.actions.append(torch.tensor(action_id, device=self.agent.device))
                    #self.agent.buffer.teacher_probs.append(torch.tensor(teacher_logprob, device=self.agent.device))
                    if isinstance(teacher_logprob, torch.Tensor):
                        self.agent.buffer.teacher_probs.append(teacher_logprob.clone().detach().to(self.agent.device))
                    else:
                        self.agent.buffer.teacher_probs.append(
                            torch.as_tensor(teacher_logprob, dtype=torch.float32, device=self.agent.device)
                        )
                    self.agent.buffer.logprobs.append(logprob.to(self.agent.device))
                    self.agent.buffer.state_values.append(value.to(self.agent.device))
                    self.agent.buffer.rewards.append(torch.tensor(reward, device=self.agent.device, dtype=torch.float32))
                    self.agent.buffer.is_terminals.append(torch.tensor(done, device=self.agent.device, dtype=torch.float32))

                state  = next_state

                # update PPO agent
                if time_step % self.config['update_timestep'] == 0:
                    self.agent.update(teacher=self.teacher)

                # if continuous action space; then decay action std of ouput action distribution
                if self.config['has_continuous_action_space'] and time_step % self.config['action_std_decay_freq'] == 0:
                    self.agent.decay_action_std(self.config['action_std_decay_rate'], self.config['min_action_std'])

                # log in logging file
                if time_step % self.config['log_freq'] == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.config['print_freq'] == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    logger.info("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % self.config['save_model_freq'] == 0:
                    logger.info("--------------------------------------------------------------------------------------------")
                    logger.info("saving model at : " + self.directory)
                    self.agent.save(self.directory)
                    logger.info("model saved")
                    logger.info("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    logger.info("--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    self.survival_steps.append(t)
                    break
            
            self.episode_rewards.append(current_ep_reward)  
            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        log_f.close()
        self.env.close()

        # print total training time
        logger.info("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)
        logger.info("Finished training at (GMT) : ", end_time)
        logger.info("Total training time  : ", end_time - start_time)
        logger.info("============================================================================================")

        np.save(os.path.join(self.reward_folder, f"ppo_{self.config['ENV_NAME']}_step_rewards.npy"), np.array(self.step_rewards))
        np.save(os.path.join(self.reward_folder, f"ppo_{self.config['ENV_NAME']}_episode_rewards.npy"), np.array(self.episode_rewards))
        np.save(os.path.join(self.reward_folder, f"ppo_{self.env_name}_survival_steps.npy"), np.array(self.survival_steps))
        logger.info(f"Saved step_rewards and episode_rewards to {self.log_dir}")
