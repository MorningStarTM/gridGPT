import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.utils.logger import logger
from collections import Counter
from grid2op import Environment
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
from src.utils.custom_reward import LossReward, MarginReward
from grid2op.Exceptions import *
import random
import matplotlib.pyplot as plt
import math
from src.agents.ppo import PPO







class AgentTrainer:
    def __init__(self, agent:PPO, env, config):
        self.env = env
        self.env_name = config['ENV_NAME']
        self.agent = agent
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

        self.log_dir = self.log_dir + '/' + self.env_name + '/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        run_num = 0
        current_num_files = next(os.walk(self.log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        self.log_f_name = self.log_dir + '/PPO_' + self.env_name + "_log_" + str(run_num) + ".csv"

        logger.info("current logging run number for " + self.env_name + " : ", run_num)
        logger.info("logging at : " + self.log_f_name)

        self.directory = "models"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.directory = self.directory + '/' + self.env_name + '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        self.directory = self.directory + '/' + 'PPO' + '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        logger.info(f"directory for saving models : {self.directory}")

        self.reward_folder = 'Rewards'
        if not os.path.exists(self.reward_folder):
            os.makedirs(self.reward_folder)

        self.reward_folder = self.reward_folder + '/' + self.env_name + '/'
        if not os.path.exists(self.reward_folder):
            os.makedirs(self.reward_folder) 
        self.reward_folder = self.reward_folder + '/' + 'PPO' + '/'
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

            for t in range(1, self.config['max_ep_len']+1):

                # select action with policy
                is_safe = self.is_safe(state)
                
                if not is_safe:
                    action_idx, grid_action, logprob, value, state_vec = self.agent.select_action(state.to_vect())
                else:
                    grid_action = self.env.action_space()
                    action_idx, logprob, value, state_vec = -1, None, None, None

                state, reward, done, _ = self.env.step(grid_action)

                time_step +=1
                current_ep_reward += reward

                if not is_safe:
                    if isinstance(state_vec, torch.Tensor):
                        self.agent.buffer.states.append(state_vec.to(self.agent.device))
                    else:
                        self.agent.buffer.states.append(torch.FloatTensor(state_vec).to(self.agent.device))
                    self.agent.buffer.actions.append(torch.tensor(action_idx, device=self.agent.device))
                    self.agent.buffer.logprobs.append(logprob.to(self.agent.device))
                    self.agent.buffer.state_values.append(value.to(self.agent.device))
                    self.agent.buffer.rewards.append(torch.tensor(reward, device=self.agent.device, dtype=torch.float32))
                    self.agent.buffer.is_terminals.append(torch.tensor(done, device=self.agent.device, dtype=torch.float32))


                # update PPO agent
                if time_step % self.config['update_timestep'] == 0:
                    self.agent.update()

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

        np.save(os.path.join(self.reward_folder, f"ppo_{self.env_name}_step_rewards.npy"), np.array(self.step_rewards))
        np.save(os.path.join(self.reward_folder, f"ppo_{self.env_name}_episode_rewards.npy"), np.array(self.episode_rewards))
        np.save(os.path.join(self.reward_folder, f"ppo_{self.env_name}_survival_steps.npy"), np.array(self.survival_steps))
        logger.info(f"Saved step_rewards and episode_rewards to {self.log_dir}")
