import grid2op
import torch
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
from src.utils.custom_reward import LossReward, MarginReward
from src.agents.ppo import PPO
from src.agents.gpt import gridGPTAgent
from src.trainer.ppo_trainer import AgentTrainer
from src.trainer.gpt_trainer import OnlineBC
from src.utils.logger import logger


iconfig = {"ENV_NAME" : "l2rpn_case14_sandbox",
            "middle_agent_type" : "capa",  # Options: "capa", "fixed_sub"
            "agent_type" : "ppo",  
            "input_dim" : 493,
            'has_continuous_action_space': False,
            'action_std_init': 0.6,
            'lr_actor': 1e-4,
            'lr_critic': 1e-4,
            'lr':1e-4, 

            'update_timestep': 400*3,
            'gamma': 0.99,
            'K_epochs': 80,
            'eps_clip': 0.2,
            'update_freq':512,

            'max_ep_len': 8063,                       # Max timesteps per episode
            'max_training_timesteps': int(3e6),       # Total training steps before stopping
            'action_std_init':0.6,

            'print_freq': 1000 * 10,                  # Print avg reward every n timesteps
            'log_freq': 1000 * 2,                     # Log reward every n timesteps
            'save_model_freq': int(1e5), 

            'action_std': 0.6,                        # Initial std for action distribution
            'action_std_decay_rate': 0.05,            # Decay rate for action std
            'min_action_std': 0.1,                    # Minimum std after decay
            'action_std_decay_freq': int(2.5e5), 

            'network': "normal",  #resnet, resgcn, gcn
            
            'model_path': "src\\models"
            }


env = grid2op.make(iconfig['ENV_NAME'],
                    reward_class=L2RPNSandBoxScore,
                    backend=LightSimBackend(),
                    other_rewards={"loss": LossReward, "margin": MarginReward})

teacher = PPO(env.observation_space.shape.sum(), env=env, sublist=None, config=iconfig)
teacher.load(checkpoint_path="E:\\github_clone\\gridGPT\\models\\l2rpn_case14_sandbox\\PPO", filename="ppo_checkpoint_2.pth")
logger.info("Teacher loaded from checkpoint.")

agent = gridGPTAgent(env=env, config=iconfig)
logger.info("Student Agent initialized with \n")
logger.info(f"{agent.print_param_count()}")

trainer = OnlineBC(agent=agent, teacher=teacher, env=env, config=iconfig)
trainer.train()
