import grid2op
import torch
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
from src.utils.custom_reward import LossReward, MarginReward
from src.agents.ppo import PPO
from src.agents.actor_critic import ActorCritic
from src.agents.gpt import gridGPTAgent, gridGPTAC
from src.trainer.ppo_trainer import AgentTrainer
from src.utils.converter import ActionConverter
from src.trainer.gpt_trainer import OnlineBC, OnlineBC_AC_SeqTrainer
from src.utils.logger import logger
from src.trainer.icm_trainer import ICMTrainer

iconfig = {"ENV_NAME" : "l2rpn_case14_sandbox",
            "middle_agent_type" : "capa",  # Options: "capa", "fixed_sub"
            "agent_type" : "ppo",  
            "input_dim" : 493,
            'has_continuous_action_space': False,
            'action_std_init': 0.6,
            'lr_actor': 1e-4,
            'lr_critic': 1e-4,
            'lr':1e-4, 

            'update_timestep': 400*6,
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


converter = ActionConverter(env=env)


actor_config = {
    "ENV_NAME": "l2rpn_case14_sandbox",
    "input_dim":493, #env.observation_space.shape.sum(),
    "action_dim":converter.n,
    "gamma": 0.99,
    "lr": 0.0003,
    "betas": (0.9, 0.999),
    "update_freq": 512,
    "save_path":"ICM\models",
    'episodes': 10000,
    'max_ep_len':10000,
    'icm_lr':1e-4,
    'beta':1e-4,
    'alpha':1e-4,
    'batch_size':256,
    'model_path' : 'models',
    'file_name':'final_actor_critic_checkpoint.pth'
}

gpt_config = {
            'action_size': 178,
            'block_size': 64,
            'state_dim': 493,
            'n_embd': 128,
            'lr': 1e-4,
            'fusion_embed_dim': 3 * 128,  # 3 * n_embd
            'n_head': 4,
            'head_size': 32,
            'n_layers': 4,
            'dropout': 0.1,
            'betas': (0.9, 0.999),
            'context_len': 16,       # window length L (slot indices 0..L-1)
            'max_timestep': 10000,    # max absolute env step used for embeddings
            'folder' : 'models\\l2rpn_case14_sandbox\\gridGPT_student',
            'filename':'gpt_checkpoint.pth'
        }

icm_config = {
    "input_dim":493, #env.observation_space.shape.sum(),
    "action_dim":converter.n,
    "gamma": 0.99,
    "lr": 0.0003,
    "betas": (0.9, 0.999),
    "update_freq": 512,
    "save_path":"models\\l2rpn_case14_sandbox\\gridGPT_student",
    'episodes': 10000,
    'max_ep_len':10000,
    'icm_lr':1e-4,
    'beta':1e-4,
    'alpha':1e-4,
    'batch_size':256,
    'intrinsic_reward_weight':1,
}
# teacher = PPO(env.observation_space.shape.sum(), env=env, sublist=None, config=iconfig)
# trainer = AgentTrainer(agent=teacher, env=env, config=iconfig)
# trainer.train()
# teacher.load(checkpoint_path="E:\\github_clone\\gridGPT\\models\\l2rpn_case14_sandbox\\PPO", filename="ppo_checkpoint_2.pth")
# logger.info("Teacher loaded from checkpoint.")

# agent = gridGPTAgent(env=env, config=iconfig)
# logger.info("Student Agent initialized with \n")
# logger.info(f"{agent.print_param_count()}")

# trainer = OnlineBC(agent=agent, teacher=teacher, env=env, config=iconfig)
# trainer.train()


# trainer = OnlineBC_AC_SeqTrainer(env=env, converter=converter, ac_config=actor_config, gpt_config=gpt_config)
# trainer.train()

trainer = ICMTrainer(env=env, gpt_config=gpt_config, config=icm_config)
trainer.fit()