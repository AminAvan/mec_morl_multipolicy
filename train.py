#!/usr/bin/env python
# coding: utf-8

import torch, numpy as np, torch.nn as nn

import gymnasium as gym ## added
import platform
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from copy import deepcopy
from tianshou.env import DummyVectorEnv
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import os
import time
import json
import math
from tqdm import tqdm
from env import MEC_Env
from network import conv_mlp_net
from datetime import datetime
import psutil

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.logger.tensorboard import TensorboardLogger
from torch.distributions import Categorical
from tianshou.policy import PPOPolicy
from tianshou.env import DummyVectorEnv
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import GPUtil
import subprocess
import psutil
"""
added by amin but it fails to each the intended goal
from zeus.monitor import ZeusMonitor
from zeus.device.cpu import get_current_cpu_index
from zeus.device.gpu.common import ZeusGPUInvalidArgError
import pynvml
"""
import argparse


###############################################################
### added by amin to log the memory consumption and power #####
from datetime import datetime
# Generate the filename with the current date and time
current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS
filename = f"Power_RAM_MORL_{current_date_time}.csv"
file = open(filename, 'w')  # Open the file
file.write("Results:\n")  # Header

## memory calculator
process = psutil.Process(os.getpid())
# accum_mem = 0.0  # will hold byte-seconds
## power usage
# total_power_usage = 0.0
# last_logged_step = -1
#####################################################


edge_num = 2
expn = 'exp1'
config = 'multi-edge'
# lr, epoch, batch_size = 1e-6, 1, 1024*4 ## was -- too slow
"""
'lr' is learning rate where '1e-6' is considered slow learning rate.
'1e-6' is extremely conservative learning rate for PPO, typically PPO uses lr in the range of 1e-4 to 3e-3 for faster learning.
Epoch number is '1', although fewer epochs reduce training time, but too few may prevent convergence.
The issue with 'single' epoch may be insufficient for the agent to learn effectively, especially in a complex multi-edge environment.
A single epoch limits learning, especially in edge computing; thus, more epochs allow the rl-agent to refine the policy over multiple passes.
A smaller batch size reduces memory and computation per update, speeding up training on GPU. 1024 is still large enough for stable PPO updates.
"""
lr, epoch, batch_size = 1e-4, 1, 1024
# train_num, test_num = 64, 1024 ## was
"""
1024 test episodes per epoch is high, increasing evaluation time without proportional benefits.
256 episodes provide reliable statistics with less computation.
"""
train_num, test_num = 64, 256
gamma, lr_decay = 0.9, None
buffer_size = 100000
eps_train, eps_test = 0.1, 0.00
# step_per_epoch, episode_per_collect = 100*train_num*700, train_num ## was
"""
4.48 million steps per epoch is excessive, contributing to the ~45-minute training time.
Reducing to 640,000 steps (~1/7th) speeds up data collection while still providing ample exploration.
"""
# step_per_epoch, episode_per_collect = 100*train_num*100, train_num ## amin was
step_per_epoch, episode_per_collect = 100*train_num*125, train_num ## step per epoch is 800000
# writer = SummaryWriter('tensor-board-log/ppo')  # tensorboard is also supported! ## was
# logger = ts.utils.BasicLogger(writer) ## was
writer = SummaryWriter('tensor-board-log/ppo')
logger = TensorboardLogger(writer)
is_gpu = True
#ppo
gae_lambda, max_grad_norm = 0.95, 0.5
vf_coef, ent_coef = 0.5, 0.0
# rew_norm, action_scaling = False, False ## was
"""
Normalizing rewards stabilizes training with a higher lr and varying reward scales.
"""
rew_norm, action_scaling = True, False
bound_action_method = "clip"
#eps_clip, value_clip = 0.2, False
"""
A larger clipping range allows bigger policy changes, potentially speeding convergence, especially with a higher lr.
"""
eps_clip, value_clip = 0.3, False
# repeat_per_collect = 2 ## was
"""
More updates per collected batch improve sample efficiency, allowing the agent to learn more from less data.
This compensates for reduced step_per_epoch.
"""
repeat_per_collect = 4
dual_clip, norm_adv = None, 0.0
recompute_adv = 0


INPUT_CH = 67
FEATURE_CH = 512
MLP_CH = 1024
class mec_net(nn.Module):
    def __init__(self, mode='actor', is_gpu=True):
        super().__init__()
        self.is_gpu = is_gpu
        self.mode = mode
        
        if self.mode == 'actor':
            self.network = conv_mlp_net(conv_in=INPUT_CH, conv_ch=FEATURE_CH, mlp_in=(edge_num+1)*FEATURE_CH,\
                                    mlp_ch=MLP_CH, out_ch=edge_num+1, block_num=3)
        else:
            self.network = conv_mlp_net(conv_in=INPUT_CH, conv_ch=FEATURE_CH, mlp_in=(edge_num+1)*FEATURE_CH,\
                                    mlp_ch=MLP_CH, out_ch=1, block_num=3)
        
    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')

    def save_model(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create parent directory
        torch.save(self.state_dict(), filename)
        print('save model!')

    def forward(self, obs, state=None, info={}):
        state = obs#['servers']
        state = torch.tensor(state).float()
        if self.is_gpu:
            state = state.cuda()

        logits = self.network(state)
        
        return logits, state



class Actor(nn.Module):
    def __init__(self, is_gpu=True):
        super().__init__()
        
        self.is_gpu = is_gpu

        self.net = mec_net(mode='actor')

    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')
    
    def save_model(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create parent directory
        torch.save(self.state_dict(), filename)
        # print('save model!')

    def forward(self, obs, state=None, info={}):
            
        logits,_ = self.net(obs)
        logits = F.softmax(logits, dim=-1)

        return logits, state


class Critic(nn.Module):
    def __init__(self, is_gpu=True):
        super().__init__()

        self.is_gpu = is_gpu

        self.net = mec_net(mode='critic')

    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')

    def save_model(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create parent directory
        torch.save(self.state_dict(), filename)
        # print('save model!')

    def forward(self, obs, state=None, info={}):
        v,_ = self.net(obs)

        return v


actor = Actor(is_gpu = is_gpu)
critic = Critic(is_gpu = is_gpu)

load_path = None

if is_gpu:
    actor.cuda()
    critic.cuda()

    
from tianshou.utils.net.common import ActorCritic
actor_critic = ActorCritic(actor, critic)

optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)




dist = torch.distributions.Categorical

action_space = gym.spaces.Discrete(edge_num)

if lr_decay:
    lr_scheduler = LambdaLR(
        optim, lr_lambda=lambda epoch: lr_decay**(epoch-1)
    )
else:
    lr_scheduler = None

policy = PPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=Categorical,            # <- distribution for discrete actions
    discount_factor=gamma,
    max_grad_norm=max_grad_norm,
    eps_clip=eps_clip,
    vf_coef=vf_coef,
    ent_coef=ent_coef,
    reward_normalization=rew_norm,
    advantage_normalization=norm_adv,
    recompute_advantage=recompute_adv,
    dual_clip=dual_clip,
    value_clip=value_clip,
    gae_lambda=gae_lambda,
    action_space=action_space,      # gym.spaces.Discrete(edge_num)
    lr_scheduler=lr_scheduler,
    action_scaling=False,           # <‑‑ key line for Discrete spaces
)

# naming the files that are going to be saved with unique names ## added
# Get unique timestamp ## added
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S') ## added
# Compose base directory with timestamp ## added
base_dir = f'save/pth-e{edge_num}/{expn}-{timestamp}' ## added

# for i in range(101): ## was
for i in range(100,0-1,-25):
    try:
        os.makedirs(os.path.join(base_dir, f'w{i:03d}'), exist_ok=True) ## added
        # os.mkdir('save/pth-e%d/'%(edge_num) + expn + '/w%03d'%(i)) ## was
    except:
        pass


# for wi in range(100,0-1,-25):
for wi in range(100, -1, -25):
    # epoch_a = 5 if wi == 100 else 1
    # ## was
    # if wi==100:
    #     # epoch_a = epoch * 10
    # else:
    #     # epoch_a = epoch ## was
    # ## was

    epoch_a = epoch ## is

    train_envs = DummyVectorEnv([lambda: MEC_Env(conf_name=config,w=wi/100.0,fc=4e9,fe=2e9,edge_num=edge_num) for _ in range(train_num)])
    test_envs = DummyVectorEnv([lambda: MEC_Env(conf_name=config,w=wi/100.0,fc=4e9,fe=2e9,edge_num=edge_num) for _ in range(test_num)])

    buffer = ts.data.VectorReplayBuffer(buffer_size, train_num)  ## was
    train_collector = ts.data.Collector(policy, train_envs, buffer)
    test_collector = ts.data.Collector(policy, test_envs)
    train_collector.reset() ## added
    train_collector.collect(n_episode=train_num)

    def save_best_fn (policy):
        pass

    def test_fn(epoch, env_step):
        policy.actor.save_model('save/pth-e%d/'%(edge_num) + expn + '/w%03d/ep%02d-actor.pth'%(wi,epoch))
        policy.critic.save_model('save/pth-e%d/'%(edge_num) + expn + '/w%03d/ep%02d-critic.pth'%(wi,epoch))

    # def train_fn(epoch, env_step): ## was
    #     pass ## was
    last_logged_step = -1
    accum_mem = 0
    total_power_usage = 0
    def train_fn(epoch, env_step):
        global last_logged_step, accum_mem, total_power_usage
        if env_step > last_logged_step:
            current_mem = process.memory_info().rss  # bytes
            accum_mem += (current_mem / (1024.0 ** 2))
            # print(f"wi={wi}, env_step={env_step}, memory usage: {accum_mem:.2f} MB-seconds.\n")
            file.write(f"wi={wi}, env_step={env_step}, memory usage: {accum_mem:.2f} MB-seconds\n")

            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    ### Values for NVIDIA GeForce GTX 1070
                    P_idle = 10  # Idle power for GTX 1070 (in watts)
                    P_max = 150  # Maximum power for GTX 1070 (in watts)

                    info_gpu = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    gpu_utilization_percentage = float(info_gpu.stdout.strip())
                    # Apply the GPU power consumption formula (similar to CPU)
                    power_estimate = P_idle + (P_max - P_idle) * (gpu_utilization_percentage / 100)
                    total_power_usage += power_estimate
                else:
                    raise ValueError("No GPU found.")
            except (ImportError, ValueError):
                print("No GPU found.")
                # Fallback to CPU utilization if GPU is not available
                cpu_percent = psutil.cpu_percent(interval=0.1)
                # Apply the CPU power consumption formula
                power_estimate = P_idle + (P_max - P_idle) * (cpu_percent / 100)
                total_power_usage += power_estimate
            # print(f"wi={wi}, env_step={env_step}, power usage: {total_power_usage:.2f} W")
            file.write(f"wi={wi}, env_step={env_step}, power usage: {total_power_usage:.2f} W\n")

            last_logged_step = env_step

    def reward_metric(rews):
        return rews

    trainer = ts.trainer.OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch_a,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        episode_per_collect=episode_per_collect,
        save_best_fn=save_best_fn,
        logger=logger,
        train_fn=train_fn,
        test_fn=test_fn,
        test_in_train=False
    )
    result = trainer.run()
    file.write(f"===============================================\n")

file.close()