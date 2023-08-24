import torch
import numpy as np
import logging
from env.hanoi import TowersOfHanoi
from Muzero import Muzero
from utils import setup_logger

def get_env(env_name):
    if env_name == 'Hanoi':
        N = 3 
        env = TowersOfHanoi(N)
        max_steps= 200
        s_space_size = env.oneH_s_size 
        n_action = 6 # n. of action available in each state for Tower of Hanoi (including illegal ones)
        max_steps= max_steps
    else:        
        raise NotImplementedError
        #env = gym.make(env_name)
    return env, s_space_size, n_action, max_steps


## ======= Set seeds for debugging =======
s = 11 # seed
torch.manual_seed(s)
np.random.seed(s)
setup_logger(s)
## =======================

## ========= Useful variables: ===========
episodes = 10000
pre_training = 10
discount = 0.8
dirichlet_alpha = 0.25
temperature = 1 # 
n_mcts_simulations = 25 #25 during acting n. of mcts passes for each step
unroll_n_steps = 5
batch_s = 1000
buffer_size = 50000
priority_replay = True
lr = 0.002
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env_name = 'Hanoi'

logging.info(f'Env: {env_name}, Episodes: {episodes}, lr: {lr}, discount: {discount}, n. MCTS: {n_mcts_simulations}, batch size: {batch_s}, Priority Buff: {priority_replay}')

## ========= Initialise env ========
env, s_space_size, n_action, max_steps = get_env(env_name)

## ======== Initialise alg. ========
muzero = Muzero(env, s_space_size, n_action, max_steps, discount, dirichlet_alpha, n_mcts_simulations, unroll_n_steps, batch_s, lr, buffer_size, priority_replay, dev)

## ======== Run training ==========
tot_acc = muzero.training_loop(episodes, pre_training)
