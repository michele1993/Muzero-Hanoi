import torch
import numpy as np
from MuzeroHanoi import MuzeroHanoi

## ======= Set seeds for debugging =======
torch.manual_seed(9)
np.random.seed(9)
## =======================

## ========= Useful variables: ===========
discount = 0.99
dirichlet_alpha = 0.03
temperature = 1
n_mcts_simulations = 10
batch_s = 1
lr = 0.001
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## ========= Env variables ========
N = 3 
max_steps= 200

muzero = MuzeroHanoi(N,discount, dirichlet_alpha, n_mcts_simulations, batch_s, lr, dev, max_steps)

## CHANGE: this should all happen within MuzeroHanoi class =========
for ep in range(10000):
    steps, states, rwds, actions, pi_probs, mc_returns = muzero.play_game(temperature)
    loss = muzero.train(states, rwds, actions, pi_probs, mc_returns,  unroll_n_steps=5)  
    muzero.networks.update(loss)
    print(steps)
