import torch
import numpy as np
from MuzeroHanoi import MuzeroHanoi

## ======= Set seeds for debugging =======
s = 7
torch.manual_seed(s)
np.random.seed(s)
## =======================

## ========= Useful variables: ===========
discount = 0.99
dirichlet_alpha = 0.25
temperature = 1 # 
n_mcts_simulations = 25
batch_s = 1
lr = 0.0001
muzero_train_steps=1
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_acc = 10

## ========= Env variables ========
N = 3 
max_steps= 200

muzero = MuzeroHanoi(N,discount, dirichlet_alpha, n_mcts_simulations, batch_s, lr, dev, max_steps)

## CHANGE: this should all happen within MuzeroHanoi class =========
accuracy = [] # in terms of mean n. steps to solve task
value_loss,rwd_loss,pi_loss = [],[],[]

steps, states, rwds, actions, pi_probs, mc_returns = muzero.play_game(temperature,deterministic=False)
for ep in range(3000):
    #steps, states, rwds, actions, pi_probs, mc_returns = muzero.play_game(temperature,deterministic=False)
    accuracy.append(steps)

    for t in range(muzero_train_steps):
        loss, v_loss, r_loss, p_loss  = muzero.train(states, rwds, actions, pi_probs, mc_returns,  unroll_n_steps=5)  
        # Update network
        muzero.networks.update(loss)

    value_loss.append(v_loss)
    rwd_loss.append(r_loss)
    pi_loss.append(p_loss)


    if ep % print_acc == 0:
        print("Episode: ",ep)
        print("Mean acc: ",sum(accuracy) / len(accuracy))
        print("V loss: ", sum(value_loss)/len(value_loss))
        print("rwd loss: ", sum(rwd_loss)/len(rwd_loss))
        print("Pi loss: ", sum(pi_loss)/len(pi_loss))
        print("\n")
        accuracy = []
        value_loss,rwd_loss,pi_loss = [],[],[]


steps, states, rwds, actions, pi_probs, mc_returns = muzero.play_game(temperature,deterministic=True)
print(steps)

