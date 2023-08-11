import torch
import numpy as np
from MuzeroHanoi import MuzeroHanoi
from buffer import Buffer

## ======= Set seeds for debugging =======
s = 7
torch.manual_seed(s)
np.random.seed(s)
## =======================

## ========= Useful variables: ===========
episodes = 10000
pre_training = 10
discount = 0.8
dirichlet_alpha = 0.25
temperature = 1 # 
n_mcts_simulations = 25 # during acting n. of mcts passes for each step
unroll_n_steps = 5
batch_s = 1000
buffer_size = 20000
lr = 0.004
muzero_train_steps=1 # Muzero training steps x env step
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_acc = 100

## ========= Env variables ========
N = 3 
max_steps= 200
d_action = 6 # n. of action available in each state for Tower of Hanoi (including illegal ones)

muzero = MuzeroHanoi(N,discount, dirichlet_alpha, n_mcts_simulations, unroll_n_steps, d_action, batch_s, lr, max_steps, dev)
buffer = Buffer(buffer_size, unroll_n_steps, d_state=1, d_action=d_action, device=dev) # d_state=1 since store the indexes of states

## CHANGE: this should all happen within MuzeroHanoi class =========
accuracy = [] # in terms of mean n. steps to solve task
value_loss,rwd_loss,pi_loss = [],[],[]

#steps_2, _, _, actions_2, _, _ = muzero.play_game(temperature,deterministic=True)

for ep in range(1,episodes):
    steps, states, rwds, actions, pi_probs, mc_returns = muzero.play_game(temperature,deterministic=False)
    # Store episode in buffer only if successful
    if mc_returns[-1,0] > 0:
        buffer.add(states, rwds, actions, pi_probs, mc_returns)
    #print(type(states),rwds.size(),actions.size(),pi_probs.size(),mc_returns.size())
    accuracy.append(steps)

    if ep > pre_training:
        for t in range(muzero_train_steps):
            states, rwds, actions, pi_probs, mc_returns = buffer.sample_transitions(batch_s)
            loss, v_loss, r_loss, p_loss  = muzero.train(states, rwds, actions, pi_probs, mc_returns)
            # Update network
            muzero.networks.update(loss)

        value_loss.append(v_loss)
        rwd_loss.append(r_loss)
        pi_loss.append(p_loss)


    if ep % print_acc == 0:
        print("Episode: ",ep)
        print("Mean acc: ",sum(accuracy) / print_acc)
        print("V loss: ", sum(value_loss)/print_acc)
        print("rwd loss: ", sum(rwd_loss)/print_acc)
        print("Pi loss: ", sum(pi_loss)/print_acc)
        print("\n")
        accuracy = []
        value_loss,rwd_loss,pi_loss = [],[],[]

#print(steps_2)
steps, states, rwds, actions, pi_probs, mc_returns = muzero.play_game(temperature,deterministic=True)
print(steps)

#print(actions[:,0])
#print(actions_2[:,0])
