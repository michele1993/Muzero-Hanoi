import torch
import numpy as np
from MuzeroHanoi import MuzeroHanoi
from buffer import Buffer

## ======= Set seeds for debugging =======
s = 11
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
buffer_size = 50000
priority_replay = True
lr = 0.002
muzero_train_steps=1 # Muzero training steps x env step
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_acc = 100

## ========= Env variables ========
N = 3 
max_steps= 200
d_action = 6 # n. of action available in each state for Tower of Hanoi (including illegal ones)

muzero = MuzeroHanoi(N,discount, dirichlet_alpha, n_mcts_simulations, unroll_n_steps, d_action, batch_s, lr, max_steps, dev)
buffer = Buffer(buffer_size, unroll_n_steps, d_state=1, d_action=d_action, device=dev) #Note: d_state=1 since store the indexes of states

## CHANGE: this should all happen within MuzeroHanoi class =========
accuracy = [] # in terms of mean n. steps to solve task
value_loss,rwd_loss,pi_loss = [],[],[]

#steps, states, rwds, actions, pi_probs, mc_returns = muzero.play_game(temperature,deterministic=False)
for ep in range(1,episodes):

    steps, states, rwds, actions, pi_probs, mc_returns, priorities = muzero.play_game(temperature,deterministic=False)

    # Store episode in buffer only if successful
    if mc_returns[-1,0] > 0:
        buffer.add(states, rwds, actions, pi_probs, mc_returns, priorities)
    accuracy.append(steps)

    if ep > pre_training:
        for t in range(muzero_train_steps):

            if priority_replay:
                states, rwds, actions, pi_probs, mc_returns, priority_indx, priority_w = buffer.priority_sample(batch_s)
            else:
                states, rwds, actions, pi_probs, mc_returns = buffer.uniform_sample(batch_s)

            raise NotImplemented(" NOTE: Must implement train() update usign weights AND update priorities of the buffer - look at calc_loss()")

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

steps, states, rwds, actions, pi_probs, mc_returns = muzero.play_game(temperature,deterministic=True)
print(steps)
