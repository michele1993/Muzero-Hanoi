import sys
sys.path.append('/Users/px19783/code_repository/cerebellum_project/latent_planning')
import os
import torch
import numpy as np
import logging
from env.hanoi import TowersOfHanoi
from MCTS.mcts import MCTS
from networks import MuZeroNet
from utils import setup_logger
from env.hanoi_utils import hanoi_solver 

""" Try to compute a seq of actions from a single Muzero planning step (MCTS), rather than planning at each step"""

## ======= Set seeds for debugging =======
s = 1 # seed
torch.manual_seed(s)
np.random.seed(s)
setup_logger(s)

## ========= Useful variables: ===========
episode = 100
dirichlet_alpha = 0
temperature = 0#0.1
discount= 0.8
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## ========= Initialise env ========
env_name = 'Hanoi' 
N = 3 
max_steps = 30
env = TowersOfHanoi(N=N,max_steps=max_steps)
s_space_size = env.oneH_s_size 
n_action = 6 # n. of action available in each state for Tower of Hanoi (including illegal ones)
discount = 0.8
n_mcts_simulations_range = [1000]#[5,10,30,50,80,110,150] #25 during acting n. of mcts passes for each step
lr = 0
TD_return = True # needed for NN to use logit to scalar transform
model_run_n = 1 
save_results = False
seed_indx = s
max_plan_len = 7


# ======= Load pre-trained NN ========
mcts = MCTS(discount=discount, root_dirichlet_alpha=dirichlet_alpha, n_simulations=n_mcts_simulations_range[0], batch_s=1, device=dev)
networks = MuZeroNet(rpr_input_s= s_space_size, action_s = n_action, lr=lr,TD_return=TD_return).to(dev) 
model_dict = torch.load(f'/Users/px19783/code_repository/cerebellum_project/latent_planning/results/{model_run_n}/muzero_model.pt')
networks.load_state_dict(model_dict['Muzero_net'])
networks.optimiser.load_state_dict(model_dict['Net_optim'])

## ======== Experimental set-up ==========
# Set variables below to run different experiments

# Reset latent policy
reset_latent_policy = False
if reset_latent_policy:
    networks.policy_net.apply(networks.reset_param) # Try randomly initialising policy net, to simulate cerebellum damage

# Reset latent values
reset_latent_values = False
if reset_latent_values:
    networks.value_net.apply(networks.reset_param)

reset_latent_rwds = False
if reset_latent_rwds:
    networks.rwd_net.apply(networks.reset_param)

## ------ Define starting states for additional analysis ----
start = None # set to None for random starting state
if start is not None:
    # I specifically selected states which are not ecounted during the optimal traject from the training starting state
    # ES: early state, MS: mid state, LS: late state
    if start ==0:
        init_state = (2,2,0) # 7 moves away from goal 
        file_indx = 'ES'
    elif start==1:
        init_state = (0,0,2) # 3 moves away from goal for N=3
        file_indx = 'MS'
    elif start ==2:
        init_state = (1,2,2) # 1 move away 
        file_indx = 'LS'
    init_state_idx = env.states.index(init_state)
    env.init_state_idx = init_state_idx

# Start from random state
else: 
    file_indx = 'RandState'

## ====== Log command line =====
command_line = f'Seed: {s}, Env: {env_name}, Run type: {file_indx}, Episodes: {episode},  Reset latent pol: {reset_latent_policy}, Reset latent value: {reset_latent_values}, Reset latent rwd: {reset_latent_rwds}, n. MCTS: {n_mcts_simulations_range}, N. disks: {N}'
logging.info(command_line)

## ======== Run acting ==========
data = []
for n in n_mcts_simulations_range:
    errors = []
    mcts.n_simulations = n
    for ep in range(episode):
        # Initialise list to store game variables
        episode_state = []
        episode_action = []
        episode_rwd = []
        episode_piProb = []
        episode_rootQ = []

        if start is not None:
            # Reset from pre-defined initial state
            c_state = env.reset()
        else:            
            # Start from an initial random state (apart from goal)
            c_state = env.random_reset()

        # Compute min n. moves from current random state
        min_n_moves = hanoi_solver(env.current_state()) # pass state represt not in one-hot form  

        done=False
        step = 0
        # Run MCTS to select the sequence of actions
        action, pi_prob, rootNode_Q = mcts.run_mcts(c_state, networks, temperature=temperature, deterministic=False)
        action_seq = mcts.return_latent_actions()
        action_seq = action_seq[:max_plan_len]
        print("Ep: ",ep,"; Action seq length", len(action_seq), "; init state: ",c_state)
        t = 0
        while not done:

            # Take a step in env based on MCTS action
            n_state, rwd, done, illegal_move = env.step(action_seq[t].item())
            step +=1
            t+=1
            
            # Store variables for training
            # NOTE: not storing the last terminal state (don't think it is needed)
            episode_state.append(c_state)
            episode_action.append(action)
            episode_rwd.append(rwd)
            episode_piProb.append(pi_prob)
            episode_rootQ.append(rootNode_Q)

            # current state becomes next state
            c_state = n_state

            # If action seq finishes before done, re-plan
            if t >= len(action_seq) and not done:
                action, pi_prob, rootNode_Q = mcts.run_mcts(c_state, networks, temperature=temperature, deterministic=False)
                action_seq = mcts.return_latent_actions()[:max_plan_len]
                print("\n Re-plan Action seq length ", len(action_seq), '\n')
                t = 0

        print(step)
        errors.append(step - min_n_moves)

    data.append([n,sum(errors)/len(errors)])

print(data)
## ===== Save results =========
# Create directory to store results
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results',str(seed_indx),str(file_indx))
# Create directory if it did't exist before
os.makedirs(file_dir, exist_ok=True)

label_1,label_2,label_3='','',''
if reset_latent_policy or reset_latent_values or reset_latent_rwds:
    if reset_latent_policy:
        label_1 = 'ResetLatentPol_'
    if reset_latent_values:
        label_2 = 'ResetLatentVal_'
    if reset_latent_rwds:
        label_3= 'ResetLatentRwd_'

# Allow to combine reset rwd with other reset
label = label_1 + label_2 + label_3

if label == '':
    label='Muzero_'

acc_dir = os.path.join(file_dir,label+'actingAccuracy.pt')

# Store command line
if save_results:
    with open(os.path.join(file_dir,label+'commands.txt'), 'w') as f:
        f.write(command_line)
    # Store accuracy
    torch.save(torch.tensor(data),acc_dir)
