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
max_steps = 200
env = TowersOfHanoi(N=N,max_steps=max_steps)
s_space_size = env.oneH_s_size 
n_action = 6 # n. of action available in each state for Tower of Hanoi (including illegal ones)
discount = 0.8
n_mcts_simulations_range = [5,10,20,30,40,50,80,100] #25 during acting n. of mcts passes for each step
lr = 0
TD_return = True # needed for NN to use logit to scalar transform
run_n = 1

## ======= Experiemnt variables ===============
pertub_p = True # perturb p() function in Muzero (i.e. latent policy)
perturb_p_magnitude = 0.5 # magnitude of perturbation to p()

## ====== Log command line =====
command_line = f'Env: {env_name}, Episodes: {episode},  Latent pol perturb: {pertub_p} and magn. {perturb_p_magnitude}, n. MCTS: {n_mcts_simulations_range}, N. disks: {N}'
logging.info(command_line)

# Load pre-trained NN
mcts = MCTS(discount=discount, root_dirichlet_alpha=dirichlet_alpha, n_simulations=n_mcts_simulations_range[0], batch_s=1, device=dev)
networks = MuZeroNet(rpr_input_s= s_space_size, action_s = n_action, lr=lr,TD_return=TD_return, perturb_p=pertub_p,perturb_p_magnitude=perturb_p_magnitude).to(dev) 
model_dict = torch.load(f'/Users/px19783/code_repository/cerebellum_project/latent_planning/results/{run_n}/muzero_model.pt')
networks.load_state_dict(model_dict['Muzero_net'])
networks.optimiser.load_state_dict(model_dict['Net_optim'])

# Try randomly initialising policy net
#networks.policy_net.apply(networks.reset_param)

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

        # Start from an initial random state (apart from goal)
        c_state = env.random_reset()
        # Compute min n. moves from current random state
        min_n_moves = hanoi_solver(env.current_state()) # pass state represt not in one-hot form  

        done=False
        step = 0
        while not done:
            # Run MCTS to select the action
            action, pi_prob, rootNode_Q = mcts.run_mcts(c_state, networks, temperature=temperature, deterministic=False)
            # Take a step in env based on MCTS action
            n_state, rwd, done, illegal_move = env.step(action)
            step +=1
            
            # Store variables for training
            # NOTE: not storing the last terminal state (don't think it is needed)
            episode_state.append(c_state)
            episode_action.append(action)
            episode_rwd.append(rwd)
            episode_piProb.append(pi_prob)
            episode_rootQ.append(rootNode_Q)

            # current state becomes next state
            c_state = n_state

        errors.append(step - min_n_moves)

    data.append([n,sum(errors)/len(errors)])

print(data)
## ===== Save results =========
file_indx = 1 
# Create directory to store results
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'results',str(file_indx))
# Create directory if it did't exist before
#os.makedirs(file_dir, exist_ok=True)

# Store command line
#with open(os.path.join(file_dir,'commands.txt'), 'w') as f:
    #f.write(command_line)
# Store accuracy
acc_dir = os.path.join(file_dir,'acting_accuracy.pt')
#torch.save(torch.tensor(tot_acc),acc_dir)
