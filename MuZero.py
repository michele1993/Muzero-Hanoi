import torch
from utils import compute_MCreturns

class MuzeroHanoi():

    def __init__(self,
                 N,
                 discount,
                 dirichlet_alpha,
                 n_mcts_simulations,
                 batch_s,
                 lr,
                 dev,
                 max_steps):

        self.discount = discount
        ## ========= Initialise env ========
        self.env = TowersOfHanoi(N)
        self.max_steps= max_steps
        state_space = env.states
        s_space_s = len(state_space)
        a_space_s = 6 # n. of action available in each state for Tower of Hanoi (including illegal ones)

        ## We assume a 1-to-1 correspondance between 1hot representations and the order of the original states in the tower of Hanoi
        ## namely, the vector [1,0,0,...] refers to the first generated state in the tower and so on
        self.one_hot_s = np.eye(s_space_s) # this creates a matrix whose columns represent a different 1hot vector for each state
        ## =================================

        ## ========== Initialise MuZero components =======
        self.mcts = MCTS(discount=self.discount, root_dirichlet_alpha=dirichlet_alpha, n_simulations=n_mcts_simulations, batch_s=batch_s,lr=lr, device=dev)
        self.networks = MuZeroNet(rpr_input_s= s_space_s, action_s = a_space_s)
        ## ===================================

    
    def train(self, n_episodes):

        memory_buffer = []
        for e in range(n_episodes):
            game_data = play_game()
            memory_buffer.append(game_data)
    
    def play_game(self):

        c_state,done = self.env.reset()
        c_s_indx = self.env.init_state_idx
        
        episode_state = []
        episode_action = []
        episode_rwd = []
        episode_piProb = []
        episode_rootQ = []

        step = 0
        while not done:

            oneH_c_s = self.one_hot_s[:,c_s_indx]

            # Run MCTS to select the action
            action, pi_prob, rootNode_Q = self.mcts.run_mcts(oneH_c_s, self.networks, temperature, deterministic=False)

            n_state, rwd, done, illegal_move = self.env.step(action)
            step +=1

            # Store variables for training
            episode_state.append(c_s_indx)
            episode_action.append(action)
            episode_rwd.append(rwd)
            episode_piProb.append(pi_prob)
            episode_rootQ.append(rootNode_Q)

            if step == self.max_steps:
               done = True

            if not done:
               n_s_indx = state_space.index(n_state) # Compute new c_state index if not done
            
            # current state becomes next state
            c_s_indx = n_s_indx

        MC_returs = compute_MCreturns(episode_rwd, self.discount)

        return  episode_state, episode_action, episode_rwd, episode_piProb, MC_returs, episode_rootQ
