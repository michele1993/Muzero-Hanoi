import torch
import torch.nn.functional as F
import numpy as np
from utils import compute_MCreturns, adjust_temperature
from networks import MuZeroNet
from MCTS.mcts import MCTS
from env.hanoi import TowersOfHanoi

class MuzeroHanoi():

    def __init__(self,
                 N,
                 discount,
                 dirichlet_alpha,
                 n_mcts_simulations,
                 unroll_n_steps,
                 d_action,
                 batch_s,
                 lr,
                 max_steps,
                 device,
                ):

        self.discount = discount
        self.unroll_n_steps = unroll_n_steps
        self.dev = device

        ## ========= Initialise env ========
        self.env = TowersOfHanoi(N)
        self.max_steps= max_steps
        self.state_space = self.env.states
        s_space_s = len(self.state_space)
        ## We assume a 1-to-1 correspondance between 1hot representations and the order of the original states in the tower of Hanoi
        ## namely, the vector [1,0,0,...] refers to the first generated state in the tower and so on
        self.one_hot_s = np.eye(s_space_s) # this creates a matrix whose columns represent a different 1hot vector for each state
        ## =================================

        ## ========== Initialise MuZero components =======
        self.mcts = MCTS(discount=self.discount, root_dirichlet_alpha=dirichlet_alpha, n_simulations=n_mcts_simulations, batch_s=batch_s, device=self.dev)
        self.networks = MuZeroNet(rpr_input_s= s_space_s, action_s = d_action,lr=lr).to(self.dev)
        ## ===================================

    
    def play_game(self, temperature, deterministic=False):

        # Initialise list to store game variables
        episode_state = []
        episode_action = []
        episode_rwd = []
        episode_piProb = []
        episode_rootQ = []

        c_state,done = self.env.reset()
        c_s_indx = self.env.init_state_idx
        
        step = 0
        while not done:

            oneH_c_s = self.one_hot_s[:,c_s_indx]

            # Run MCTS to select the action
            action, pi_prob, rootNode_Q = self.mcts.run_mcts(oneH_c_s, self.networks, temperature=adjust_temperature(step), deterministic=deterministic)

            n_state, rwd, done, illegal_move = self.env.step(action)
            n_s_indx = self.state_space.index(n_state) # Compute new c_state index if not done
            step +=1

            if step == self.max_steps:
               done = True
            
            # Store variables for training
            # NOTE: not storing the last terminal state (don't think it is needed)
            episode_state.append(c_s_indx)
            episode_action.append(action)
            episode_rwd.append(rwd)
            episode_piProb.append(pi_prob)
            episode_rootQ.append(rootNode_Q)

            # current state becomes next state
            c_s_indx = n_s_indx


        #Compute MC return for each state
        episode_returns = compute_MCreturns(episode_rwd, self.discount)
        priorities = np.abs(np.array(episode_returns) - np.array(episode_rootQ)) 

        # Organise ep. trajectory into appropriate transitions for training - i.e. each transition should have unroll_n_steps associated transitions for training
        states, rwds, actions, pi_probs, mc_returns = self.organise_transitions(episode_state, episode_rwd, episode_action, episode_piProb, episode_returns)
        
        return step, states, rwds, actions, pi_probs, mc_returns, priorities   

    def train(self, states, rwds, actions, pi_probs, mc_returns, priority_w):
      # TRIAL: expands all states (including final ones) of mcts_steps for simplicty for steps after terminal just map everything to zero   
      # if does not work then need to adapt for terminal states not to expand tree of mcts_steps

      rwd_loss, value_loss, policy_loss = (0, 0, 0)

      # Convert states to one hot encoding
      oneH_states = self.one_hot_s[:,states].T # flip to have shape [batch_s, n_dim]
      h_states = self.networks.represent(torch.from_numpy(oneH_states).float().to(self.dev))

      tot_pred_values = []

      for t in range(self.unroll_n_steps):
          pred_pi_probs, pred_values = self.networks.prediction(h_states)
          # Convert action to 1-hot encoding
          oneH_action = torch.nn.functional.one_hot(actions[:,t], num_classes=self.networks.num_actions).squeeze().to(self.dev)
          h_states, pred_rwds = self.networks.dynamics(h_states,oneH_action)

          # Scale the gradient for dynamics function by 0.5.
          h_states.register_hook(lambda grad: grad * 0.5)

          ## NOTE: For some reason F.mse_loss gives me data type error, so manually implement mse_loss
          value_loss += 0.5 * (pred_values.squeeze() - mc_returns[:,t])**2 
          rwd_loss += 0.5 * (pred_rwds.squeeze() - rwds[:,t])**2 
          policy_loss += F.cross_entropy(pred_pi_probs, pi_probs[:,t], reduction='none')

          tot_pred_values.append(pred_values)

      loss = value_loss + rwd_loss + policy_loss

      new_priorities = None # predefine new priorities in case no priority buffer

      if priority_w is not None:
          #Scale loss using importance sampling weights (based on priorty sampling from buffer)
          loss = loss * priority_w.detach()
          # Compute new priorities to update priority buffer
          with torch.no_grad():
              tot_pred_values = torch.stack(tot_pred_values, dim=1).squeeze(-1)
              new_priorities = (tot_pred_values[:,0] - mc_returns[:,0]).abs().cpu().numpy()

      loss = loss.mean()
      # Scale the loss by 1/unroll_steps.
      loss.register_hook(lambda grad: grad * (1/self.unroll_n_steps))

      # Update network
      self.networks.update(loss)

      return  new_priorities, value_loss.mean().detach() , rwd_loss.mean().detach(), policy_loss.mean().detach()   

    def organise_transitions(self, episode_state, episode_rwd, episode_action, episode_piProb, episode_mc_returns):
        """ Orgnise transitions in appropriate format, each state is associated to the n_step target values (pi_probs, rwds, MC_returs) for unroll_n_steps
            Returns:
                pi_probs: np.array(n_steps,unroll_n_steps,d_action)
                all others: np.array(n_steps,unroll_n_steps)
        """

        ## ===========================================
        # Try to remove unroll_n_steps terminal states, NOTE: Not a permanent solution, need terminal states to solve Hanoi with as few moves as possible
        #episode_state = episode_state[:-self.unroll_n_steps] # REMOVE this line, just to see if learning problem is driven by terminal states
        ## ===========================================

        n_states = len(episode_state) 

        # Add "padding" for terminal states, which don't have enough unroll_n_steps ahead
        # by trating states over the end of game as absorbing states
        # NOTE: This can cause issues, since a lot of cycles in Hanoi and zero is a real action
        episode_rwd += [0] * self.unroll_n_steps
        #episode_action += [0] * self.unroll_n_steps
        episode_action += [np.random.randint(0,6)] * self.unroll_n_steps # select uniform random action for unroll_n_steps over the end
        episode_mc_returns += [0] * self.unroll_n_steps
        absorbing_policy = np.ones_like(episode_piProb[-1]) / len(episode_piProb[-1])
        episode_piProb += [absorbing_policy] * self.unroll_n_steps

        # Initialise variables for storage
        rwds = np.zeros((n_states, self.unroll_n_steps))
        actions = np.zeros((n_states, self.unroll_n_steps),dtype=int)
        pi_probs = np.zeros((n_states, self.unroll_n_steps, len(episode_piProb[0])))
        mc_returns = np.zeros((n_states, self.unroll_n_steps))

        for i in range(n_states):
            rwds[i,:] = episode_rwd[i:i+self.unroll_n_steps]
            actions[i,:] = episode_action[i:i+self.unroll_n_steps]
            pi_probs[i,:,:] = episode_piProb[i:i+self.unroll_n_steps]
            mc_returns[i,:] = episode_mc_returns[i:i+self.unroll_n_steps]

        return np.array(episode_state), rwds, actions, pi_probs, mc_returns 

