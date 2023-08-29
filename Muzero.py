import torch
import torch.nn.functional as F
import numpy as np
import logging
from utils import compute_MCreturns, adjust_temperature
from networks import MuZeroNet
from MCTS.mcts import MCTS
from buffer import Buffer

class Muzero():

    def __init__(self,
                 env,
                 s_space_size,
                 n_action,
                 max_steps,
                 discount,
                 dirichlet_alpha,
                 n_mcts_simulations,
                 unroll_n_steps,
                 batch_s,
                 lr,
                 buffer_size,
                 priority_replay,
                 device,
                 n_ep_x_loop =1,
                 n_update_x_loop=1
                ):

        self.dev = device

        ## ========= Set env variables========
        self.env = env
        self.n_ep_x_loop = n_ep_x_loop # how many env eps we collect for each training loop
        self.discount = discount
        self.max_steps = max_steps
        self.n_action = n_action

        ## ======= Set MuZero training variables =======
        self.unroll_n_steps = unroll_n_steps
        self.n_update_x_loop = n_update_x_loop # Muzero training steps x n. of training loop
        self.batch_s = batch_s

        ## ========== Initialise MuZero components =======
        self.mcts = MCTS(discount=self.discount, root_dirichlet_alpha=dirichlet_alpha, n_simulations=n_mcts_simulations, batch_s=batch_s, device=self.dev)
        # NOTE: use latent reprs size = env state size (i.e., reasonable)
        self.networks = MuZeroNet(rpr_input_s= s_space_size, action_s = self.n_action, reprs_output_size=64, lr=lr).to(self.dev) # NOTE: CHANGED NOT TO LOGITS

        ## ========== Initialise buffer ========
        self.buffer = Buffer(buffer_size, unroll_n_steps, d_state=s_space_size, n_action=self.n_action, device=self.dev) 
        self.priority_replay = priority_replay
    
    def training_loop(self, n_loops, pre_training, print_acc = 100):

        logging.info('Training started \n')

        accuracy = [] # in terms of mean n. steps to solve task
        tot_accuracy = [] 
        value_loss,rwd_loss,pi_loss = [],[],[]

        for n in range(1,n_loops):
            ep_accuracy = []
            for ep in range(self.n_ep_x_loop):
                # Play one episode
                steps, states, rwds, actions, pi_probs, mc_returns, priorities = self._play_game(deterministic=False)
                ep_accuracy.append(steps)
                # Store episode in buffer only if successful
                if mc_returns[-1,0] > 0:
                    self.buffer.add(states, rwds, actions, pi_probs, mc_returns, priorities)
            accuracy.append(sum(ep_accuracy)/self.n_ep_x_loop)

            # If time to train, train MuZero network
            if n > (pre_training//self.n_ep_x_loop):
                for t in range(self.n_update_x_loop):
                    if self.priority_replay:
                        states, rwds, actions, pi_probs, mc_returns, priority_indx, priority_w = self.buffer.priority_sample(self.batch_s)
                    else:
                        states, rwds, actions, pi_probs, mc_returns = self.buffer.uniform_sample(self.batch_s)
                        priority_w, priority_indx = None,None

                    # Update network
                    new_priority_w, v_loss, r_loss, p_loss  = self._update(states, rwds, actions, pi_probs, mc_returns, priority_w)
                    # Update buffer priorities 
                    self.buffer.update_priorities(priority_indx, new_priority_w)

                value_loss.append(v_loss)
                rwd_loss.append(r_loss)
                pi_loss.append(p_loss)

            if n % print_acc == 0:
                mean_acc = sum(accuracy) / print_acc
                logging.info(f'| Training Loop: {n} | Mean accuracy: {mean_acc} \n')
                logging.info(f"Mean acc:  {mean_acc}")
                logging.info(f"V loss:  {sum(value_loss)/print_acc}")
                logging.info(f"rwd loss:  {sum(rwd_loss)/print_acc}")
                logging.info(f"Pi loss:  {sum(pi_loss)/print_acc}")
                logging.info("\n")
                tot_accuracy.append(mean_acc)
                accuracy = []
                value_loss,rwd_loss,pi_loss = [],[],[]

        return tot_accuracy
    
    def _play_game(self, deterministic=False):

        # Initialise list to store game variables
        episode_state = []
        episode_action = []
        episode_rwd = []
        episode_piProb = []
        episode_rootQ = []

        c_state = self.env.reset()
        done=False
        step = 0

        while not done:
            # Run MCTS to select the action
            action, pi_prob, rootNode_Q = self.mcts.run_mcts(c_state, self.networks, temperature=adjust_temperature(step), deterministic=deterministic)
            # Take a step in env based on MCTS action
            n_state, rwd, done, illegal_move = self.env.step(action)
            step +=1

            if step == self.max_steps:
               done = True
            
            # Store variables for training
            # NOTE: not storing the last terminal state (don't think it is needed)
            episode_state.append(c_state)
            episode_action.append(action)
            episode_rwd.append(rwd)
            episode_piProb.append(pi_prob)
            episode_rootQ.append(rootNode_Q)

            # current state becomes next state
            c_state = n_state


        #Compute MC return for each state
        episode_returns = compute_MCreturns(episode_rwd, self.discount)
        priorities = np.abs(np.array(episode_returns) - np.array(episode_rootQ)) 

        # Organise ep. trajectory into appropriate transitions for training - i.e. each transition should have unroll_n_steps associated transitions for training
        states, rwds, actions, pi_probs, mc_returns = self.organise_transitions(episode_state, episode_rwd, episode_action, episode_piProb, episode_returns)
        
        return step, states, rwds, actions, pi_probs, mc_returns, priorities   

    def _update(self, states, rwds, actions, pi_probs, mc_returns, priority_w):
      # TRIAL: expands all states (including final ones) of mcts_steps for simplicty for steps after terminal just map everything to zero   
      # if does not work then need to adapt for terminal states not to expand tree of mcts_steps

      rwd_loss, value_loss, policy_loss = (0, 0, 0)

      h_states = self.networks.represent(states.float())

      tot_pred_values = []

      for t in range(self.unroll_n_steps):

          pred_pi_probs, pred_values = self.networks.prediction(h_states)

          # Convert action to 1-hot encoding
          oneH_action = torch.nn.functional.one_hot(actions[:,t], num_classes=self.networks.num_actions).squeeze().to(self.dev)
          #oneH_action = actions[:,t].squeeze().to(self.dev)
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
        episode_action += [np.random.randint(0,self.n_action)] * self.unroll_n_steps # select uniform random action for unroll_n_steps over the end
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

