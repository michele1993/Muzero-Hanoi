import torch
import numpy as np

class Buffer:
    def __init__(self, size, unroll_n_steps, d_state, n_action, device, priority_exponent=1, importance_sampling_exponent=0):
        """ data buffer that holds transitions already organised for unroll_steps during training in np.arrays"""

        self.dev = device
        self._priority_exponent = priority_exponent
        self._importance_sampling_exponent = importance_sampling_exponent 

        #Dimensions
        self.size = size
        self.unroll_n_steps = unroll_n_steps
        self.n_action = n_action
        self.d_state = d_state
        
        # Storage variables
        self.states = np.zeros((size,d_state), dtype=int) #NOTE: state is not unrolled, just need one initial state for each step
        self.rwds = np.zeros((size,unroll_n_steps))
        self.actions = np.zeros((size, unroll_n_steps), dtype=int) # store action indx
        self.pi_probs = np.zeros((size, unroll_n_steps, n_action))
        self.mc_returns = np.zeros((size, unroll_n_steps))

        self.priorities = np.zeros((size,))

        # Other attributes
        self.ptr = 0 # counter for n. transitions currently inside buffer
        self.is_full = False # needed for when buffer is full to estimate n. of elements
    
    def _add(self, buffer, transitions):
        """ General purpose methods to add any type of data to the buffer"""
        n = transitions.shape[0]
        excess = self.ptr + n - self.size # compute by how many elements the buffer is execeeded

        # Compute two indexes, a: n. element that fit in the buffer, b:  n. elements that exceed the buffer
        if excess <= 0: # all elements fit in the buffer
            a,b = n,0 
        else:
            a,b = n - excess, excess 

        # Store data based on appropriate indexes
        buffer[self.ptr:self.ptr+a] = transitions[:a]        
        buffer[:b] = transitions[a:]
    
    def add(self, states, rwds, actions, pi_probs, mc_returns, priorities):
        """ Add transitions to the buffer """
        n_transitions = states.shape[0]

        assert n_transitions <= self.size

        self._add(self.states, states)
        self._add(self.rwds, rwds)
        self._add(self.actions, actions)
        self._add(self.pi_probs, pi_probs)
        self._add(self.mc_returns, mc_returns)

        # Add priorities
        self._add(self.priorities, priorities)

        if self.ptr + n_transitions >= self.size:
            self.is_full = True
        self.ptr = (self.ptr + n_transitions) % self.size

    def uniform_sample(self, batch_s):
        """ return a random batch of size batch_s for each data element"""
        num = len(self)
        indx = np.random.randint(0,num,size=batch_s)
        indx = np.random.choice(np.arange(num), size=batch_s, replace=True)
        states, rwds, actions, pi_probs, mc_returns = self._sample(indx)

        return states, rwds, actions, pi_probs, mc_returns

    def priority_sample(self, batch_s):
        """ return a random batch of size batch_s for each data element"""
        num = len(self)
        priorities = self.priorities[:num]**self._priority_exponent # take priorities for all elements from buffer
        priorities_probs = priorities / np.sum(priorities) # compute a probabilities based on priorities

        indx = np.random.choice(np.arange(num), size=batch_s, replace=True, p=priorities_probs)
        states, rwds, actions, pi_probs, mc_returns = self._sample(indx)

        #Compute importance weights to scale gradient of each element in the batch based on priority
        weights = ((1.0 / self.size) / priorities_probs[indx]) ** self._importance_sampling_exponent
        weights /= np.max(weights)

        weights = torch.from_numpy(weights).to(self.dev)

        return states, rwds, actions, pi_probs, mc_returns, indx, weights

    def _sample(self, indx):
        #Sample transitions from buffer
        states, rwds = torch.from_numpy(self.states[indx]).to(self.dev), torch.from_numpy(self.rwds[indx]).to(self.dev)
        actions = torch.from_numpy(self.actions[indx]).to(self.dev) 
        pi_probs, mc_returns = torch.from_numpy(self.pi_probs[indx]).to(self.dev), torch.from_numpy(self.mc_returns[indx]).to(self.dev)

        return states, rwds, actions, pi_probs, mc_returns


    def update_priorities(self, indx, new_priorities):
        """ Update priorities in the buffer, after the network has been update"""
        if indx is not None:
            assert np.isfinite(new_priorities).all() and (new_priorities>0.0).any(), 'Priorities must be finite and positive.'
            self.priorities[indx] = new_priorities

    def __len__(self):
        return self.size if self.is_full else self.ptr
