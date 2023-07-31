import numpy as np
import math
class Node:
    """ Nodes in the MCTS"""

    def __init__(self, prior=None, move=None, parent=None):
        """
        Args:
            prior (float): probability of the node for a specific action (i.e., based on the parent), 'None' for root node
            move: action associated to the prior probability (i.e. what action led to this node, I think)
            parent: the parent node, 'None' for root node
        """
        self.prior = prior
        self.move = move
        self.parent = parent
        self.is_expanded = False # has the node been expanded into children

        self.N = 0 # n of visits
        self.W = 0.0 # summed action value
        self.rwd = 0.0
        self.h_state = None

        self.children = []
    
    def expand(self, prior, h_state, reward):
        """Expand all actions, including illegal actions.

        Args:
            prior: 1D numpy.array contains prior probabilities of the state for all actions,
                whoever calls this should pre-processing illegal actions first.
            hidden_state: the corresponding hidden state for current timestep.
            reward: the reward for current timestep.
        """

        if self.is_expanded:
            raise RuntimeError("Node has already been expanded")

        self.h_state = h_state
        self.rwd = reward

        # Expand into all possible children
        for action in range(0, prior.shape[0]):
            child = Node(prior=prior[action], move=action, parent=self)
            self.children.append(child)
                
        self.is_expanded = True        
    
    def best_child(self, config):
        """ Returns best child node with maximum action value Q plus an upper confidence bound U.
        Args:
            config: a MCTS instance.
        Returns:
            The best child node.
        """    

        if not self.is_expanded:
            raise ValueError('Expand leaf node first.')

        ucb_results = self.child_Q(config) + self.child_U(config)

        # Break ties when have multiple 'max' value.
        a_indx = np.random.choice(np.where(ucb_results == ucb_results.max())[0])

        return self.children[a_indx] # return best child

    def child_Q(self, config):
        """Returns a 1D numpy.array contains mean action value for all children.
        Returns:
            a 1D numpy.array contains Q values for each of the children nodes.
        """

        # Compute normalised value for each children, if never visisted value=0
        Q = []
        for child in self.children:
            if child.N >0:
                Q.append(child.rwd + config.discount * child.Q)
            else:
                Q.append(0)
        return np.array(Q)
    
    def child_U(self, config):
        """ Returns a 1D numpy.array contains UCB score for all children (i.e., the exploration bonus).
         Returns:
             a 1D numpy.array contains UCB score for each of the children nodes.
        """
        U = []
        for child in self.children:
            # Note self.N refers to the current node, so to the sum of all actions counts
            # child.N refers to the child count, so how many times the action leading to the child has been taken
            w = (( math.log((self.N + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init) * math.sqrt(self.N) / (child.N +1))
            U.append(child.prior * w)
        return np.array(U)    

    def backup(self, value, config):
        """Update statistics of the this node and all travesed parent nodes.
        Args:
            value: the predicted state value from NN network.
            config: instance of MCTS object to get config
        """

        current = self

        while current is not None:
            current.W += value
            current.N +=1

            value = current.rwd + config.discount * value

            current = current.parent
    @property 
    def Q(self):
        """ Returns the mean action value Q(s, a)."""

        if self.N == 0:
            return 0.0
        return self.W / self.N

    @property
    def child_N(self):
        """Returns a 1D numpy.array contains visits count for all child."""
        return np.array([child.N for child in self.children])

    @property
    def has_parent(self):
        """ Returns boolean if node has parent """
        return isinstance(self.parent, Node)
