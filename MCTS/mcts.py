import torch
from MCTS.node import Node
import numpy as np

class MCTS():
    """ Define class to run MCTS"""

    def __init__(
        self,
        discount,
        root_dirichlet_alpha,
        n_simulations,
        batch_s,
        lr,
        device,
        h_dim=64,
        clip_grad=True,
        root_exploration_eps = 0.25,
    ):
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        self.discount = discount
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_eps = root_exploration_eps
        self.n_simulations = n_simulations
        self.batch_s = batch_s
        self.dev = device
            
    def run_mcts(self,state, network, temperature, deterministic=False):           
        """ Run MCT
        Args:
            state: current obs from the environment.
            network: MuZeron network instance
            tempreature: a parameter controls the level of exploration when generate policy action probabilities after MCTS search.
            deterministic: after the MCTS search, choose the child node with most visits number to play in the real environment,
         instead of sample through a probability distribution, default off.
         Returns:
            tuple contains:
            a integer indicate the sampled action to play in the environment.
            a 1D numpy.array search policy action probabilities from the MCTS search result.
            a float represent the search value of the root node.
        """

        # Create root node
        state = torch.from_numpy(state).to(self.dev).float()
        h_state, rwd, pi_probs, value = network.initial_inference(state)
        prior_prob = pi_probs
        root_node = Node(prior=0.0) # the root node does not have prior probs since it is the root

        # Add dirichlet noise to the prior probabilities to root node.
        if not deterministic and self.root_dirichlet_alpha > 0.0 and self.root_exploration_eps > 0.0:
            prior_prob = self.add_dirichlet_noise(prior_prob, eps=self.root_exploration_eps, alpha=self.root_dirichlet_alpha)
        
        # fill node with data and add "children actions", by expanding
        root_node.expand(prior_prob,h_state, rwd) 

        for _ in range(self.n_simulations):
            ## ====  Phase 1 - Select ====
            # Reset initial node to root for each mcts simulation
            node = root_node 

            # Select best child node until reach a leaf
            while node.is_expanded:
                node = node.best_child(self) # pass MCTS object to have access to the config

            ## ==== Phase 2 - Expand and evaluation ==== 
            h_state = torch.from_numpy(node.parent.h_state).to(self.dev).float() # node.parent because while loop ends at not expanded (best) child
            action = torch.tensor([node.move], device=self.dev)

            # Convert action to 1-hot encoding
            action = torch.nn.functional.one_hot(action, num_classes=network.num_actions).squeeze()

            # Take a step in latent space
            h_state, rwd, pi_probs, value = network.recurrent_inference(h_state, action) # compute latent state for best action (child)

            node.expand(prior_prob, h_state, rwd) #NOTE: I don't understand prior prob here, shouldn't come from a pi_probs ?

            ## ==== Phase 3 - Backup on leaf node ====
            node.backup(value, self)
        
        # Play: generate action prob from the root node to be played in the env.
        child_visits = root_node.child_N

        pi_prob = self.generate_play_policy(child_visits, temperature)

        if deterministic:
            # Choose the action with the most visit n.
            action_idx = np.argmax(child_visits)
        else:
            # Sample a action.
            action_idx = np.random.choice(np.arange(pi_prob.shape[0]), p=pi_prob)

        action = root_node.children[action_idx].move        

        return (action, pi_prob, root_node.Q)

    def add_dirichlet_noise(self, prob, eps=0.25, alpha=0.25):
        """Add dirichlet noise to a given probabilities.
        Args:
            prob: a numpy.array contains action probabilities we want to add noise to.
            eps: epsilon constant to weight the priors vs. dirichlet noise.
            alpha: parameter of the dirichlet noise distribution.

        Returns:
            action probabilities with added dirichlet noise.
        """    

        alphas = np.ones_like(prob) * alpha
        noise = np.random.dirichlet(alphas)
        noised_prob = (1 - eps) * prob + eps * noise

        return noised_prob
                    
    def generate_play_policy(self,visits_count, temperature):
        """ Returns policy action probabilities proportional to their exponentiated visit count during MCTS 
        Args:
            visits_count: a 1D numpy.array contains child node visits count.
            temperature: a parameter controls the level of exploration.
        Returns:        
            a 1D numpy.array contating the action prob for real env
        """

        if not 0.0 <= temperature <= 1.0:
            raise ValueError(f"Expect `temperature` to be in the range [0.0, 1.0], got {temperature}")

        visits_count = np.asarray(visits_count)

        if temperature > 0.0:
            # limit the exponent in the range of [1.0, 5.0]
            # to avoid overflow when doing power operation over large numbers
            exp = max(1.0,min(5.0, 1.0/temperature))
            visits_count = np.power(visits_count,exp)

        return visits_count / np.sum(visits_count)
