import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

# NOTE: The original code uses an additional transformation from policy, reward and value logits (i.e. value_suppot_vector ect.) to the actual polcy, value etc.
# in the current implementation, we skip this additional transform. and map directly onto the values.

class MuZeroNet(nn.Module):
    """ Class containing all MuZero nets"""

    def __init__(
        self,
        rpr_input_s,
        action_s,
        lr,
        value_s = 1,
        reward_s = 1,
        h1_s = 64,#256
        h2_s = 64,
        weight_decay=1e-4
    ):
        super().__init__()

        self.num_actions = action_s

        self.representation_net = nn.Sequential(
            nn.Linear(rpr_input_s,h1_s),
            nn.ReLU(),
            nn.Linear(h1_s, h2_s),
        )

        self.dynamic_net = nn.Sequential(
            nn.Linear(h2_s + action_s, h1_s),
            nn.ReLU(),
            nn.Linear(h1_s, h2_s),
        )

        self.rwd_net = nn.Sequential(
            nn.Linear(h2_s, h1_s),
            nn.ReLU(),
            nn.Linear(h1_s, reward_s)
        )

        self.policy_net = nn.Sequential(
            nn.Linear(h2_s, h1_s),
            nn.ReLU(),
            nn.Linear(h1_s, action_s),
        )

        self.value_net = nn.Sequential(
            nn.Linear(h2_s, h1_s),
            nn.ReLU(),
            nn.Linear(h1_s, value_s),
        )

        self.optimiser = opt.Adam(self.parameters(),lr, weight_decay=weight_decay)

    @torch.no_grad()
    def initial_inference(self,x):
        """During self-play, given environment observation, use representation function to predict initial hidden state.
        Then use prediction function to predict policy probabilities and state value (on the hidden state)."""

        # Representation
        h_state = self.represent(x)

        # Prediction
        pi_logits, value = self.prediction(h_state)
        pi_probs = F.softmax(pi_logits,dim=-1) # NOTE: dim ?

        rwd = torch.zeros_like(value) # NOTE: Not sure why it doesn't predict rwd for initial inference

        pi_probs = pi_probs.squeeze(0).cpu().numpy()
        value = value.squeeze(0).cpu().item()
        rwd = rwd.squeeze(0).cpu().item()
        h_state = h_state.squeeze(0).cpu().numpy()

        return h_state, rwd, pi_probs, value
        
    @torch.no_grad()
    def recurrent_inference(self, h_state, action):
        """During self-play, given hidden state at timestep `t-1` and action at t,
        use dynamics function to predict the reward and next hidden state,
        and use prediction function to predict policy probabilities and state value (on new hidden state)."""

        # Dynamic 
        h_state, rwd = self.dynamics(h_state, action)

        # Prediction
        pi_logits, value = self.prediction(h_state)
        pi_probs = F.softmax(pi_logits,dim=-1) # NOTE: dim ?

        pi_probs = pi_probs.squeeze(0).cpu().numpy()
        value = value.squeeze(0).cpu().item()
        rwd = rwd.squeeze(0).cpu().item()
        h_state = h_state.squeeze(0).cpu().numpy()

        return h_state, rwd, pi_probs, value

    def update(self, loss):
        
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def represent(self,x):
        return self.representation_net(x)

    def dynamics(self, h, action):
        x = torch.cat([h,action],dim=-1)
        h = self.dynamic_net(x)
        rwd_prediction = self.rwd_net(h)
        return h, rwd_prediction

    def prediction(self, h):

        pi_logits = self.policy_net(h)

        value_prediction = self.value_net(h)

        return pi_logits, value_prediction
