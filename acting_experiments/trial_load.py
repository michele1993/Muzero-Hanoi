import torch
import sys
sys.path.append('/Users/px19783/code_repository/cerebellum_project/latent_planning')
from networks import MuZeroNet

# Load
model_dict = torch.load('/Users/px19783/code_repository/cerebellum_project/latent_planning/results/1/muzero_model.pt')
networks = MuZeroNet(rpr_input_s= 9, action_s = 6, lr=0.003, TD_return=True) # based on trial example with N =3
networks.load_state_dict(model_dict['Muzero_net'])
networks.optimiser.load_state_dict(model_dict['Net_optim'])
