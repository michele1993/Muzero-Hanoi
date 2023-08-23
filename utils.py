import numpy as np
import logging
import torch

def compute_MCreturns(rwds,discount):
    """ Compute MC return based on a list of rwds
    Args:
        rwds: list of rwds for a given episode
        discount: discount factor
    """        
    rwds = np.array(rwds)
    discounts = (discount**(np.array(range(len(rwds)))))
    return list(np.flip(np.cumsum(np.flip(discounts * rwds, axis=(0,)), axis=0), axis=(0,)) / discounts)
     
def adjust_temperature(env_step):
    """ Adjust temperature based on which step you're in the env - higher temp for early steps"""
    if env_step < 6:
        return 1.0
    # else:
    #     return 0.0  # Play according to the max.
    return 0.1

def setup_logger(seed):
    """ set useful logger set-up"""
    logging.basicConfig(format='%(asctime)s %(message)s', encoding='utf-8', level=logging.INFO)
    logging.debug(f'Pytorch version: {torch.__version__}')
    if seed is not None:
        logging.info(f'Seed: {seed}')


