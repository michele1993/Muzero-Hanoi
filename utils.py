import numpy as np
import logging
import torch

def oneHot_encoding(x, n_integers):
    """ Provide efficient one-hot encoding for integer vector x, by
        using a separate one-hot encoding for each dimension of x and then
        concatenating all the one-hot representations into a single vector 
        Args:
            x: integer vector for which need one-hot representatio
            n_integers: number of possible integer values in the entire x-space (i.e., across all x)
        Returns:
            one-hot vector representation of x
    """
    x_dim = len(x)
    # Create a one-hot vector for each dim(x) of size based on n. of possible integer values across x-space
    oneH_mat = np.zeros((x_dim,n_integers))
    # Fill in the 1 based on the value in each dim of x
    oneH_mat[np.arange(x_dim),x] = 1
    # Return one-hot vector
    return oneH_mat.reshape(-1)

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


