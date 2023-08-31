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

def compute_n_step_returns(rwds, root_values, n_step, discount):
    """Compute n-step TD return.
    Args:
        rwds: a list of rewards received from the env, length T.
        root_values: a list of root node value from MCTS search, length T.
        td_steps: the number of steps into the future for n-step value.
        discount: discount for future reward.

    Returns:
        a list of n-step target value, length T.

    Raises:
        ValueError:
            lists `rewards` and `root_values` do not have equal length.
    """

    assert n_step > 0, 'the n_step return must be greater than zero'

    assert len(rwds) == len(root_values), '`rewards` and `root_values` don have the same length.'

    T = len(rwds)

    # Make a shallow copy to avoid manipulate on the original data.
    _rwds = list(rwds)
    _root_values = list(root_values)

    # Add padding for the end of the trajectory
    _rwds += [0] * n_step
    _root_values += [0] * n_step

    td_returns = []
    for t in range(T):
        bootstrap_idx = t + n_step
        # Compute first component of n_step TD targets as n_step discounted sum of rwds
        dis_rwd_sum = sum([discount**i * r for i,r in enumerate(_rwds[t:bootstrap_idx])])

        # Add the bootstrapped value based on MCTS to the rwd sum
        value = dis_rwd_sum + discount**n_step * _root_values[bootstrap_idx]

        td_returns.append(value)
    return td_returns

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
    if env_step < 500:
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


