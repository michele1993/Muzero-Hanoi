import numpy as np

def compute_MCreturns(rwds,discount):
    """ Compute MC return based on a list of rwds
    Args:
        rwds: list of rwds for a given episode
        discount: discount factor
    """        
    
    rwds = np.array(rwds)
    discuounts = (discount**(np.array(range(len(rwds)))))

    return np.flip(np.cumsum(np.flip(discounts * rwds, axis=(0,)), axis=0), axis=(0,)) / discounts
     
