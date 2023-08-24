import sys
sys.path.append('..')
import numpy as np
import itertools
from utils import oneHot_encoding

class TowersOfHanoi:
    ''' Implementation of Towers of Hanoi game for RL algorithms'''
    def __init__(self, N, init_state_idx=0):
        self.discs = N # n. of disks to be used
        self.n_pegs = 3 # NOTE: in tower of Hanoi the number of peg is always 3 (only the n. of disks can increase)

        # ===== Generate the state space =======
        #NOTE: state representation is a tuple where the position denotes a different disk while the value denotes the peg, so
        # a 5 disk tower has a 5 dim state space (each dim for a different disk with the value denoting the peg 
        # since there are 3 pegs, possible values can be {0,1,2}).
        self.states = list(itertools.product(list(range(self.n_pegs)), repeat=self.discs)) # Cartesian product of [0,1,2], N elements at the time 
        self.oneH_s_size = self.discs * self.n_pegs
        ## =====================================

        goal_peg = 2 # intialise indx for the goal peg, 2 = right-most peg
        self.goal = tuple([goal_peg]*self.discs) # initialise goal in tuple representation
        self.init_state_idx = init_state_idx # initialise indx for starting position

        ## Moves is a tuple where the first entry denotes the peg we are going from and the second entry denotes the peg we are going to 
        ## e.g. [0,1] is moving a disk from the first peg (0) to the second peg (1) 
        ## here we are denoting all possible moves, checking later whether the move is allowd
        self.moves = list(itertools.permutations(list(range(self.n_pegs)), 2)) # No matter the number of disks, there are always 6 (legal/illegal) moves 

    def step(self,action):
        ## Perform a step in tower of Hanoi, by passing an action indexing one of the 6 available moves
        ## If an illegal move is chosen, give rwd=-1 and remain in current state
        move = self.moves[action]
        illegal_move =  not self.move_allowed(move) # return False if move is allowed

        if not illegal_move:
            moved_state = self.get_moved_state(move) 
            if moved_state != self.goal: ## return rwd=0 if the goal has not been reached (with a legal move)
                rwd = 0
                done = False
                self.c_state = moved_state
            else: ## return rwd=100 if goal has been reached
                rwd = 100 
                done = True
        else:
            ## if selected illegal move, don't terminate state but state in the same state and rwd=-1
            rwd = -100/200 # 0
            moved_state = self.c_state 
            done = False

        # Compute one-hot representation for new_state to be returned
        oneH_moved_state = oneHot_encoding(moved_state,n_integers=self.n_pegs)
        return oneH_moved_state, rwd, done, illegal_move

    def reset(self):
        ## NOTE: at the moment reset always from same state based on init_s_idx, but later can randomise this
        ## by randomising self.init_state_idx
        self.c_state = self.states[self.init_state_idx] # reset to some initial state, e.g., first state (0,0,0,...), all disks on first peg
        self.oneH_c_state = oneHot_encoding(self.c_state, n_integers=self.n_pegs)
        return self.oneH_c_state, False # also return done=False

    def discs_on_peg(self, peg):
        ## Allows to create a list contatining all the disks that are on that specific peg at the moment (i.e. self.state)
        return [disc for disc in range(self.discs) if self.c_state[disc] == peg] # add to list only disks that are on that specific peg

    def move_allowed(self, move):
        discs_from = self.discs_on_peg(move[0]) # Check what disks are on the peg we want to move FROM 
        discs_to = self.discs_on_peg(move[1]) # Check what disks are on the peg we want to move TO

        if discs_from: # Check the list is not empty (i.e. there is at list a disk on the peg we want to move from)
            ## NOTE: Here needs the extra if ... else ... because if disc_to is empty, min() returns an error
            return (min(discs_to) > min(discs_from)) if discs_to else True # return True if we are allowed to make the move (i.e. disk from is smaller than disk to)
        else:
            return False # else return False, the move is not allowed

    def get_moved_state(self, move):
        if self.move_allowed(move):
            disc_to_move = min(self.discs_on_peg(move[0])) # select smallest disk among disks on peg we want to move FROM
        moved_state = list(self.c_state) # take current state
        ## NOTE: since each state dim is a disk (not a peg) then a move only changes that one dim of the state referring to the moved disk
        moved_state[disc_to_move] = move[1] # update current state by simply chaging the value of the (one) disk that got moved
        return tuple(moved_state) # Return new (moved) state
