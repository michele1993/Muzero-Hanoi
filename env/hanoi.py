''' Reinforcement learning of the Towers of Hanoi game.
Reference: Watkins and Dayan, "Q-Learning", Machine Learning, 8, 279-292 (1992).'''
import numpy as np
import itertools

#NOTE: state representations where the position denotes a different disk while the value denotes the peg, so
# a 5 disk tower has a 5 dim state space.
class TowersOfHanoi:

    def __init__(self, N, init_state_idx=0):
        self.discs = N
        self.states = list(itertools.product(list(range(3)), repeat=self.discs)) # Cartesian product of [0,1,2], N elements at the time 
        self.goal = tuple([2]*self.discs)
        self.init_state_idx = init_state_idx
        ## Moves is a tuple where the first entry denotes the peg we are going from and the second entry denotes the peg we are going to 
        ## e.g. [0,1] is moving a disk from the first peg (0) to the second peg (1) 
        ## here we are denoting all possible moves, checking later whether the move is allowd
        self.moves = list(itertools.permutations(list(range(3)), 2)) # No matter the number of disks, there are always 6 moves overall

    def step(self,action):
        ## Perform a step in tower of Hanoi, by passing an action indexing one of the 6 available moves
        ## If an illegal move is chosen, no longer return done=True (i.e. terminate episode) and rwd= 0
        ## NOTE: don't need to access R matrix, just need to check if move is allowed and whether the goal has been reached
        ## to determine the rwd

        move = self.moves[action]
        illegal_move =  not self.move_allowed(move) # return False if move is allowed

        if not illegal_move:
            moved_state = self.get_moved_state(move) 
            ## If the goal has not been reached (with a legal move) return r=0
            if moved_state != self.goal:
                rwd = 0#0.001 # give tiny bonus to encourage legal actions, not good idea for n=5
                done = False
                self.c_state = moved_state
            else: ## return r=100 if goal has been reached
                rwd = 100 #self.R[self.c_state, moved_state]
                done = True

        else:

            ## if selected illegal move, terminate the episode and rwd= -1000
            #rwd = - 1000#np.inf
            #moved_state= None
            #done = True

            ## if selected illegal move, don't terminate state but state in the same state and rwd=0
            rwd = -1 # 0
            moved_state = self.c_state 
            done = False

        return moved_state, rwd, done, illegal_move

    def reset(self):
        ## NOTE: at the moment reset always from same state based on init_s_idx, but later can randomise this
        self.c_state = self.states[self.init_state_idx] # reset to first state (0,0,0,...) - i.e. all disks on first peg
        return self.c_state, False # also return done=False

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
