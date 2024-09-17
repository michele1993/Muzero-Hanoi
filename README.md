# MuZero algorithm for Tower of Hanoi
## Overview
Implementation of the famous [MuZero](https://arxiv.org/abs/1911.08265) algorithm  to solve the Tower of Hanoi problem. The repository includes an implementation of the Tower of Hanoi problem, scalable to an unlimited number of disks, on which MuZero can be trained. The implemnation of the Tower of Hanoi can be found in [hanoy.py](env/hanoy.py) . Additionally, I include an implementation of an algorithm to solve the Tower of Hanoi optimally from any starting position. This 'optimal' algorithm serves as a baseline to assess MuZero performances and can be found in [hanoi_utils.py](env/hanoi_utils.py) . Finally, the repository includes a 'core' implementation of the MuZero algorithm, which attempts to simplify MuZero to its core components, taking away some of the additional elements needed to solve more compex tasks.

## MuZero

## Tower of Haoi
Let's brifly look at how the Tower of Hanoi problem is implemented. The first two key components we need to consider is how the state space and the action space are represented. The Tower of Hanoi consists of 3 pegs with `n` number of disk, the bigger `n` the harder the task. Therefore, we can represent each state as a tuple with `n` entries. In this tuple, each entry represents a different disk with the corresponding value of the entry encoding the peg on which the corresponding disk currently resides. As a result, each entry in the tuple can only take one of three possible values, $`\{0,1,2\}`$, representing one of the 3 pegs.  For simplicity, we assume the smallest disk is represented by the first entry in the tuple (i.e., the entry at index 0) and so on for larger disks. For instance, if we are in a state, $`\{0,0,2,2\}`$, it means the two smallest disks are on the first (leftmost) peg, while the two largest disk are on the last (rightmost) peg, with no disk lying on the mid peg. Note, the state space does not explicitly encode the order of the disks on any given peg (i.e., if multiple disks are on one peg). That is because the rules of the Tower of Hanoi implies the smaller disks must always be on top of the larger disks, therefore the order is always 'given' (i.e., the smaller disks are always assumed to be on top). In practice, all we have to do is to ensure moves which would bring a larger disk on top of a smaller one are not allowed.

This brings us to the structure of the action space. In priciple, there are six possible moves from any state in the Tower of Hanoi problem (i.e., including illegal moves). This is because there are always 3 pegs and, in priciple, we can move a disk from any of the 3 pegs to any of 2 other pegs (i.e., $ 3 x 2 = 3!$ moves). Therefore, the action space can be represented by a simple 1d interger taking values from 0 to 5 (i.e., indexing one of the 6 possible moves). Next, we need to check whether the selected move is allowed from the currest state and if it is not, we should remain in the current state at the next time step (e.g., the move involved a peg with no disks or it led to a bigger disk being placed on top of a smaller one). Here making a choice about the reward functions is vital. Specifically, I decided to provide a negative reward whenever an illegal move is taken, so that the agent should learn which moves are allowed in any given state. The risk with this is that the agent just learns to avoid illegal moves, while never learning the task. Therefore, it is important to ensure the reward for completing the task successufully is much larger than the punishment for taking an illegal move across each step. I do not provide any positive rewards at intermediatery step. Finally, since a positive reward is only provided at successful termination, any RL agent using discounting less than 1, should be encouraged to solve the task with as fewer moves as possible.   



the implementation does not allow you to move are taken where a larger                 


Next, a brief overview on MuZero...

Some final results, we ablate or perturbe different MuZero components to unpack the resulting planning deficits hoping to draw broad comparisons with how different neurological disorders, such as Parkinson's and cerebellar impairments affect planning performances.   

<img src="https://github.com/michele1993/Muzero-Cerebellum/blob/master/img/TOH_MuZero.png" alt="Tower of Hanoi (left) Muzero (right) planning diagrams" width="60%" height="60%">

## Run
Simply run:

```python
python main.py
```
