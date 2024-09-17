# MuZero algorithm for Tower of Hanoi
## Introduction
Implementation of the famous [MuZero](https://arxiv.org/abs/1911.08265) algorithm  to solve the Tower of Hanoi problem. The repository includes an implementation of the Tower of Hanoi problem, scalable to an unlimited number of disks, on which MuZero can be trained. The implemnation of the Tower of Hanoi can be found in the [env](env) folder. Additionally, the [env](env) folder includes an implementation of the recursive algorithm to solve the Tower of Hanoi optimally from any starting position. This 'optimal' algorithm serves as a baseline to assess MuZero performances and can be found in [hanoi_utils](env/hanoi_utils.py). Next, the repository includes a 'core' implementation of the MuZero algorithm, which attempts to simplify MuZero to its core components, taking away some of the additional elements needed to solve more compex tasks.

First, let's brifly look at how the Tower of Hanoi problem is implemented. The first two key components we need to consider is how we are going to represent states (i.e., the structure of the state space) and actions (i.e., the structure of the action space). The Tower of Hanoi consists of 3 pegs with `n` number of disk, the bigger `n` the harder the task. Therefore, we can represent each state as a tuple with `n` entries. In this tuple, each entry represents a different disk with the corresponding value of the entry encoding the peg on which the corresponding disk currently resides. As a result, each entry in the tuple can only take one of three possible values, $`\{0,1,2\}`$, representing one of the 3 pegs.  For simplicity, we assume the smallest disk is represented by the first entry in the tuple (i.e., the entry at index 0) and so on for larger disks. For instance, if we are in a state, $`\{0,0,2,2\}`$, it means the two smallest disks are on the first (leftmost) peg, while the two largest disk are on the last (rightmost) peg, with no disk lying on the mid peg. Note, the state space does not explicitly encode the order of the disks on any given peg (i.e., if multiple disks are on one peg). That is because the rules of the Tower of Hanoi implies the smaller disks must always be on top of the larger disks, therefore the order is always 'given' (i.e., the smaller disks are always assumed to be on top). In practice, all we have to do is to ensure moves which would bring a larger disk on top of a smaller one are not allowed.

This brings us to the structure of the action space.



the implementation does not allow you to move are taken where a larger                 


Next, a brief overview on MuZero...

Some final results, we ablate or perturbe different MuZero components to unpack the resulting planning deficits hoping to draw broad comparisons with how different neurological disorders, such as Parkinson's and cerebellar impairments affect planning performances.   

<img src="https://github.com/michele1993/Muzero-Cerebellum/blob/master/img/TOH_MuZero.png" alt="Tower of Hanoi (left) Muzero (right) planning diagrams" width="60%" height="60%">

## Run
Simply run:

```python
python main.py
```
