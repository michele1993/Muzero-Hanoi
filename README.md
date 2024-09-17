# MuZero algorithm for Tower of Hanoi
## Introduction
Implementation of the famous [MuZero](https://arxiv.org/abs/1911.08265) algorithm  to solve the Tower of Hanoi problem. The repository includes an implementation of the Tower of Hanoi problem, scalable to an unlimited number of disks, on which MuZero can be trained. The implemnation of the Tower of Hanoi can be found in the [env](env) folder. Additionally, the [env](env) folder includes an implementation of the algorithm to solve the Tower of Hanoi optimally from any starting position. This 'optimal' algorithm serves as a baseline to assess MuZero performances and can be found in [hanoi_utils](env/hanoi_utils.py). Next, the repository includes a 'core' implementation of the MuZero algorithm, which attempts to simplify MuZero to its core components, taking away some of the additional elements needed to solve more compex tasks.

First, let's brifly look at how the Tower of Hanoi problem is implemented. The first two key components we need to consider is how we are going to represent states (i.e., the structure of the state space) and actions (i.e., the structure of the state space). The Tower of Hanoi consists of 3 pegs with n\` number of disk, the bigger n the harder the task.     


Next, a brief overview on MuZero...

Some final results, we ablate or perturbe different MuZero components to unpack the resulting planning deficits hoping to draw broad comparisons with how different neurological disorders, such as Parkinson's and cerebellar impairments affect planning performances.   

<img src="https://github.com/michele1993/Muzero-Cerebellum/blob/master/img/TOH_MuZero.png" alt="Tower of Hanoi (left) Muzero (right) planning diagrams" width="60%" height="60%">

## Run
Simply run:

```python
python main.py
```
