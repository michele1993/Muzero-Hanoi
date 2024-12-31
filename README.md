# MuZero algorithm for Tower of Hanoi
## Overview
Implementation of the famous [MuZero](https://arxiv.org/abs/1911.08265) algorithm  to solve the Tower of Hanoi problem. This repository aims to provide a 'playground' to investigate the relation between MuZero and human cognitive planning in healthy and abnormal conditions. 

The repository includes an implementation of the Tower of Hanoi problem, scalable to an unlimited number of disks, on which MuZero can be trained. The implemnation of the Tower of Hanoi can be found in `hanoy.py`. Additionally, I include an implementation of an algorithm to solve the Tower of Hanoi optimally from any starting position. This 'optimal' algorithm serves as a baseline to assess MuZero performances and can be found in `hanoi_utils.py`. Finally, the repository includes a 'core' implementation of the MuZero algorithm, which attempts to simplify MuZero to its core components, taking away some of the additional elements needed to solve more compex tasks.

## MuZero
MuZero is best described through its 3 distinct phases: Planning, Action Selection and Training.
### Planning
<img src="https://github.com/michele1993/Muzero-Cerebellum/blob/master/img/Latent_planning.png" alt="Figure: Planning process in MuZero" width="30%" height="30%">

The planning process of MuZero relies on three key components: an encoder, $h$, that maps observations $[o_1, \dots, o_t]$ to a latent space, $s^0$; a MLP, $f$, mapping latent representations onto a policy as well as a value function; and finally, a recurrent network $g$, which evolves the latent (planning) dynamics and predicts one-step rewards, starting from $s^0$. For any real-time step $t$, we have:

$$s^0 = h_\theta(o_1, \dots, o_t)$$

$$p^k,v^k = f_\theta(s^k) $$

$$r^k,s^k = g_\theta(s^{k-1},a^k)$$


Based on these three components, MuZero performs a Monte Carlo Tree Search (MCTS) in latent space to select the action to perform at each step $t$ of the (real) environment. The search always starts at the (root) node associated to state $s^0$ and unfolds through the latent dynamics provided by $g_\theta()$, with actions $a^k$ driving these dynamics. These actions refer to "latent" actions and are not equal to the actions taken in the environment, which originate from the MCTS.  These latent actions reflect a cobination of the latent policy and the estimated value for each node/latent-state visited in the MCTS. Specifically, the latent actions (driving the MCTS) are selected based on the following formula:

$$a^k = argmax \left[Q(s,a) + w P(s,a) \right]$$

where $Q()$ is the estimated value for the state-action, $(s,a)$, $P()$ is the policy provided by $f_\theta$  and, finally, $w$ represents the UCB bonus, which encourages taking actions that haven't been taken before in the MCTS.

The state-action value, $Q()$, is computed as the sum between the (estimated) one step reward, $r^{k+1}$, and the discounted value of the child node, $s^{k+1}$, resulting from taking action, $a$ in the node/latent-state $s^k$,

$$ Q(s^k,a^k) = r^{k+1} + V^{\text{node}}(s^{k+1}) $$

Here, it is important to stress $V^{\text{node}}$ is not the same as the model predictions $v$. Specifically, the value predictions $v$ are only used during the back-up phase, providing an estimate of the value of the leaf node reached at end of each MCTS pass (i.e., this is different from standard MCTS algorithms without values functions, which simply use the sum of rewards up to that point as a value to back-up). During the back-up phase, the model estimated values $v$ are used to update the parent nodes' values, $V^{\text{node}}$, together with the estimated one step rewards. The actual back-up update is similar to a TD(n) update, where n reflects how many step away the updated node is from the leaf node with estimated value $v$. The resulting value of each node, $V^{\text{node}}$, ends up reflecting an averaged over TD(n) targets for different n.      
Finally, in the MuZero paper, the UCB bonus, $w$, is computed as following,

$$     w = \frac{\sqrt{\sum_{b} N(s,b)}}{1+N(s,a)} \left(c_1 + \log\frac{\sum_{b} N(s,b) + c_2 +1}{c_2}\right) $$

where $N$ encodes the number of times a state-action pair, $(s,a)$, has been visited within the (current) MCTS, while  $c_1$ and $c_2$ represent fixed hyper-paramters.    
Note: On the first pass of each MCTS, $w=0$ because $\sum_b N(s,b)=0$. Additionally, all $Q(s,\cdot)$ are initialised to zero in the child nodes. Hence, the expanded action (i.e., best child from the root) is random on the first pass of MCTS, even for a trained model.

In summary, both $v$ and $p$ influence the latent planning dynamics through distinct mechanisms. As we will see below, $p$ is trained to reproduce the action actually taken in the environment (i.e., the result of the MCTS for each step), while $v$ is train to reflect the returns.


### Action Selection
<img src="https://github.com/michele1993/Muzero-Cerebellum/blob/master/img/ActionSelect.png" alt="Figure: Action selection process in MuZero" width="50%" height="50%">

In the environment, actions are chosen based on the (MCTS) planning process described above. Specifically, MuZero creates a histogram of how many times each action at the initial (root) node is taken during the planning process (for a fixed number of MCTS passes). Based on this histogram (i.e., the discrete probability for each action), an action is sampled to be performed in the environment. Note that the more often an action is selected in the MCTS, the more likely the action is to be a good one. This process is repeated at each time step in the environment. The MCTS is always started at the latent representation of a real (observed) state (computed by the encoder, $h$).

### Training
<img src="https://github.com/michele1993/Muzero-Cerebellum/blob/master/img/Training.png" alt="Figure: Training process in MuZero" width="50%" height="50%">

All MuZero components are trained jointly by unfolding the recurrent dynamics a second time based on the real state and the real actions in a separate training step. By "real", I refer to states and actions actually observed and performed in the environment, in contrast to those of the latent planning process. A real state, $s$, and the subsequent $n$ number of real actions, $a$, are sampled from a buffer (where $n$ is a hyper-parameter). The real state is encoded in a latent representation by $h$ and then used as the initial (latent) state to unfold the recurrent latent dynamics of $g$ for $n$ steps based on the real $n$ actions.

For each step, $g()$ and $f()$ are provided with targets based on the real (observed) rewards, final returns (or TD target can also be used), and action distributions (i.e., based on the action histogram created by the MCTS at each step in the environment). By backpropagating through the $n$ steps given the corresponding targets, each component of MuZero is jointly trained for each (real) time step $t$:

$$     l_t(\theta) = \sum_{n=0}^{N} l^\text{r}(u_{t+n},r_t^n) + l^\text{v}(z_{t+n},v_t^n) + l^\text{p}(\pi_{t+n},p_t^n) + c || \theta ||^2 $$

For each real state at time $t$ sampled from a buffer, we unfold the latent (recurrent) dynamics for $n$ steps and use the real targets given by $t+n$ (i.e., the targets observed after visiting the state at time $t$) to train the MuZero components in one go. This is repeated for each sampled state.



## Tower of Hanoi
<img src="https://github.com/michele1993/Muzero-Cerebellum/blob/master/img/TOH_only.png" alt="Tower of Hanoi (left) Muzero (right) planning diagrams" width="50%" height="50%">

Let's brifly look at how the Tower of Hanoi problem is implemented. The first two key components we need to consider is how the state space and the action space are represented. The Tower of Hanoi consists of 3 pegs with $n$ number of disk, the bigger $n$ the harder the task. Therefore, we can represent each state as a tuple with $n$ entries. In this tuple, each entry represents a different disk with the corresponding value of the entry encoding the peg on which the corresponding disk currently resides. As a result, each entry in the tuple can only take one of three possible values, $`\{0,1,2\}`$, representing one of the 3 pegs.  For simplicity, we assume the smallest disk is represented by the first entry in the tuple (i.e., the entry at index 0) and so on for larger disks. For instance, if we are in a state, $`\{0,0,2,2\}`$, it means the two smallest disks are on the first (leftmost) peg, while the two largest disk are on the last (rightmost) peg, with no disk lying on the mid peg. Note, the state space does not explicitly encode the order of the disks on any given peg (i.e., if multiple disks are on one peg). That is because the rules of the Tower of Hanoi implies the smaller disks must always be on top of the larger disks, therefore the order is always 'given' (i.e., the smaller disks are always assumed to be on top). In practice, all we have to do is to ensure moves which would bring a larger disk on top of a smaller one are not allowed.

This brings us to the structure of the action space. In priciple, there are six possible moves from any state in the Tower of Hanoi problem (i.e., including illegal moves). This is because there are always 3 pegs and, in priciple, we can move a disk from any of the 3 pegs to any of 2 other pegs (i.e., $ 3 x 2 = 3!$ moves). Therefore, the action space can be represented by a simple 1d interger taking values from 0 to 5 (i.e., indexing one of the 6 possible moves). Next, we need to check whether the selected move is allowed from the currest state and if it is not, we should remain in the current state at the next time step (e.g., the move involved a peg with no disks or it led to a bigger disk being placed on top of a smaller one). Here making a choice about the reward functions is vital. Specifically, I decided to provide a negative reward whenever an illegal move is taken, so that the agent should learn which moves are allowed in any given state. The risk with this is that the agent just learns to avoid illegal moves, while never learning the task. Therefore, it is important to ensure the reward for completing the task successufully is much larger than the punishment for taking an illegal move across each step. I do not provide any positive rewards at intermediatery step. Finally, since a positive reward is only provided at successful termination, any RL agent using discounting less than 1, should be encouraged to solve the task with as fewer moves as possible.   


## Run
Simply run:

```python
python main.py
```
