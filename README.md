# MuZero algorithm for Tower of Hanoi
## Introduction
Implementation of the famous [MuZero](https://arxiv.org/abs/1911.08265) algorithm  to solve the Tower of Hanoi problem. This algorithm aims to provide a baseline algorithm to compare it to human planning performance on the same task. To do so, I simplified the MuZero algorithm to its core components, taking away some of the additional elements needed to solve more compex tasks.  
Next, we ablate or perturbe different MuZero components to unpack the resulting planning deficits hoping to draw broad comparisons with how different neurological disorders, such as Parkinson's and cerebellar impairments affect planning performances.   

<img src="https://github.com/michele1993/Muzero-Cerebellum/blob/master/img/TOH_MuZero.png" alt="Tower of Hanoi (left) Muzero (right) planning diagrams" width="60%" height="60%">

## Run
Simply run:

```python
python main.py
```
