# 0. Notes for Narrative/Report

+ == geschrieben  
- == TODO

notes für abgabe:
* TODO plotting funktion ???
*
*


## Structure

# 2. Introduction

# 3. Theory

# 4. Training


### Marvin: A/C D/C

Idea:  

+ fix algo (AC)
+ give entire gamestate to agent 
+ find deep architecture that can make use of anything thrown @ it

#### Learning Algorithm

- actor critic
- refer to theory
- base code from pytorch examples

#### Architectures

- architectures: conv, depthwise, 2d, pos info, optim==AdamW
- small vs big architectures: is big even good? especially hard to judge how trainable it is in given amount of time because progress is just so slow

#### Features

- basic idea: entire game state; input like for conv arch
- my feature engineering idea: conv input like, with scalars indicating rewarding places => depthwise conv
- random transforms & gedaechtnislosigkeit
- board boundary as maneuvering space (cite ukoethe)

#### Rewards

- rules vs rewards, inspiration from rule\_based\_agent \& breadth first search with multiple targets

#### Tasks

-  ?????

#### Problems \& Solutions

- exploitation vs exploration:
 alpha (-> lecture) => difficult to evaluate
- suicide: disallow bomb/learned fast
=> experiment for this???
- what did my agents (until saturday night) learn? to avoid placing a bomb altogether they all learned really fast, but the rest of the (not very lengthy) training seemed very random => they probably couldnt really make sense of the inputs => the final layer just learned not to output one class but it is likely the conv layers didnt really learn anything
=> experiment for this???
- reward shaping is finnicky: it looked like punishing invalid actions and waiting highly led to very low reward sums, I hypothesized this could be a problem in training and didnt punish invalid actions and waiting (reward = 0) for the big agents, which didnt do anything anymore
=> experiment for this???

# 5. Experiments

- ??? samuels bomber \& silas' plots
- performance vgl in task 3: VERSUS


# 6. Takeaways

+ Describe team workflow: Separately work on tasks and exchange ideas and agents
+ Different feature engineering AND different agents
+ makes it a bit hard to quickly exchange just the features or just the agent, if the agents depend on vary different input arrays, could be a bit alleviated by common use of Agent and Feature and Reward interfaces
+ I concentrated on architecture => made it hard to exploit the curriculum learning aspect that was so great in this project as I needed to discard learned weights because I continually changed my agents's code, would probably have been better to stay with one catch-all arch and then feature engineer/reward shape around it and tune hyperparams. 
+ self play => write more??
+ detriments of no curriculum learning (because of architectural focus)


