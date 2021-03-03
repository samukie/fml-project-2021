Final project for the course Fudamentals of Machine Learning 2020/2021.  
Code template: https://github.com/ukoethe/bomberman_rl  

<p align="left">
  <img src="alphabomber.png"  width="200"/> <h4>Our Mission:</h4> We are developing cutting-edge Reinforcement Learning techniques in order to solve the ancient game of Bomberman. 
</p> 

## Reinforcement Learning - A quick overview: 

Common RL methods can be separated into **policy** based and **value** based.  
A value method outputs a value representing the quality of the current state and learns through the process of value iteration. Starting by choosing a random value function, this process iteratively improves the function, until arriving at an optimal value functing. The optimal policy can then be derived from that function. 
  
On the other side, policy based methods aim at directly improving the policy of an agent, i.e., a mapping between states and actions. 

Popular methods are:  
- (Deep) Q-Learning (value method)
- Policy Gradient (policy method)
- A2C (policy and value)
- A3C (policy and value)
- Dreamer v2 (OP method)
