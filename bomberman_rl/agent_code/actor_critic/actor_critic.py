import argparse
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])




class ActorCritic(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(
        self,
        num_states,
        num_actions, 
        state_dim=64,
        actor_hiddens=[128,32],
        critic_hiddens=[128],
        dropout=.2,
        gamma=.99
        ):

        super(ActorCritic, self).__init__()

        self.state_dim = state_dim # hidden state dimension which is input to A and to C

        self.affine1 = nn.Sequential(
            nn.Linear(num_states, state_dim),
            nn.ReLU6()
        )

        # --------------- ACTOR -----------------
        
        actor_layers = []
        actor_prev_hidden = state_dim

        for actor_hidden in range(actor_hiddens):
            actor_layers += [
                nn.Linear(actor_prev_hidden, actor_hidden),
                nn.Dropout(p=dropout),
                nn.ReLU6(),
            ]
            actor_prev_hidden = actor_hidden

        actor_layers += [
            nn.Linear(actor_prev_hidden, num_actions),
            nn.Softmax(dim=-1)
        ]

        self.action_head = nn.Sequential(actor_layers)

        # --------------- CRITIC -----------------

        critic_layers = []
        critic_prev_hidden = state_dim

        for critic_hidden in range(critic_hiddens):
            critic_layers += [
                nn.Linear(critic_prev_hidden, critic_hidden),
                nn.ReLU6(),
            ]
            critic_prev_hidden = critic_hidden

        critic_layers += [
            nn.Linear(critic_prev_hidden, 1),
        ]

        self.value_head = nn.Sequential(critic_layers)

        # action & reward buffer
        self.saved_actions = []
        self.episode_reward = 0
        self.episode_rewards = []
        self.gamma = gamma

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = self.affine1(x)

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = self.action_head(x)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values


    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        # the action to take (left or right)
        return action.item()

    def update(self, optimizer, eps):
        """
        Call at end of one episode 

        Training code. 
        Calculates actor and critic loss and performs backprop.

        For meaning of "advantage", "reward", "value", 
        see "Back to Baselines" section of
        https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
        """

        R = 0
        saved_actions = model.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using episode_rewards returned from the environment
        for r in model.episode_rewards[::-1]:
            # calculate the discounted value
            R = r + model.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        optimizer.step()

        # reset episode_rewards and action buffer
        model.episode_reward = 0
        del model.episode_rewards[:]
        del model.saved_actions[:]