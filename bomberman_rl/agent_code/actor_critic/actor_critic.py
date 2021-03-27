import argparse
from collections import namedtuple
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from vit_pytorch import ViT

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])


class ActorCritic(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(
        self,
        num_actions=1,
        gamma=0.99,
        eps=1e-7,
        sam_mult=10000,
        **kwargs,
    ):
        super(ActorCritic, self).__init__()

        self.samuels_multiplier = sam_mult

        # action & reward buffers
        self.saved_actions = []
        self.episode_reward = 0
        self.episode_rewards = []
        self.gamma = gamma
        self.eps = eps

    def encode(self, x):
        try:
            return self._encoder(x)
        except AttributeError:
            raise NotImplementedError(f"ActorCritic must implement _encoder")

    def actor(self, x):
        try:
            return self._action_head(x)
        except AttributeError:
            raise NotImplementedError(f"ActorCritic must implement _action_head")

    def critic(self, x):
        try:
            return self._value_head(x)
        except AttributeError:
            raise NotImplementedError(f"ActorCritic must implement _value_head")

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = self.encode(x)

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = self.actor(x)

        # assert False, f"Action prob: {action_prob}"
        # critic: evaluates being in the state s_t
        state_values = self.critic(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device=self._dev)
        probs, state_value = self(state)

        # create a categorical distribution over the list of probabilities of actions

        # probs[:] = 0.25
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value[0]))

        # the action to take (left or right)
        return action.item()

    def reward(self, r):
        # R = r*self.samuels_multiplier
        mult = 10
        R = r * mult

        self.episode_rewards.append(R)
        self.episode_reward += R

    def update(self, optimizer):
        """
        Call at end of one episode

        Training code.
        Calculates actor and critic loss and performs backprop.

        For meaning of "advantage", "reward", "value",
        see "Back to Baselines" section of
        https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
        """

        R = 0
        saved_actions = self.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        # calculate the true value using episode_rewards returned from the environment
        for r in self.episode_rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

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
        self.episode_reward = 0
        del self.episode_rewards[:]
        del self.saved_actions[:]


class ActorCriticLinear(ActorCritic):
    """
    implements both actor and critic in one model, linearly
    """

    def __init__(
        self,
        num_states,
        num_actions=1,
        state_dim=64,
        actor_hiddens=[128, 32],
        critic_hiddens=[128, 32, 4],
        dropout=0.2,
        gamma=0.99,
        **kwargs,
    ):

        super(ActorCriticLinear, self).__init__(
            num_states,
            num_actions,
            gamma=gamma,
            **kwargs,
        )

        self.state_dim = (
            state_dim  # hidden state dimension which is input to A and to C
        )

        # --------------- ENCODER --------------

        self._encoder = nn.Sequential(nn.Linear(num_states, state_dim), nn.ReLU6())

        # --------------- ACTOR -----------------

        actor_layers = []
        actor_prev_hidden = state_dim

        for actor_hidden in actor_hiddens:
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

        self._action_head = nn.Sequential(*actor_layers)

        # --------------- CRITIC -----------------

        critic_layers = []
        critic_prev_hidden = state_dim

        for critic_hidden in critic_hiddens:
            critic_layers += [
                nn.Linear(critic_prev_hidden, critic_hidden),
                nn.ReLU6(),
            ]
            critic_prev_hidden = critic_hidden

        critic_layers += [
            nn.Linear(critic_prev_hidden, 1),
        ]

        self._value_head = nn.Sequential(*critic_layers)


class ActorCriticConv(ActorCritic):
    """
    implements both actor and critic in one model, conventionally
    """

    def __init__(
        self,
        board_size,  # e.g. 17
        num_actions,
        in_channels,
        actor_channels=[34, 68, 34, 17, 9],
        flattened_dim_actor=75, # TODO determine by running model,
        critic_channels=[34, 68, 34, 17, 9],
        flattened_dim_critic=75, # TODO determine by running model
        dropout=0.1,
        gamma=0.99,
        **kwargs,
    ):

        super(ActorCriticConv, self).__init__()

        state_dim = in_channels  # hidden channels which are input to A and to C

        # --------------- ENCODER --------------

        self._encoder = nn.Sequential(
            nn.Conv2d(in_channels, state_dim, kernel_size=1), nn.ReLU6()
        )

        # --------------- ACTOR -----------------

        actor_layers = []
        actor_prev_channel = state_dim

        for actor_channel in actor_channels:
            actor_layers += [
                nn.Conv2d(actor_prev_channel, actor_channel, 5),
                nn.ReLU6(),
            ]
            actor_prev_channel = actor_channel

        actor_layers += [
            nn.Flatten(1),
            nn.Linear(flattened_dim_actor, num_actions),
            nn.Softmax(dim=-1),
        ]

        self._action_head = nn.Sequential(*actor_layers)

        # --------------- CRITIC -----------------

        # TODO rename to hiddens
        critic_layers = []
        critic_prev_channel = state_dim

        for critic_channel in critic_channels:
            critic_layers += [
                nn.Conv2d(critic_prev_channel, critic_channel, 5),
                nn.ReLU6(),
            ]
            critic_prev_channel = critic_channel

        critic_layers += [
            nn.Flatten(1),
            nn.Linear(flattened_dim_critic, 25),
            nn.ReLU6(),
            nn.Linear(25,1)
        ]

        self._value_head = nn.Sequential(*critic_layers)


class ActorCriticTransformer(ActorCritic):
    """
    implements both actor and critic in one model, conventionally
    """

    def __init__(
        self,
        board_size,  # e.g. 17
        num_states,
        num_actions,
        hidden_dim,
        num_heads,
        mlp_dim,
        num_layers=2,
        critic_hiddens=[9],
        dropout=0.08,
        gamma=0.99,
        **kwargs,
    ):

        super(ActorCriticTransformer, self).__init__(
            num_actions=num_actions,
            gamma=gamma,
            **kwargs
        )

        # --------------- ENCODER --------------

        self._encoder = nn.Identity()

        # --------------- ACTOR -----------------

        actor = ViT(
            image_size=board_size,
            channels=num_states,
            patch_size=1,
            num_classes=num_actions,
            dim=hidden_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=dropout,
        )
        self._action_head = nn.Sequential(
            actor, 
            nn.Softmax(dim=-1)
        )

        # --------------- CRITIC -----------------

        critic_layers = [
            nn.Flatten(2),
        ]
        critic_prev_hidden = board_size ** 2

        for critic_hidden in critic_hiddens:
            critic_layers += [
                nn.Linear(critic_prev_hidden, critic_hidden),
                nn.ReLU6(),
                nn.Dropout(p=dropout),
            ]
            critic_prev_hidden = critic_hidden

        critic_layers += [
            nn.Flatten(1),
            nn.Linear(critic_prev_hidden * num_states, 1),
        ]

        self._value_head = nn.Sequential(*critic_layers)
