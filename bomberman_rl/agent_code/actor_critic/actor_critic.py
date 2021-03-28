import argparse
from collections import namedtuple
from itertools import count
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from vit_pytorch import ViT

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])

class Residual(nn.Module):
    """Sequential with downsample module"""
    def __init__(self, downsample, *mods, name="actor", num=0, **kwargs):
        super().__init__()
        for idx, mod in enumerate(mods):
            self.add_module(
                name=str(idx),
                module=mod
            )
        self.n = idx+1
        self.downsample = downsample
        self.num = num
        self.name = name

    def forward(self, x):
        residual = x
        for i in range(self.n):
            mod = self._modules[str(i)]
            try:
                x = mod(x)
            except Exception as e:
                input(
f"{self.name}'s Residual #{self.num} got Error at its child #{i}/{len(self._modules)}: {e}"
                )
                raise e
        try:
            ds = self.downsample(residual)
            out = x + ds 
        except Exception as e:
            input(
f"{self.name}'s Residual #{self.num} got Error: {e}"
            )
            input(f"Shapes: (after main strand; then after downsample):\n{x.shape}\n{ds.shape}")
            raise
        return out

class Temperature(nn.Module):
    """
    For use in nn.Sequential, before Softmax:

    logits = logits / alpha

    alpha < 0: peakier (exploit)
    alpha > 0: widerer (explore)

    temperature is dict of model so we can update it in place without having to go down to this model
    """
    def __init__(self, temperature: Dict[str,float]):
        super().__init__()
        self.temperature = temperature
    def forward(self, logits):
        return logits / self.temperature["alpha"]


class ActorCritic(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(
        self,
        num_actions=1,
        gamma=0.99,
        eps=1e-7,
        sam_mult=1,
        **kwargs,
    ):
        super(ActorCritic, self).__init__()

        self.samuels_multiplier = sam_mult

        # action & reward buffers
        self.saved_actions = []
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
        except AttributeError as e:
            input(e)

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
        R = r * self.samuels_multiplier

        self.episode_rewards.append(R)

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

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device=self._dev)))

        # reset gradients
        optimizer.zero_grad()

        # # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # input(loss)

        # perform backprop
        loss.backward()
        optimizer.step()

        # reset episode_rewards and action buffer
        self.episode_rewards = []
        self.saved_actions = []


class ActorCriticLinear(ActorCritic):
    """
    implements both actor and critic in one model, linearly
    """

    def __init__(
        self,
        num_channels,
        num_actions=1,
        state_dim=17,
        actor_hiddens=[17, 9],
        critic_hiddens=[17, 9],
        alpha=1.0,
        dropout=0.0,
        **kwargs,
    ):

        super(ActorCriticLinear, self).__init__(
            num_channels,
            num_actions,
            **kwargs,
        )
        input_dim = 1734

        self.state_dim = state_dim  # hidden state dimension which is input to A and to C

        # --------------- ENCODER --------------

        self._encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(input_dim, state_dim), 
            nn.ReLU6()
        )

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

        self.temperature = {"alpha": alpha}

        actor_layers += [
            nn.Linear(actor_prev_hidden, num_actions), 
            Temperature(self.temperature),
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
        flattened_dim_actor=5, # TODO determine by running model,
        critic_channels=[34, 68, 34, 17, 9],
        flattened_dim_critic=5, # TODO determine by running model
        alpha=1.0,
        dropout=0.0,
        gamma=0.99,
        **kwargs,
    ):

        super().__init__()

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

        self.temperature = {"alpha": alpha}

        actor_layers += [
            nn.Flatten(1),
            nn.Linear(flattened_dim_actor, num_actions),
            Temperature(self.temperature),
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



class ActorCriticConvRes(ActorCritic):
    """
    implements both actor and critic in one model, residually
    """

    def __init__(
        self,
        board_size, # e.g. 17
        num_actions,
        in_channels,
        actor_residual_settings=[[17, 13], [13, 9]],
        flattened_dim_actor=729, 
        kernel=5,
        critic_residual_settings=[[17, 13], [13, 9]], # FIXME unused as of yet
        flattened_dim_critic=729, 
        alpha=1,
        dropout=0.0,
        gamma=0.99,
        **kwargs,
    ):

        super().__init__()

        state_dim = in_channels  # hidden channels which are input to A and to C

        # --------------- ENCODER --------------

        self._encoder = nn.Sequential(
            nn.Conv2d(in_channels, state_dim, kernel_size=1), nn.ReLU6()
        )

        # --------------- ACTOR -----------------

        actor_blocks = []
        residual_prev_channel = state_dim

        pad = kernel // 2
        padding = [[pad,0] for _ in range(len(actor_residual_settings))] # adjust based on actor residual settings

        for i, residual_channels in enumerate(actor_residual_settings):
            residual_layers = []
            # build sequential with given channels, and
            # and add downsample connection to directly go down

            IO_channel = residual_prev_channel

            for j, residual_channel in enumerate(residual_channels):

                residual_layers += [
                    nn.Conv2d(residual_prev_channel, residual_channel, kernel, padding=padding[i][j]),  # FIXME padding = 2 !!!
                    nn.LayerNorm(residual_channel),
                    nn.Dropout(p=dropout),
                    nn.ReLU6(),
                ]
                residual_prev_channel = residual_channel
            
            # add the depthwise downsample to get to same size directly:
            downsample_depthw = nn.Sequential(*[
                nn.Conv2d(
                    IO_channel, residual_prev_channel*IO_channel, kernel, 
                    groups=IO_channel, 
                    padding=pad
                ),
                nn.Conv2d(IO_channel*residual_prev_channel, residual_prev_channel, kernel),
                nn.LayerNorm(residual_prev_channel),
                nn.Dropout(p=dropout),
                nn.ReLU6(),
            ])

            actor_blocks.append(
                Residual(downsample_depthw, *residual_layers, name="actor", num=i)
            )

        self.temperature = {"alpha": alpha}
        
        actor_blocks += [
            nn.Flatten(1),
            nn.Linear(flattened_dim_actor, num_actions),
            Temperature(self.temperature),
            nn.Softmax(dim=-1),
        ]

        self._action_head = nn.Sequential(*actor_blocks)

        # --------------- CRITIC -----------------

        critic_blocks = []
        residual_prev_channel = state_dim

        padding = [[pad,0] for _ in range(len(actor_residual_settings))] # adjust based on actor residual settings

        for i, residual_channels in enumerate(critic_residual_settings):
            residual_layers = []
            # build sequential with given channels, and
            # and add downsample connection to directly go down

            IO_channel = residual_prev_channel

            for j, residual_channel in enumerate(residual_channels):

                residual_layers += [
                    nn.Conv2d(residual_prev_channel, residual_channel, kernel, padding=padding[i][j]),
                    nn.LayerNorm(residual_channel),
                    nn.Dropout(p=dropout),
                    nn.ReLU6(),
                ]
                residual_prev_channel = residual_channel
            
            # add the depthwise downsample to get to same size directly:
            downsample_depthw = nn.Sequential(*[
                nn.Conv2d(
                    IO_channel, residual_prev_channel*IO_channel, kernel, 
                    groups=IO_channel, 
                    padding=pad
                ),
                nn.Conv2d(IO_channel*residual_prev_channel, residual_prev_channel, kernel),
                nn.LayerNorm(residual_prev_channel),
                nn.Dropout(p=dropout),
                nn.ReLU6(),
            ])

            critic_blocks.append(
                Residual(downsample_depthw, *residual_layers, name="critic", num=i)
            )
        
        critic_blocks += [
            nn.Flatten(1),
            nn.Linear(flattened_dim_critic, 1)
        ]

        self._value_head = nn.Sequential(*critic_blocks)
 

class ActorCriticDepthwiseConvResTransformer(ActorCritic):
    """
    implements both actor and critic in one model, 
        basically like ActorCriticTransformer,
            but with a depthwise convolutional encoder,
                consisting of residual blocks
    """

    def __init__(
        self,
        board_size, # e.g. 17
        num_actions,
        in_channels,
        encoder_residual_settings=[[17, 17]],
        kernel=5,
        num_transf_layers=2,
        num_heads=2,
        alpha=1,
        dropout=0.0,
        **kwargs,
    ):

        super().__init__()

        state_dim = in_channels  # hidden channels which are input to A and to C


        # --------------- ENCODER -----------------

        encoder_blocks = []
        residual_prev_channel = state_dim

        pad = kernel // 2
        padding = [[pad,pad] for _ in range(len(encoder_residual_settings))] # adjust based on encoder residual settings

        for i, residual_channels in enumerate(encoder_residual_settings):
            residual_layers = []
            # build sequential with given channels, and
            # and add downsample connection to directly go down

            IO_channel = residual_prev_channel

            for j, residual_channel in enumerate(residual_channels):

                residual_layers += [
                    # depthwise
                    nn.Conv2d(
                        residual_prev_channel, 
                        residual_prev_channel * residual_channel, 
                        kernel, 
                        padding=padding[i][j],
                        groups=residual_prev_channel,
                    ),
                    nn.Conv2d(
                        residual_prev_channel * residual_channel,
                        residual_channel,
                        kernel,
                        padding=padding[i][j],
                    ),
                    nn.LayerNorm(residual_channel),
                    nn.Dropout(p=dropout),
                    nn.ReLU6(),
                ]
                residual_prev_channel = residual_channel
            
            # add the residual to get to same size with only one conv:
            ds = nn.Sequential(*[
                nn.Conv2d(
                    IO_channel,
                    residual_channel,
                    kernel,
                    padding=pad
                ),
                nn.ReLU6()
            ])

            encoder_blocks.append(
                Residual(ds, *residual_layers, name="encoder", num=i)
            )

        self._encoder = nn.Sequential(*encoder_blocks)

        # ----------------- ACTOR ------------------

        actor = ViT(
            image_size=residual_channel,
            channels=residual_channel,
            patch_size=1,
            num_classes=num_actions,
            dim=num_heads * residual_channel,
            depth=num_transf_layers,
            heads=num_heads,
            mlp_dim=num_heads * residual_channel,
            dropout=dropout,
            emb_dropout=dropout,
        )
        
        self.temperature = {"alpha": alpha}
        
        actor_blocks = [
            actor,
            Temperature(self.temperature),
            nn.Softmax(dim=-1),
        ]

        self._action_head = nn.Sequential(*actor_blocks)

        # --------------- CRITIC -----------------

        critic = ViT(
            image_size=residual_channel,
            channels=residual_channel,
            patch_size=1,
            num_classes=1,
            dim=num_heads * residual_channel,
            depth=num_transf_layers,
            heads=num_heads,
            mlp_dim=num_heads * residual_channel,
            dropout=dropout,
            emb_dropout=dropout,
        )

        self._value_head = critic
 



class ActorCriticTransformer(ActorCritic):
    """
    implements both actor and critic in one model, conventionally
    """

    def __init__(
        self,
        board_size,  # e.g. 17
        num_channels,
        num_actions,
        hidden_dim,
        num_heads,
        mlp_dim,
        num_layers=2,
        critic_hiddens=[9],
        alpha=1.0,
        dropout=0.00,
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
            channels=num_channels,
            patch_size=1,
            num_classes=num_actions,
            dim=hidden_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=dropout,
        )

        self.temperature = {"alpha": alpha}

        self._action_head = nn.Sequential(
            actor, 
            Temperature(self.temperature),
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
            nn.Linear(critic_prev_hidden * num_channels, 1),
        ]

        self._value_head = nn.Sequential(*critic_layers)



