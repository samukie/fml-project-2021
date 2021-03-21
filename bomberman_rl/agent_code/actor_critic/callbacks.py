import os
import pickle
import random
import operator
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .actor_critic import *

torch.manual_seed(4269420)

MODELS = "models/"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

# ------------------ HELPER functions ---------------------

def get_minimum(current, targets, board_size):
    if targets == []: 
        return -1
    else:
        return np.argmin(np.sum(np.abs(np.subtract(targets, current)), axis=1))

def get_minimum_distance(current, targets, board_size):
    if targets == []: 
        return False
    else:
        return np.sum(np.abs(np.subtract(targets, current)), axis=1).min()


def get_features(namespace, game_state):
    # ------------------ FEATURE ENGINEERING HERE ---------------------
    # assert False, game_state

    # 1st feat: 4 neighboring fields empty or not

    # _, _, _, (x, y) = game_state['self']
    # arena = game_state['field']
    # directions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    # binary = ''
    # for index, direction in enumerate(directions): 
    #     if arena[directions[index]] == 0:
    #         binary += '1'
    #     else: 
    #         binary += '0'

    # to_decimal = 0 
    # for index, digit in enumerate(binary[len(binary)::-1]):
    #     to_decimal += int(digit)*2**(index)

    # feat1 = np.array([to_decimal])

    # # 2nd feats: nearest x, y coins

    # # observation
    # current = (x,y)

    # min_coin_index =  get_minimum(current, game_state['coins'], namespace.board_size)

    # if min_coin_index == -1:
    #     feat2 = np.array([10000, 10000])
    # else:
    #     min_coin = game_state['coins'][min_coin_index]
    #     feat2 = np.array([x-min_coin[0], y-min_coin[1]])
    # features = np.concatenate([feat1, feat2])

    conv_feats = []
    conv_feats += [game_state["field"]]
    conv_feats += [game_state["explosion_map"]]
    
    coin_feat = np.zeros((namespace.board_size, namespace.board_size))
    for coin in game_state["coins"]:
        coin_feat[coin[0], coin[1]] = 1
    conv_feats += [coin_feat]

    bomb_feat = np.zeros((namespace.board_size, namespace.board_size))
    for bomb in game_state["bombs"]:
        bomb_feat[bomb[0], bomb[1]] = 1
    conv_feats += [bomb_feat]

    agents_feat = np.zeros((namespace.board_size, namespace.board_size))
    scores_feat = np.zeros((namespace.board_size, namespace.board_size))

    agent_idx = 7
    my_agent_pos = game_state["self"][-1]
    agents_feat[my_agent_pos[0], my_agent_pos[1]] = agent_idx 
    scores_feat[my_agent_pos[0], my_agent_pos[1]] = game_state["self"][1] 

    for (other_score, other_bomb, other_pos) in [other[1:] for other in game_state["others"]]:
        agent_idx += 2
        agents_feat[other_pos[0], other_pos[1]] = agent_idx
        scores_feat[other_pos[0], other_pos[1]] = other_score

    conv_feats += [agents_feat]
    conv_feats += [scores_feat]

    features = np.concatenate([feat[np.newaxis,:,:] for feat in conv_feats], axis=0)
    assert features.shape[0] == namespace.num_features, f"Correct feature size: {features.shape[0]}, instead of {namespace.num_features}"

    return np.expand_dims(features, 0)

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # setup env info

    self.board_size = 17

    # NOTE: per task:  EDIT to allow/disallow actions; see final_project.pdf 
    self.action_dict = {
        0:'LEFT',
        1:'RIGHT',
        2:'UP',
        3:'DOWN',
        # 4:'BOMB', # disallow these for coin agent
        # 5:'WAIT',
    }
    self.num_features = 6 # determine by looking at get_features()
    self.num_actions = 1 # model outputs int to index action_dict

    self.model_path = MODELS+"my-saved-model.pt"

    if self.train or not os.path.isfile(self.model_path):
        self.logger.info("Setting up model from scratch.")

        # self.model = ActorCriticLinear(
        #     num_states=self.num_features,
        #     num_actions=self.num_actions,
        #     gamma=0.99,
        # )
    
        # self.model = ActorCriticConv(
        #     in_channels=self.num_features,
        #     board_size=self.board_size,
        #     num_actions=self.num_actions,
        #     gamma=0.99,
        # )

        num_heads = 6

        self.model = ActorCriticTransformer(
            board_size=self.board_size,
            num_states=self.num_features,
            num_actions=self.num_actions,
            hidden_dim=num_heads*self.board_size,
            num_heads=num_heads,
            mlp_dim=self.board_size*num_heads*2,
        )

        self.logger.info("Successfully set up model:")
        self.logger.info(self.model)
        
    else:
        self.logger.info("Loading model from saved state.")
        with open(self.model_path, "rb") as file:
            self.model = pickle.load(file)
    
    self.lr = 3e-2
    self.eps = 1e-7
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    self.running_reward = 10
    self.EMA = 0.05 # Exponential moving average decay to calc running reward to display



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return action_str: The action to take as a string.
    """
    features = get_features(self, game_state)
    
    action = self.model.select_action(features)

    action_str = self.action_dict[action]

    return action_str

