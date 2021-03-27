import operator
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .actor_critic import *

torch.manual_seed(4269420)

MODELS = "models/"

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]

# ------------------ HELPER functions ---------------------


def look_for_targets(free_space, start, targets, logger=None):
    """
    Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [
            (x, y)
            for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            if free_space[x, y]
        ]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger:
        logger.debug(f"Suitable target found at {best}")
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]


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

def random_dieder4(arr, axes=(2,3)):
    # applies random flips and rotations along plane specified by given axes on a numpy array

    if len(arr.shape) == 2:
        axes = (0,1)
    assert len(axes) == 2, "need unambiguous plane to do transforms on"

    flip_axis = random.choice(axes)
    num_flip = random.choice([0,1])
    num_rot = random.choice([0,1,2,3])

    # flip
    if num_flip:
        arr = np.flip(arr, axis=flip_axis)

    # rotate
    arr = np.rot90(arr, num_rot, axes=axes)
    return arr




    pass


def get_features(namespace, game_state):
    # ------------------ FEATURE ENGINEERING HERE ---------------------

    # IDEA:
    # structure: np.array: feats x board x board
    # use entire gamestate,
    # making stuff we dont like negative and stuff we like positive
    # with magnitude indicating importance/expected reward

    conv_feats = []

    field = game_state["field"]
    field *= 10
    conv_feats += [field]

    explosions = game_state["explosion_map"]
    explosions *= -600
    conv_feats += [explosions]

    coin_feat = np.zeros((namespace.board_size, namespace.board_size))
    for coin in game_state["coins"]:
        coin_feat[coin[0], coin[1]] = 50
    conv_feats += [coin_feat]

    bomb_feat = np.zeros((namespace.board_size, namespace.board_size))
    for bomb in game_state["bombs"]:
        bomb_feat[bomb[0], bomb[1]] = -200
    conv_feats += [bomb_feat]

    agents_feat = np.zeros((namespace.board_size, namespace.board_size))
    scores_feat = np.zeros((namespace.board_size, namespace.board_size))

    my_agent_pos = game_state["self"][-1]
    agents_feat[my_agent_pos[0], my_agent_pos[1]] = -10  # hint at WAIT penalty?
    scores_feat[my_agent_pos[0], my_agent_pos[1]] = game_state["self"][1]

    self_bomb = game_state["self"][-2]

    for (other_score, other_bomb, other_pos) in [
        other[1:] for other in game_state["others"]
    ]:
        # determine based on availability of self+other's reward for getting to agent
        if self_bomb and other_bomb:
            agent_reward = 10
        elif self_bomb and not other_bomb:
            agent_reward = 60
        elif not self_bomb and other_bomb:
            agent_reward = -60
        elif not self_bomb and not other_bomb:
            agent_reward = 10

        agents_feat[other_pos[0], other_pos[1]] = agent_reward
        scores_feat[other_pos[0], other_pos[1]] = other_score

    conv_feats += [agents_feat]
    conv_feats += [scores_feat]

    # add noise to all features to prevent slice of tensor being all 0
    # (i think this was problematic in fwd)

    # shape = scores_feat.shape
    # eps = 1e-3
    # for feat in conv_feats:
    #     noise = np.random.rand(*shape) * eps
    #     feat = feat.astype(noise.dtype)
    #     feat += noise

    features = np.concatenate([feat[np.newaxis, :, :] for feat in conv_feats], axis=0)
    # features = np.concatenate([field[np.newaxis, :, :] for _ in range(6)])

    assert (
        features.shape[0] == namespace.num_features
    ), f"Correct feature size: {features.shape[0]}, instead of {namespace.num_features}"

    # add batch dim because transformer expects it:
    features = np.expand_dims(features, 0) # 1 x FEAT x BOARD x BOARD

    # randomly flip+rotate for better generalization:
    features = random_dieder4(arr, axes=(2,3))

    return features


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
        0:"LEFT",
        1:"RIGHT",
        2:"UP",
        3:"DOWN",
        4:'BOMB', # disallow these for coin agent
        5:'WAIT',
    }
    self.num_features = 6  # determine by looking at get_features()
    self.num_actions = len(self.action_dict)  # model outputs int to index action_dict

    typ = "Conv"
    self.model_path = MODELS + typ + ".pt"

    if not os.path.isfile(self.model_path):
        self.logger.info("Setting up model from scratch.")

        if typ == "Linear":
            self.model = ActorCriticLinear(
                num_states=self.num_features,
                num_actions=self.num_actions,
                gamma=0.99,
            )
        elif typ == "Conv":
            channels = [34, 68, 136, 272, 34, 17, 9]

            self.model = ActorCriticConv(
                in_channels=self.num_features,
                board_size=self.board_size,
                num_actions=self.num_actions,
                actor_channels=channels,
                critic_channels=channels,
            )
        elif typ == "Transformer":
            num_heads = 2
            self.model = ActorCriticTransformer(
                board_size=self.board_size,
                num_states=self.num_features,
                num_actions=self.num_actions,
                hidden_dim=num_heads * self.board_size,
                num_heads=num_heads,
                mlp_dim=self.board_size * num_heads * 2,
                num_layers=2,
            )

        self.logger.info("Successfully set up model:")
        self.logger.info(self.model)

    else:
        self.logger.info("Loading model from saved state.")
        with open(self.model_path, "rb") as file:
            self.model = pickle.load(file)


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
