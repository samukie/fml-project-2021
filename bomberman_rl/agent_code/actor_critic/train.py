import pickle
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


MODELS = "models/"

# This is only an example!
# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
# 
# # Hyper parameters -- DO modify
# TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
# 
# # Events
# PLACEHOLDER_EVENT = "PLACEHOLDER"


# -------------------- Helper functions ---------------------

def optimizer_to(optim, device):
    # via https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def get_minimum_distance(current, targets, board_size):
    if targets == []: 
        return -1
    else:
        min_dist = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        return min_dist



# ----------------------------- Training methods --------------------------------- 

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.lr = 3e-2
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    self.running_reward = 10 # arbitrarily initialize running reward
    self.EMA = 0.05 # Exponential moving average decay to calc running reward to display
    self.i_episode = 0

    if torch.cuda.is_available():
        device = "cuda"
        self.model.to(device)
        optimizer_to(self.optimizer, device)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    if old_game_state:

        # ------------------- REWARD ENGINEERING HERE ---------------

        # initialize rewards:
        reward = reward_from_events(self, events)

        # additional_reward 
        if old_game_state['coins']!=[] and new_game_state['coins']!=[]:

            prev_dist =  get_minimum_distance(old_game_state['self'][3], old_game_state['coins'], self.board_size)
            curr_dist =  get_minimum_distance(new_game_state['self'][3], new_game_state['coins'], self.board_size)

            if curr_dist == -1:
                reward += 1000
            elif curr_dist < prev_dist:
                reward += 20
            elif curr_dist > prev_dist:
                if not e.COIN_COLLECTED in events:
                    reward -= 20

        self.model.episode_rewards.append(reward)
        self.model.episode_reward += reward


    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    # NOTE: design rewards here:
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.INVALID_ACTION: -50
    }
    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # update cumulative reward
    self.running_reward = self.EMA * self.model.episode_reward + (1 - self.EMA) * self.running_reward

    self.logger.info('Episode {}\tLast reward: {:.2f}\tEMA reward: {:.2f}'.format(
        self.i_episode, self.model.episode_reward, self.running_reward))

    # main training code
    self.model.update(self.optimizer)

    self.i_episode += 1

    # Store the model
    # TODO only if its best ? torch.save? ckpt dict mit model/episode/...
    with open(self.model_path, "wb") as file:
        pickle.dump(self.model, file)

