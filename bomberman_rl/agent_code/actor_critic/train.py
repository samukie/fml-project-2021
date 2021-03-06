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
MOVES = {e.MOVED_DOWN, e.MOVED_UP, e.MOVED_LEFT, e.MOVED_RIGHT}
e.APPROACHED = "APPROACHED"
e.DISTANCED = "DISTANCED"

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

def set_win_reward_if_not_set(self, game_state):
    if not hasattr(self, "win_reward"):
        reward_per_enemy = 500
        setattr(self, "win_reward", reward_per_enemy * len(game_state["others"]))

def argmax(l):
    # return set of indices of argmax elements
    # elems must be comparable with float by >, <
    the_max = float("-inf")
    r = set()
    for i, elem in enumerate(l):
        if elem == the_max:
            r.add(i)
        elif elem > the_max:
            r = set([i])
            the_max = elem
    return r

def get_minimum_distance(current, targets, board_size):
    if targets == []: 
        return -1
    else:
        min_dist = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        return min_dist

def reward_moving_closer_breadth_first(free_space, start, targets, move, logger=None):
    """
    Rewards moving closer to best target using breadth first search


    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        move: move made by 
        logger: optional logger object for debugging.
    Returns:
        True/False based on correct move
    """
    if len(targets) == 0: return None

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

        # Add unexplored free neighboring tiles to the queue in random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        random.shuffle(neighbors)

        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    closest = [best]
    # gather remaining targets with best 
    while len(frontier) > 0:
        candidate = frontier.pop(0)
        if dist_so_far[candidate] == best_dist:
            closest.append(candidate)
        

    for current in closest:
        # Determine the first step towards the best found target tile
        while True:
            if parent_dict[current] == start:
                break
            current = parent_dict[current]

        d = current
        
        x, y  = start
        if d == (x, y - 1) and move == e.MOVED_UP \
            or d == (x, y + 1) and move == e.MOVED_DOWN \
            or d == (x - 1, y) and move == e.MOVED_LEFT \
            or d == (x + 1, y) and move == e.MOVED_RIGHT:
                return True
    
    return False


# ----------------------------- Training methods --------------------------------- 


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.lr = 1e-3
    critic_mult = 2

    self.optimizer = self.optim_type([
        {"params": self.model._encoder.parameters(), "lr": self.lr * critic_mult},
        {"params": self.model._action_head.parameters(), "lr": self.lr},
        {"params": self.model._value_head.parameters(), "lr": self.lr * critic_mult},
    ])

    self.running_reward = 10 # arbitrarily initialize running reward
    self.EMA = 0.05 # Exponential moving average decay to calc running reward to display
    self.i_episode = 0

    if torch.cuda.is_available():
        device = "cuda"
        self.model._dev = device

        self.model.to(device)
        optimizer_to(self.optimizer, device)
    
    printdbg(f"Starting to train on {self.model._dev} ...")


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
        # set reward for win based on num opponents at start of game
        set_win_reward_if_not_set(self, old_game_state)

        # ------------------- REWARD ENGINEERING HERE ---------------

        # initialize rewards:
        reward = reward_from_events(self, events)

        # additional_reward 

        # coins
        if old_game_state['coins']!=[] and new_game_state['coins']!=[]:

            free_space = new_game_state["field"] == 0
            start = new_game_state["self"][3]
            targets = new_game_state["coins"]

            made_move = None
            for made_move in events:
                if made_move in MOVES:
                    break

            if made_move is not None:

                approach_made = reward_moving_closer_breadth_first(free_space, start, targets, made_move, logger=None)

                if approach_made:
                    APPROACH_REWARD = 30
                    reward += APPROACH_REWARD
                    printdbg(f"Awarded {APPROACH_REWARD} for getting closer to next coin.")
                    events.append(e.APPROACHED)
                else:
                    REPROACH_REWARD = 0
                    reward += REPROACH_REWARD
                    printdbg(f"Awarded {REPROACH_REWARD} for moving farther from next coin.")
                    events.append(e.DISTANCED)

            # this alternative reward would be sufficient for initial sparse coin task with no crates:

            """
            prev_dist = get_minimum_distance(old_game_state['self'][3], old_game_state['coins'], self.board_size)
            curr_dist = get_minimum_distance(new_game_state['self'][3], new_game_state['coins'], self.board_size)

            if curr_dist == -1:
                # all coins found (dont think this ever happens because of above if call preventing coin reward)

                # reward based on score now
                all_coins_found_reward = 1000

                self_score = new_game_state["self"][1]

                all_scores = [self_score]
                for other in new_game_state["others"]:
                    all_scores += [other[1]]
                best_indices = argmax(all_scores)

                if 0 in best_score_indices:
                    reward += all_coins_found_reward
                else:
                    reward -= all_coins_found_reward

            elif curr_dist < prev_dist:
                reward += 20
            elif curr_dist > prev_dist:
                if not e.COIN_COLLECTED in events:
                    reward -= 20
            """

        # scores
        # biggest_score_gain_reward = 40

        # self_gain = new_game_state["self"][1] - old_game_state["self"][1]
        # all_gains = [self_gain]

        # others_before = old_game_state["others"]
        # for other in new_game_state["others"]:
        #     name = other[0]
        #     other_score = other[1]
        #     for old_other in others_before:
        #         if name == old_other[0]:
        #             all_gains += [other_score - old_other[1]]

        # # give extra reward for being only one to increase score (this reward is calculated very inefficiently)
        # best_indices = argmax(all_gains)
        # if 0 in best_indices:
        #     reward += biggest_score_gain_reward # we gained the most score this step
        # else:
        #     reward -= biggest_score_gain_reward # someone else gained the most score this step

        # input(f"Reward: {reward}")

        self.model.reward(reward)

    print(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    # NOTE: design initial rewards here:
    game_rewards = {

        e.MOVED_LEFT : 5,
        e.MOVED_RIGHT : 5,
        e.MOVED_UP : 5,
        e.MOVED_DOWN : 5,
        e.WAITED : -100,
        e.INVALID_ACTION : -200,

        e.BOMB_DROPPED : 0,
        e.BOMB_EXPLODED : 500,

        e.CRATE_DESTROYED : 200,
        e.COIN_FOUND : 50,
        e.COIN_COLLECTED : 200,

        e.KILLED_OPPONENT : 4000,
        e.KILLED_SELF : -10000,

        e.GOT_KILLED : -12000,
        e.OPPONENT_ELIMINATED : 33333,
        e.SURVIVED_ROUND : 10000,
    }
    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    print(f"Awarded {reward_sum} for events {', '.join(events)}")
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

    printdbg(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # ------------------------ finalize model reward -------------------------
    # end_reward = 0
    # # 1. winning:

    # self_score = last_game_state["self"][1]
    # all_scores = [self_score]

    # for other in last_game_state["others"]:
    #     all_scores += [other[1]]

    # best_indices = argmax(all_scores)
    # if 0 in best_indices:
    #     end_reward += self.win_reward # dependent on number of opponents

    # self.model.reward(end_reward)

    episode_reward = sum(self.model.episode_rewards)

    # update cumulative reward (for logging)
    self.running_reward = self.EMA * episode_reward + (1 - self.EMA) * self.running_reward

    print('Episode {}\tNum Steps: {}\tReward SUM: {:.2f}\tEMA reward: {:.2f}'.format(
        self.i_episode, last_game_state["step"], episode_reward, self.running_reward))
    
    # main training code
    self.model.update(self.optimizer)

    self.i_episode += 1

    # reset current_action_dict
    self.current_action_dict = deepcopy(self.action_dict)

    printdbg(f"Round {self.i_episode}: storing model at "+self.model_path)
    # Store the model
    # TODO only if its best ? torch.save? ckpt dict mit model/episode/...
    torch.save(self.model, self.model_path)

