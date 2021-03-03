import os
import pickle
import random
import operator
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

def construct_table(feature_size, board_size):
    # 8**17*17
    q_table =  np.zeros((board_size+2, board_size+2, len(ACTIONS)))
    return q_table

def get_minimum(current, targets, board_size):
    if targets == []: 
        return False
    else:
        return np.argmin(np.sum(np.abs(np.subtract(targets, current)), axis=1))
    

def get_minimum(current, targets, board_size):
    if targets == []: 
        return False
    else:
        return np.argmin(np.sum(np.abs(np.subtract(targets, current)), axis=1).min())

def get_minimum_distance(current, targets, board_size):
    if targets == []: 
        return False
    else:
        return np.sum(np.abs(np.subtract(targets, current)), axis=1).min()

def get_observation_and_action(self, game_state):
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    current = (x,y)
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] <= 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # We do not disallow BOMB actions
    # observation
    min_coin_index =  get_minimum(current, game_state['coins'], self.board_size)
    if min_coin_index == False:
        best_action = np.random.choice(ACTIONS, 1)
        observation = [18,18]
    else: 
        min_coin = game_state['coins'][min_coin_index]
        observation = [x-min_coin[0],y-min_coin[1]]
        action_values = self.model[observation[0]][observation[1]]
        action_value_dict = {index:action for index, action in enumerate(action_values)}
        self.logger.debug(f'Valid actions: {valid_actions}')
        inverted_actions = {v: k for k, v in self.action_dict.items()}
        invalid_actions = list(set(self.action_dict.values())-set(valid_actions))
        invalid_actions_indices = [inverted_actions[action] for action in invalid_actions]
        valid_action_values = {key:val for key, val in action_value_dict.items() if key not in invalid_actions_indices}
        best_action = self.action_dict[max(valid_action_values, key=valid_action_values.get)]
    return best_action[0], observation


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
    agent_dir = 'agent_code/qagent/'
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()
        self.model = construct_table(17,17)
        print('contructed')
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            print(self.model)
            print('loaded')
    self.board_size = 17
    self.action_dict = {
        0:'LEFT',
        1:'RIGHT',
        2:'UP',
        3:'DOWN',
        4:'WAIT',
    }

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    action, _ = get_observation_and_action(self, game_state)
    return action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
