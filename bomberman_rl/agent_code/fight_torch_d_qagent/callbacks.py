import os
import pickle
import random
import operator
import numpy as np
from .DQN import *

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', "WAIT", "BOMB"] 

def get_minimum(current, targets, board_size):
    #print(targets)
    if targets == []: 
        return -1
    else:
        return np.argmin(np.sum(np.abs(np.subtract(targets, current)), axis=1))

def get_minimum_distance(current, targets, board_size):
    if targets == []: 
        return False
    else:
        return np.sum(np.abs(np.subtract(targets, current)), axis=1).min()

def get_environment(game_state):
    #print(game_state)
    _, _, _, (x, y) = game_state['self']
    arena = game_state['field']
    # bomb = 2
    for bomb in game_state["bombs"]:
        arena[bomb[0][0], bomb[0][1]] = 2
    # other = 3
    for other in game_state["others"]:
        arena[other[3][0], other[3][1]] = 3
    # coins = 4    
    for coin in game_state["coins"]:
        arena[coin[0], coin[1]] = 4
    # self = 5 
    arena[game_state["self"][3][0], game_state["self"][3][1]] = 5
    #print(game_state)
    # return one hot encoding
    return torch.from_numpy(arena).view(1,arena.shape[0]*arena.shape[1])

def get_coin_representation(coin_coordinate):
    decimal = coin_coordinate[0]*19**0 + coin_coordinate[1]*19**1
    return decimal

def get_action_and_observation(self, game_state):
    
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    current = (x,y)
    # observation

    surrounding = get_environment(game_state)
    valid_actions = ['LEFT', 'RIGHT', 'UP', 'DOWN', "WAIT", "BOMB"] 
    min_coin_index =  get_minimum(current, game_state['coins'], self.board_size)
    # exploration
    state = get_environment(game_state)
    if np.random.random() > 0.1:
        # Get action from Q table
        
        #best_action = np.argmax(self.policy_net.([1,coin_decimal, surrounding]))
        #print("best action!!!! ", torch.argmax(self.policy_net(torch.FloatTensor(state))))
        #best_action = self.policy_net(torch.FloatTensor(state)).max(1)[1].view(1, 1)
        #print('actions', self.policy_net(torch.FloatTensor(state))
        #print(self.policy_net.shape)
        #print(state.shape)
        best_action = torch.argmax(self.policy_net(state.float()))
        #print("BESR", best_action)
    else:
        # Get random action
        best_action = torch.as_tensor(np.random.randint(0, len(ACTIONS)))
    #print(best_action)
    return best_action, state


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
        board_size = 17
        n_actions = len(ACTIONS)
        
        self.policy_net = DQN(board_size*board_size,120, n_actions).to(device)
        self.target_net = DQN(board_size*board_size,120, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        print('constructed')
        """
        n_actions = len(ACTIONS)
        self.policy_net = DQN(board_size*board_size,120, n_actions).to(device)
        self.target_net = DQN(board_size*board_size,120, n_actions).to(device)

        self.policy_net.load_state_dict(torch.load("policy_net.pt"))
        self.target_net.load_state_dict(torch.load("target_net.pt"))

        self.policy_net.eval()
        self.target_net.eval()
        print('contructed')
        """
    else:
        self.logger.info("Loading model from saved state.")
        #self.model = pickle.load(file)
        #self.model = keras.models.load_model("my-saved-model.pt")
        #self.policy_net = DQN(screen_height, screen_width, n_actions).to(device)
        #self.target_net = DQN(screen_height, screen_width, n_actions).to(device)
        n_actions = len(ACTIONS)
        board_size=17
        self.policy_net = DQN(board_size*board_size,120, n_actions).to(device)
        self.target_net = DQN(board_size*board_size,120, n_actions).to(device)

        self.policy_net.load_state_dict(torch.load("policy_net.pt"))
        self.target_net.load_state_dict(torch.load("target_net.pt"))

        self.policy_net.eval()
        self.target_net.eval()
        print('loaded')
    self.board_size = 17
    self.action_dict = {
        0:'LEFT',
        1:'RIGHT',
        2:'UP',
        3:'DOWN',
        4:'WAIT',
        5:'BOMB'
    }

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    action, _ = get_action_and_observation(self, game_state)
    #print(action)
    #print(action.shape)
    return  self.action_dict[action.item()]


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
