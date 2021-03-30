import os
import pickle
import random
import operator
import numpy as np
from .DQN import *


ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', "WAIT", "BOMB"] 


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
        n_actions = len(ACTIONS)
        """
        self.policy_net = DQN(32, 256, n_actions).to(device)
        self.target_net = DQN(32, 256, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        """
        n_actions = len(ACTIONS)
        self.policy_net = DQN(32, 256, n_actions).to(device)
        self.target_net = DQN(32, 256, n_actions).to(device)

        self.policy_net.load_state_dict(torch.load("policy_net.pt"))
        self.target_net.load_state_dict(torch.load("target_net.pt"))

        self.policy_net.eval()
        self.target_net.eval()
        print('contructed') 
        #self.l3
        
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            #self.model = pickle.load(file)
            #self.model = keras.models.load_model("my-saved-model.pt")
            #self.policy_net = DQN(screen_height, screen_width, n_actions).to(device)
            #self.target_net = DQN(screen_height, screen_width, n_actions).to(device)
            n_actions = len(ACTIONS)
            self.policy_net = DQN(32, 64, n_actions).to(device)
            self.target_net = DQN(32, 64, n_actions).to(device)

            self.policy_net.load_state_dict(torch.load("policy_net.pt"))
            self.target_net.load_state_dict(torch.load("target_net.pt"))

            self.policy_net.eval()
            self.target_net.eval()
            print('loaded')
    self.target = False
    self.bomb_target = False
    self.board_size = 17
    self.action_dict = {
        0:'LEFT',
        1:'RIGHT',
        2:'UP',
        3:'DOWN',
        4:'WAIT',
        5:'BOMB'
    }


def get_better_environment(game_state, target=False, bomb_target=False):
    _, _, _, (x, y) = game_state['self']
    arena = game_state['field']
    # encode explosions
    explosions = game_state["explosion_map"].nonzero()
    arena[explosions] = 2
    state = [x,y]
    arena_info = [
        arena[x+1, y], 
        arena[x-1, y], 
        arena[x, y+1], 
        arena[x, y-1],
        arena[x-1, y-1],
        arena[x+1, y+1],
        arena[x+1, y-1],
        arena[x-1, y+1],
        ]
    state.extend(arena_info)
    # reserve 4 4 player
    
    player_state = []
    for player in game_state['others']:
        player_state.append(player[3][0])
        player_state.append(player[3][1])
    player_state += [-2] * (8 - len(player_state))
    state.extend(player_state)
    #print(player_state)
    #coin_distances = [get_distance([x,y], coin) for coin in game_state['coins']]
    bomb_state = []
    for bomb in game_state["bombs"]:
        bomb_state.append(bomb[0][0])
        bomb_state.append(bomb[0][0])
    bomb_state += [-2] * (8 - len(bomb_state))
    state.extend(bomb_state)
    #print(bomb_state)
    coin_distances = [get_distance([x,y], coin, 17) for coin in game_state['coins']]
    sorted_coins = [coin for _,coin in sorted(zip(coin_distances,game_state["coins"]))]
    #for coin in sorted_coins[]:
    if sorted_coins !=[]:
        state.append(sorted_coins[0][0])
        state.append(sorted_coins[0][1])
    else: 
        state.append(-2)
        state.append(-2)
    #coin_state += [-2] * (4 - len(coin_state))
    #state.extend(coin_state)
    #print(coin_state)
    
    if target==False: 
        state.append(0)
        state.append(0)
    else: 
        state.append(target[0])
        state.append(target[1])

    
    if bomb_target==False: 
        state.append(0)
        state.append(0)
    else: 
        state.append(bomb_target[0])
        state.append(bomb_target[1])
    
    #print(state)
    #print('bomb target', target)
    #print(state)
    #return torch.from_numpy(arena).view(1,arena.shape[0]*arena.shape[1]).float()
    return state

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

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
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    #if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def get_targets(game_state):
    targets = []
    x = game_state['bombs'][0][0][0]
    y = game_state['bombs'][0][0][1]
    if x+1 <16: 
        if y+1<16: 
            targets.append([x+1, y+1])
        elif y-1>0:
            targets.append([x+1, y-1])
    elif x-1 >0:
        if y+1<16: 
            targets.append([x-1, y+1])
        elif y-1>0:
            targets.append([x-1, y-1])
    elif x+2 <16: 
        if y+1<16: 
            targets.append([x+2, y+1])
        elif y-1>0:
            targets.append([x+2, y-1])
    elif x-2 >0:
        if y+1<16: 
            targets.append([x-2, y+1])
        elif y-1>0:
            targets.append([x-2, y-1])
    elif y+2 <16: 
        if x+1<16: 
            targets.append([x-1, y+2])
        elif x-1>0:
            targets.append([x+1, y+2])
    elif y-2 >0:
        if x+1<16: 
            targets.append([x-1, y-2])
        elif x-1>0:
            targets.append([x+1, y-2])

    zeros = game_state['field']==0
    invalids= [[x+i,y] for i in range(0,4)] + [[x-i,y] for i in range(1,4)] + \
        [[x,y+i] for i in range(1,4)] + [[x,y-i] for i in range(1,4)]
    
    valid_entries = [[i,j] for i in range(zeros.shape[0]) for j in range(zeros.shape[1]) \
        if zeros[i,j] == True and [i,j] not in invalids]
    
    #print('targets ', targets)
    #print('cleaned targets ',  zeros)
    #print(valid_entries)
    return valid_entries




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

def get_distance(current, target, board_size):
    return np.sum(np.abs(np.subtract(target, current)))

def target_is_valid(target, game_state): 
    is_valid = False
    for coin in game_state['coins']:
        if coin[0]==target[0] and coin[1]==target[1]:
            is_valid=True
    return is_valid

def get_coin_representation(coin_coordinate):
    decimal = coin_coordinate[0]*19**0 + coin_coordinate[1]*19**1
    return decimal

"""
def get_free_field(game_state):
    b_x = game_state['bombs'][0][0][0]
    b_y = game_state['bombs'][0][0][1]
    if game_state['field'][b_x+1, b_y] == 0: 
        if game_state['field'][b_x+1, b_y+1] == 0: 
            return [b_x+1, b_y+1]
        elif game_state['field'][b_x+1, b_y-1] == 0:
            return [b_x+1, b_y-1]
        elif game_state['field'][b_x+2, b_y] == 0:
            if game_state['field'][b_x+2, b_y+1] == 0: 
                return [b_x+2, b_y+1]
            elif game_state['field'][b_x+2, b_y-1] == 0: 
                return [b_x+2, b_y-1]

    elif game_state['field'][b_x-1, b_y] == 0: 
        if game_state['field'][b_x-1, b_y+1] == 0: 
            return [b_x-1, b_y+1]
        elif game_state['field'][b_x-1, b_y-1] == 0:
            return [b_x-1, b_y-1]
        elif game_state['field'][b_x-2, b_y] == 0:
            if game_state['field'][b_x-2, b_y+1] == 0: 
                return [b_x-2, b_y+1]
            elif game_state['field'][b_x-2, b_y-1] == 0: 
                return [b_x-2, b_y-1]

    if game_state['field'][b_x, b_y+1] == 0: 
        if game_state['field'][b_x+1, b_y+1] == 0: 
            return [b_x+1, b_y+1]
        elif game_state['field'][b_x-1, b_y+1] == 0:
            return [b_x-1, b_y+1]
        elif game_state['field'][b_x, b_y+2] == 0:
            if game_state['field'][b_x+1, b_y+2] == 0: 
                return [b_x+1, b_y+2]
            elif game_state['field'][b_x-1, b_y+2] == 0: 
                return [b_x-1, b_y+2]

    elif game_state['field'][b_x, b_y-1] == 0: 
        if game_state['field'][b_x+1, b_y-1] == 0: 
            return [b_x+1, b_y-1]
        elif game_state['field'][b_x-1, b_y-1] == 0:
            return [b_x-1, b_y-1]
        elif game_state['field'][b_x, b_y-2] == 0:
            if game_state['field'][b_x+1, b_y-2] == 0: 
                return [b_x+1, b_y-2]
            elif game_state['field'][b_x-1, b_y-2] == 0: 
                return [b_x-1, b_y-2]
"""


def bread_first_search(game_state, start, targets):
    def get_neighbors(node, visited_nodes):
        neighs = []
        if node[0]+1 <16: 
            neighs.append([node[0]+1,node[1]])
        if node[1]+1 <16:     
            neighs.append([node[0],node[1]+1])
        if node[0]-1 >0:
            neighs.append([node[0]-1,node[1]])
        if node[1]-1 >0:
            neighs.append([node[0],node[1]-1])
        valid_neighs = [neigh for neigh in neighs if game_state[neigh[0], neigh[1]] \
            and neigh not in visited_nodes]
        return valid_neighs
    all_nodes = [start]
    visited_nodes = [start]
    while all_nodes !=[]: 
        current = all_nodes.pop()
        visited_nodes.append(current)
        if current in targets: 
            return current
        neighs = get_neighbors(current, visited_nodes)
        all_nodes.extend(neighs)
    return False

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

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
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def get_targets(game_state):
    targets = []
    _, _, _, (self_x, self_y) = game_state['self']
    x = game_state['bombs'][0][0][0]
    y = game_state['bombs'][0][0][1]

    if x+1 <16: 
        if y+1<16: 
            targets.append([x+1, y+1])
        elif y-1>0:
            targets.append([x+1, y-1])
    elif x-1 >0:
        if y+1<16: 
            targets.append([x-1, y+1])
        elif y-1>0:
            targets.append([x-1, y-1])
    elif x+2 <16: 
        if y+1<16: 
            targets.append([x+2, y+1])
        elif y-1>0:
            targets.append([x+2, y-1])
    elif x-2 >0:
        if y+1<16: 
            targets.append([x-2, y+1])
        elif y-1>0:
            targets.append([x-2, y-1])
    elif y+2 <16: 
        if x+1<16: 
            targets.append([x-1, y+2])
        elif x-1>0:
            targets.append([x+1, y+2])
    elif y-2 >0:
        if x+1<16: 
            targets.append([x-1, y-2])
        elif x-1>0:
            targets.append([x+1, y-2])

    zeros = game_state['field']==0
    invalids= [[x+i,y] for i in range(0,4)] + [[x-i,y] for i in range(1,4)] + \
        [[x,y+i] for i in range(1,4)] + [[x,y-i] for i in range(1,4)]
    valid_entries = [[i,j] for i in range(zeros.shape[0]) for j in range(zeros.shape[1]) \
        if zeros[i,j] == True and [i,j] not in invalids if get_distance([i,j], [self_x,self_y], 17) < 5]
    #print('targets ', targets)
    #print('cleaned targets ',  zeros)
    #print(valid_entries)

    return valid_entries

def get_action_and_observation(self, game_state):
    _, _, _, (x, y) = game_state['self']
    current = (x,y)
    arena = game_state['field']
    # update target
    bombs = [0,0]
    if game_state['bombs'] != [] :
        bomb_coords = [bomb[0] for bomb in game_state['bombs']]
        bomb_distances = [get_distance([x,y], bomb, self.board_size) for bomb in bomb_coords]
        sorted_bombs = [bomb for _,bomb in sorted(zip(bomb_distances,bomb_coords))]
        bombs[0] = sorted_bombs[0][0]
        bombs[1] = sorted_bombs[0][1]
        #print(bombs)
        #self.bomb_target = bombs
        targets = []
        free_space = arena == 0
        self.bomb_target = bread_first_search(free_space, bombs, get_targets(game_state))
        #print('bomb ', bombs)
        #print('bomb target ', self.bomb_target)
    else: 
        self.bomb_target = False
    if self.target: 
        if not target_is_valid(self.target, game_state):
            self.target=False
    if not self.target and not self.bomb_target:
        if not game_state['coins']==[]:
            self.target =  game_state['coins'][get_minimum(current, game_state['coins'], self.board_size)]
            self.target = False

    explosions = game_state["explosion_map"].nonzero()
    #plosions] = 2
    #state = [x,y, arena[x+1, y], arena[x-1, y], arena[x, y+1], arena[x, y-1]]
    state = get_better_environment(game_state, self.target,self.bomb_target)
    """
    if np.random.random() > 0.01 and self.target:
        state.extend([self.target[0],self.target[1]])
        best_action = torch.argmax(self.policy_net(torch.FloatTensor(state)))
    """
    """
    if self.target: 
        state.extend([self.target[0], self.targets[1]])
    else: 
        state.extend([0,0])
    if self.bomb_target: 
        state.extend([self.bomb_target[0],self.bomb_target[1]])
    else: 
        state.extend([0,0])
    """
    #print(state)
    #1/(self.game)
    if np.random.random() > 0.01:
        best_action = torch.argmax(self.policy_net(torch.FloatTensor(state)))
    else:
        best_action = torch.as_tensor(np.random.randint(0, len(ACTIONS)))
    return best_action, state

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    action, _ = get_action_and_observation(self, game_state)
    #print(self.action_dict[action.item()])
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
