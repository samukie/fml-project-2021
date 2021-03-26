from collections import deque
from random import shuffle
from .DQN import *
import numpy as np
import os

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
    """
    rule_action = act(self,game_state)
    if rule_action==None:
        print('NONE!')
        rule_action = self.action_dict[np.random.randint(0, len(ACTIONS))]
    #print(rule_action)
    reverse_dict = {v:k for k,v in self.action_dict.items()}
    #print(reverse_dict)
    return reverse_dict[rule_action], state
    """
    return best_action, state



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
        shuffle(neighbors)
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


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()
        board_size = 17
        n_actions = len(ACTIONS)
        
        self.policy_net = DQN(board_size*board_size,128,128, n_actions).to(device)
        self.target_net = DQN(board_size*board_size,128,128, n_actions).to(device)
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
        self.policy_net = DQN(board_size*board_size,128,128, n_actions).to(device)
        self.target_net = DQN(board_size*board_size,128,128, n_actions).to(device)

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



    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """

    action, _ = get_action_and_observation(self, game_state)
    #print(action)
    return self.action_dict[action.item()]

    self.logger.info('Picking action according to rule set')

    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

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
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a


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