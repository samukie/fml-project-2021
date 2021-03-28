import os
import pickle
import random
import numpy as np
from random import shuffle

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
EPSILON_MAX = 0.1
EPSILON_MIN = 0.025
EPSILON_DECAY = 0.9999
# EPSILON_MIN reached after about 7000 its


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

    ### CHANGE ONLY IF YOU KNOW WHAT YOU ARE DOING, no sync with training!
    self.model_suffix = ""

    # TEST BENCH CODE
    if "TESTING" in os.environ:
        if os.environ["TESTING"] == "YES":
            self.test_results = {"crates": [], "total_crates": []}
            self.model_suffix = "_" + os.environ["MODELNAME"]
            self.total_crates = 0
            self.last_crates = 0
            print("WARNING: TESTING (perhaps on a different model!)")
    if self.train or not os.path.isfile(f"model{self.model_suffix}/model.pt"):
        self.logger.info("Setting up model from scratch.")
        # self.regressor = Regressor(8)
        self.epsilon = EPSILON_MAX
    else:
        self.logger.info("Loading model.")
        print("Loading model.")
        with open(f"model{self.model_suffix}/model.pt", "rb") as file:
            self.Q = pickle.load(file)



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # print("-----------------")

    feat = state_to_features(game_state)
    self.logger.debug(
        "Querying model for action with feature " + str(tuple(feat)) + "."
    )
    self.logger.debug(f"save:{feat[:5]}")
    self.logger.debug(f"POI:{feat[5:7]}, type:{feat[7]}, distance:{feat[8]}")
    self.logger.debug(f"NEY:{feat[9:11]}, distance:{feat[11]}")
    self.logger.debug(f"bomb:{feat[12]}")
    # TEST BENCH CODE
    if "TESTING" in os.environ:
        if os.environ["TESTING"] == "YES":
            crates = np.count_nonzero(game_state["field"] == 1)
            if self.total_crates == 0 or self.last_crates < crates:
                self.total_crates = crates
                self.test_results["total_crates"].append(crates)
                self.test_results["crates"].append(crates)
            elif self.last_crates > crates:
                self.test_results["crates"][-1] = crates
            self.last_crates = crates

            with open(f"model{self.model_suffix}/test_results.pt", "wb") as file:
                pickle.dump(self.test_results, file)

    # ->EPSILON greedy

    if self.train and random.random() < self.epsilon:
        self.logger.debug("EPSILON-greedy: Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    # start = time.time()
    # Qs = self.regressor.predict(feat.reshape(1, -1))
    # print(2, 2, 2, 2, 2, 3, 3, 2, 3, 3, 3, 5, len(ACTIONS))
    # print(feat)
    Qs = self.Q[tuple(feat)]
    self.logger.debug(f"Qs for this situation:{Qs}")

    action_index = np.random.choice(np.flatnonzero(Qs == np.max(Qs)))
    # self.logger.debug("Choosing an action took " + str((time.time() - start)) + "ms.")

    # -> soft-max

    # ROUNDS = 100000
    # rho = np.clip((1 - game_state["round"]/ROUNDS)*0.7, a_min=1e-3, a_max=0.5) # starte sehr kalt, wegen gutem anfangsQ
    # Qvals = self.Q[:, int(feat[0] + 14), int(feat[1] + 14), int(feat[2]), int(feat[3])]
    # softmax = np.exp(Qvals/rho)/np.sum(np.exp(Qvals/rho))
    # self.logger.debug("softmax:" + str(softmax))
    # action_index = np.random.choice(np.arange(len(ACTIONS)), p=softmax)

    # print(feat)
    # print("ACTION choosen: " + ACTIONS[action_index])
    self.logger.debug("ACTION choosen: " + ACTIONS[action_index])
    return ACTIONS[action_index]


# def Q(self, X) -> np.array:
#     Xs = np.tile(X, (6, 1))
#     a = np.reshape(np.arange(6), (6, 1))
#     xa = np.concatenate((Xs, a), axis=1)
#     return self.regressor.predict(xa)


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
    # WIRD IM MOMENT 3 MAL AUSGEFÜHRT

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    # TODO: channels?

    # mod_pos = [game_state["self"][3][0] % 2, game_state["self"][3][1] % 2]

    x_off = [0, 1, 0, -1]
    y_off = [-1, 0, 1, 0]

    # compute save (danger level for nearest bombs)
    bombs = game_state["bombs"]
    free_space = game_state["field"] == 0
    for bomb in bombs: # TODO: gegnerposition entfernen
        free_space[tuple(bomb[0])] = False
    for other in game_state["others"]:
        free_space[tuple(other[3])] = False

    start = game_state["self"][3]

    save = [0, 0, 0, 0, int(not is_dangerous(start, bombs))]
    x, y = start

    # print("current position not save!")
    # else: search if tiles are save
    neighbors = [
        (x, y)
        for (x, y) in [
            (x + x_off[0], y + y_off[0]),
            (x + x_off[1], y + y_off[1]),
            (x + x_off[2], y + y_off[2]),
            (x + x_off[3], y + y_off[3]),
        ]
    ]

    for i, neighbor in enumerate(neighbors):
        if free_space[neighbor]:
            # print("checking..", neighbor)
            if not is_dangerous(neighbor, bombs):
                # print("neighbor is save!", neighbor)
                save[i] = 1
                continue
            frontier = [neighbor]
            parent_dict = {start: start, neighbor: neighbor}
            dist_so_far = {neighbor: 1}
            while len(frontier) > 0:
                current = frontier.pop(0)

                # search for most dangerous bomb TODO
                danger_dist_min = np.infty
                most_dangerous_bomb = None
                for bomb in bombs:
                    dist = bomb[0] - np.array(neighbor)
                    if any(dist == 0):
                        if np.sum(np.abs(dist)) < danger_dist_min:
                            danger_dist_min = np.sum(np.abs(dist))
                            most_dangerous_bomb = bomb

                if dist_so_far[current] > most_dangerous_bomb[1]:
                    # print("too far: stopping here", current)
                    continue
                x, y = current
                available_neighbors = [
                    (x, y)
                    for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),]
                    if free_space[x, y]
                ]

                for neineighbor in available_neighbors:
                    if neineighbor not in parent_dict:
                        frontier.append(neineighbor)
                        parent_dict[neineighbor] = neighbor
                        dist_so_far[neineighbor] = dist_so_far[current] + 1
                        if not is_dangerous(neineighbor, bombs):
                            save[i] = 1
                            # print("found save spot at ", neineighbor)
                            break
                else:
                    continue
                break
        # print(save)
    # part for computing POI and save
    # for crates, coins: BFS

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    found_targets = []  # list of tuples (*coord, type)

    free_space = game_state["field"] == 0

    while len(frontier) > 0:
        current = frontier.pop(0)
        if current in game_state["coins"]:
            found_targets.append([current, 1, dist_so_far[current]])

        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [
            (x, y)
            for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            if free_space[x, y]
        ]
        all_neighbors = [
            (x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        ]
        shuffle(all_neighbors)
        for neighbor in all_neighbors:
            if game_state["field"][neighbor] == 1:  # CRATE
                found_targets.append([neighbor, 0, dist_so_far[current] + 1])

        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

        # print(found_targets)
        if len(found_targets) == 0:
            POI_position = game_state["self"][3]
            POI_type = 0  # TODO: Verwirrung?
            POI_dist = 0
        else:
            founds = sorted(found_targets, key=lambda tar: tar[2])
            # for f in founds:
            #     if f[2] < 4:
            #
            #     if f[1] == 1: #COIN
            #         found = f
            #         break
            # else:
            found = founds[0]
            POI_position = found[0]
            POI_type = found[1]
    # compute POI vector and dist
    dist = POI_position - np.array(game_state["self"][3])
    POI_vector = np.sign(dist)
    if dist[0] != dist[1]:
        bigger = np.argmax(np.abs(dist))
        POI_vector[bigger] *= 2
    POI_vector += 2
    POI_dist = np.clip(np.sum(np.abs(dist)), a_max=2, a_min=1)-1  # 012
    # compute Nearest enemy (NEY) vector and dist
    if game_state["others"] != []:
        manhattan_min = np.infty

        for other in game_state["others"]:
            manhattan =  np.sum(np.abs(np.array(other[3]) - np.array(game_state["self"][3])))
            if manhattan <= manhattan_min:
                nearest_enemy = other
                manhattan_min = manhattan
        dist = other[3] - np.array(game_state["self"][3])
        NEY_vector = np.sign(dist)
        if dist[0] != dist[1]:
            bigger = np.argmax(np.abs(dist))
            NEY_vector[bigger] *= 2
        NEY_vector += 2
        NEY_dist = np.clip(np.sum(np.abs(dist)), a_max=4, a_min=1)-1  #01234 # 0123
    else:
        NEY_vector = [2,2]
        NEY_dist = 3  # TODO: Verwirrung II

    bomb_left = int(game_state["self"][2])
    # channels = []
    # channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)
    # and return them as a vector
    # 2 2 2 2 2, 3 3, 2, 3 | 3 3, 5
    return np.concatenate(
        (save, POI_vector, [POI_type], [POI_dist], NEY_vector, [NEY_dist], [bomb_left])
    ).astype(int)
    # stacked_channels.reshape(-1)


#### UTILITY FUNCTIONS


def get_all_rotations(index_vector):
    # rots = np.reshape(
    #     index_vector, (1, -1)
    # )  # makes list that only contains index_vector
    rots = [tuple(index_vector)]
    # would be cool to find a more readable numpy way...
    flipped_vector = flip(index_vector)  # check if already symmetric
    if flipped_vector not in rots:
        rots.append(flipped_vector)

    for i in range(0, 3):
        index_vector = rotate(index_vector)
        if index_vector not in rots:
            rots.append(index_vector)
        flipped_vector = flip(index_vector)
        if flipped_vector not in rots:
            rots.append(flipped_vector)
    return rots


def rotate(index_vector):
    """
    Rotates the state vector 90 degrees clockwise.
    """
    # TODO: BETTER FEATURES
    # feat = index_vector[:-1]
    # feat = np.reshape(feat, (7,7))
    # visual_feedback = False
    # if visual_feedback:
    #     visualize(feat, index_vector[-1])

    # rot = np.rot90(feat, k=1)

    if index_vector[-1] <= 3:  # DIRECTIONAL ACTION -> add 1
        action_index = (index_vector[-1] + 1) % 4
    else:
        action_index = index_vector[-1]  # BOMB and WAIT invariant

    rot = (
        index_vector[3],  # save tiles
        index_vector[0],
        index_vector[1],
        index_vector[2],
        index_vector[4],
        -index_vector[6] + 4,  # POI vector y->-x
        index_vector[5],  # x->y
        index_vector[7],  # POI type invariant
        index_vector[8],  # POI distance invariant
        -index_vector[10] + 4,  # NEY vector y->-x
        index_vector[9],  # x->y
        index_vector[11],  # NEY distance invariant
        index_vector[12],  # boms_left
        action_index,
    )
    # if visual_feedback:
    #     visualize(rot, action_index)
    #     print("=================================================================================")

    return rot

    # return np.concatenate((np.reshape(rot, (-1)), [action_index]))


def flip(index_vector):
    """
    Flips the state vector left to right.
    """
    # feat = index_vector[:-1]
    # feat = np.reshape(feat, (7, 7))
    # TODO: BETTER FEATURES
    # visual_feedback = False
    # if visual_feedback:
    #    visualize(feat, index_vector[-1])
    # flip = np.flipud(feat)      # our left right is their up down (coords are switched), check with visual feedback if you don't believe it ;)
    if index_vector[-1] == 1:  # LEFT RIGHT-> switch
        action_index = 3
    elif index_vector[-1] == 3:
        action_index = 1
    else:
        action_index = index_vector[-1]  # UP, DOWN, BOMB and WAIT invariant

    flip = (
        index_vector[0],  # surrounding
        index_vector[3],
        index_vector[2],
        index_vector[1],
        index_vector[4],
        # -2 |-1 0 1 | 2
        # 0 | 1 2 3 | 4
        -index_vector[5] + 4,  # POI vector x->-x
        index_vector[6],  # y->y
        index_vector[7],  # POI type invariant
        index_vector[8],  # POI distance invariant
        -index_vector[9] + 4,  # NEY vector x->-x
        index_vector[10],  # y->y
        index_vector[11], # NEY distance invariant
        index_vector[12],  # boms_left
        action_index,
    )
    # if visual_feedback:
    #     visualize(flip, action_index)
    #     print("=================================================================================")
    return flip
    # return np.concatenate((np.reshape(flip, (-1)), [action_index]))


def is_dangerous(pos, bombs):
    if bombs == []:
        return False

    for bomb in bombs:
        dist = bomb[0] - np.array(pos)
        d = np.sum(np.abs(dist))

        if d < 4 and any(dist == 0):
            return True
    return False
