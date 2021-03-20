import pickle
import random
from collections import namedtuple, deque
from typing import List

from .DQN import *

import events as e
from .callbacks import *

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.lr = 0.3
    self.discount = 0.85

    DISCOUNT = 0.99
    REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
    MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
    MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
    UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
    MODEL_NAME = '2x256'
    MIN_REWARD = -200  # For model save
    MEMORY_FRACTION = 0.20

    # Environment settings
    EPISODES = 20_000

    # Exploration settings
    self.epsilon = 1  # not a constant, going to be decayed
    EPSILON_DECAY = 0.99975
    MIN_EPSILON = 0.001

    #  Stats settings
    AGGREGATE_STATS_EVERY = 50  # episodes
    SHOW_PREVIEW = False

    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

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
        #print('model ', self.model)
        done = False 

        current_action, current_obs = get_action_and_observation(self, old_game_state)
        next_action, next_obs= get_action_and_observation(self, new_game_state)

        print('curr', current_action)

        #inverted_actions = {v: k for k, v in self.action_dict.items()}
        #current_action_index = inverted_actions[current_action]

        current_surrounding = get_environment(old_game_state)
        future_surrounding = get_environment(new_game_state)
        
        #current_value = self.model[current_obs[0]][current_obs[1]][current_surrounding][current_action_index]
        #max_future_value = np.max(self.model[next_obs[0]][next_obs[1]][future_surrounding])

        reward = reward_from_events(self, events)


        # additional_reward 
        if old_game_state['coins']!=[] and new_game_state['coins']!=[]:
            prev_dist =  get_minimum_distance(old_game_state['self'][3], old_game_state['coins'], self.board_size)
            curr_dist =  get_minimum_distance(new_game_state['self'][3], new_game_state['coins'], self.board_size)
            if curr_dist < prev_dist:
                reward+=20
            elif curr_dist > prev_dist:
                reward-=20

        self.model.update_replay_memory((current_action, current_obs, reward, next_obs, done))
        self.model.train(done)
        #print("UPDATE!!!")

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    #self.model.model.save(f'my-saved-model.pt')

    #with open("my-saved-model.pt", "wb") as file:
    #    pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
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