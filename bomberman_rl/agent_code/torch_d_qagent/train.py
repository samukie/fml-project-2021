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
    """
    BATCH_SIZE = 64
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    """

    self.optimizer = optim.RMSprop(self.policy_net.parameters())
    self.memory = ReplayMemory(10000)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.game =1

def state_is_equal(state1, state2):
    state1.sort()
    state2.sort()
    return state1 == state2

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
        current_action, current_obs = get_action_and_observation(self, old_game_state)
        next_action, next_obs= get_action_and_observation(self, new_game_state)
        reward = reward_from_events(self, events)
        reward -= new_game_state['step']
        if self.target: 
            prev_dist =  get_distance(old_game_state['self'][3], self.target, self.board_size)
            curr_dist =  get_distance(new_game_state['self'][3], self.target, self.board_size)
            if curr_dist < prev_dist:
                reward+=1000
            else:
                if not e.COIN_COLLECTED in events:
                    reward-=2000
        else: 
            reward +=10000
        """
        if len(self.memory)>=3: 
            #print(self.memory.pull(3)[0].state)
            if state_is_equal(self.memory.pull(3)[0].state, self.memory.pull(3)[-1].state):
                print('now!')
                #eward-=10
            else:
                reward+=50
        """

        print(reward)
        self.memory.push(current_obs,torch.as_tensor(current_action),torch.as_tensor(next_obs),torch.FloatTensor([reward]))
        optimize_model(self.memory, self.policy_net, self.target_net, self.optimizer)
 

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    next_states = tuple(torch.as_tensor(x).float() for x in batch.next_state)                                         
    non_final_next_states = torch.stack(next_states)
    states = tuple(torch.as_tensor(x).float() for x in batch.state)                                         
    state_batch = torch.stack(states)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.game +=1
    torch.save(self.policy_net.state_dict(), "policy_net.pt")
    torch.save(self.target_net.state_dict(), "target_net.pt")

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5000,
        #e.INVALID_ACTION: -50,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT:-1,
        e.MOVED_UP: -1,
        e.MOVED_RIGHT:-1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
