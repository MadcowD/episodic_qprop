from collections import deque
import random

class EpisodicReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = []
        self._current_epsiode = []

    def get_episode(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        if self._current_epsiode:
            cum_reward = self._current_epsiode[-1][-1] + reward
        else:
            cum_reward = reward
        experience = [state, action, reward, new_state, done, cum_reward]

        self._current_epsiode += [experience]
        if done:
            if self.num_experiences < self.buffer_size:
                self.buffer.extend(self._current_epsiode)
                self.num_experiences += 1
            else:
                self.buffer= self.buffer[1:]
                self.buffer.extend(self._current_epsiode)

            self._current_epsiode = []

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self._current_epsiode = []
        self.num_experiences = 0