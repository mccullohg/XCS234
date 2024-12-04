import copy
import random
import numpy as np

class RandomMDP:
    def __init__(self, seed=1234):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.num_states = 10
        self.num_actions = 2

        self.R = np.random.rand(self.num_states, self.num_actions)
        self.T = np.random.rand(self.num_states, self.num_actions, self.num_states)

        self.num_succs = 5
        eye = np.eye(self.num_states)
        self.rand_succs = np.array([[np.sum(eye[np.random.choice(self.num_states, self.num_succs, replace=False)], axis=0) for _ in range(self.num_actions)] for _ in range(self.num_states)])
        assert self.rand_succs.shape == self.T.shape

        # Each state-action pair only transitions to a subset of random states
        self.T = self.T * self.rand_succs

        # Normalize to ensure distribution
        normalizer = np.sum(self.T, axis=2, keepdims=True)
        self.T = self.T / normalizer
        assert np.all(np.isclose(np.sum(self.T, axis=2), 1.))

    def get_model(self):
        return copy.deepcopy(self.R), copy.deepcopy(self.T)

    def reset(self):
        return np.random.choice(self.num_states)

    def step(self, action):
        reward = self.R[self.curr_state, action]
        next_state = np.random.choice(range(self.num_states), p=self.T[self.curr_state, action])
        self.curr_state = next_state
        return reward, next_state

if __name__ == '__main__':
    env = RandomMDP()