from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from util import np2torch


class RewardModel(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int, r_min: float, r_max: float
    ):
        """Initialize a reward model

        Parameters
        ----------
        obs_dim : int
            Dimension of the observation space
        action_dim : int
            Dimension of the action space
        hidden_dim : int
            Number of neurons in the hidden layer
        r_min : float
            Minimum reward value
        r_max : float
            Maximum reward value

        TODO:
        Define self.net to be a neural network with a single hidden layer of size
        hidden_dim that takes as input an observation and an action and outputs a
        reward value. Use LeakyRelu as hidden activation function, and set the
        activation function of the output layer so that the output of the network
        is guaranteed to be in the interval [0, 1].

        Define also self.optimizer to optimize the network parameters. Use a default
        AdamW optimizer.
        """

        super().__init__()
        #######################################################
        #########   2-10 lines.   ############
        ### START CODE HERE ###
        # Create neural network
        self.net = nn.Sequential(
            nn.Linear(obs_dim+action_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # [0,1]
        )

        # Define optimizer
        self.optimizer = torch.optim.AdamW(self.net.parameters())
        ### END CODE HERE ###
        #######################################################
        self.r_min = r_min
        self.r_max = r_max

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward callback for the RewardModel

        Parameters
        ----------
        obs : torch.Tensor
            Batch of observations
        action : torch.Tensor
            Batch of actions

        Returns
        -------
        torch.Tensor
            Batch of predicted rewards

        TODO:
        Use self.net to predict the rewards associated with the input
        (observation, action) pairs, and scale them so that they are
        within [self.r_min, self.r_max].

        """
        if obs.ndim == 3:
            B, T = obs.shape[:2]
            assert action.ndim == 3 and action.shape[:2] == (B, T)
            obs = obs.reshape(-1, obs.shape[-1])
            action = action.reshape(-1, action.shape[-1])
            needs_reshape = True
        else:
            needs_reshape = False

        rewards = torch.zeros(obs.shape[0])
        #######################################################
        #########   2-3 lines.   ############
        ### START CODE HERE ###
        input_tensor = torch.cat((obs, action), dim=-1)
        reward_normalized = self.net(input_tensor)

        # Scale to within r_min and r_max
        rewards = self.r_min + (self.r_max-self.r_min)*reward_normalized
        ### END CODE HERE ###
        #######################################################

        if needs_reshape:
            rewards = rewards.reshape(B, T)
        return rewards

    def compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Given an (observation, action) pair, return the predicted reward.

        Parameters
        ----------
        obs : np.ndarray (obs_dim, )
            A numpy array with an observation.
        action : np.ndarray (act_dim, )
            A numpy array with an action

        Returns
        -------
        float
            The predicted reward for the state-action pair.

        TODO:
        Return the predicted reward for the given (observation, action) pair. Pay
        attention to the argument and return types!

        Hint: If you use the forward method of this module remember that it takes
              in a batch of observations and a batch of actions.
        """
        #######################################################
        #########   1-4 lines.   ############
        ### START CODE HERE ###
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        reward = self.forward(obs_tensor, action_tensor)
        return reward.item()
        ### END CODE HERE ###
        #######################################################

    def update(self, batch: Tuple[torch.Tensor]):
        """Given a batch of data, update the reward model.

        Parameters
        ----------
        batch : Tuple[torch.Tensor]
            A batch with two trajectories (observations and actions) and a label
            encoding which one is prefered (0 if it is the first one, 1 otherwise).

        TODO:
        Compute the cumulative predicted rewards for each trajectory, and calculate
        your loss following the Bradley-Terry preference model.

        Hint 1: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
        Hint 2: https://stackoverflow.com/questions/57161524/trying-to-understand-cross-entropy-loss-in-pytorch
        """
        obs1, obs2, act1, act2, label = batch

        loss = torch.zeros(1)
        #######################################################
        #########   5-10 lines.   ############
        ### START CODE HERE ###
        # Calculate cumulative rewards
        cum_reward1 = self.forward(obs1, act1).sum(dim=1)
        cum_reward2 = self.forward(obs2, act2).sum(dim=1)

        # Compute logits from cumulative rewards
        logits = torch.stack([cum_reward1, cum_reward2], dim=1)

        # Assign logit probs according to label definitions
        probs = torch.stack((1-label, label), dim=1)

        # Use cross-entropy to calculate loss
        loss = torch.nn.functional.cross_entropy(logits, probs)
        ### END CODE HERE ###
        #######################################################

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()