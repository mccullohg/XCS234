import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.test_env import EnvTest
from utils.general import join
from core.deep_q_learning_torch import DQN
from .q1_schedule import LinearExploration, LinearSchedule

import yaml

yaml.add_constructor("!join", join)

config_file = open("config/q2_linear.yml")
config = yaml.load(config_file, Loader=yaml.FullLoader)

############################################################
# Problem 2: Linear Approximation
############################################################


class Linear(DQN):
    """
    Implementation of a single fully connected layer with Pytorch to be utilized
    in the DQN algorithm.
    """

    ############################################################
    # Problem 2b: initializing models

    def initialize_models(self):
        """
        Creates the 2 separate networks (Q network and Target network). The input
        to these networks will be an image of shape img_height * img_width with
        channels = n_channels * self.config["hyper_params"]["state_history"].

        - self.q_network (torch model): variable to store our q network implementation
        - self.target_network (torch model): variable to store our target network implementation

        TODO:
            (1) Set self.q_network to be a linear layer with num_actions as the output
            size.

            (2) Set self.target_network to be the same configuration as self.q_network.
            but initialized by scratch.

        Hint:
            (1) Start by figuring out what the input size is to the networks.
            (2) Simply setting self.target_network = self.q_network is incorrect.
            (3) Consult nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
            which should be useful for your implementation.
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        ### START CODE HERE ###
        # Linear layer input size image*channels
        input_size = img_height*img_width*n_channels*self.config["hyper_params"]["state_history"]

        # Define Q Network
        self.q_network = nn.Linear(input_size, num_actions)
        #self.q_network = nn.Sequential(nn.Flatten(), nn.Linear(input_size, num_actions))

        # Define Target Network from scratch
        self.target_network = nn.Linear(input_size, num_actions)
        #self.target_network = nn.Sequential(nn.Flatten(), nn.Linear(input_size, num_actions))
        ### END CODE HERE ###

    ############################################################
    # Problem 2c: get_q_values

    def get_q_values(self, state, network="q_network"):
        """
        Returns Q values for all actions.

        Args:
            state (torch tensor): shape = (batch_size, img height, img width,
                                            nchannels x config["hyper_params"]["state_history"])

            network (str): The name of the network, either "q_network" or "target_network"

        Returns:
            out (torch tensor): shape = (batch_size, num_actions)

        TODO:
            Perform a forward pass of the input state through the selected network
            and return the output values.


        Hints:
            (1) Look up torch.flatten (https://pytorch.org/docs/stable/generated/torch.flatten.html)
            (2) Pay attention to the torch.flatten `start_dim` Parameter 
            (3) You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None

        ### START CODE HERE ###
        # Validate network argument
        if network == "q_network":
            selected_network = self.q_network
        elif network == "target_network":
            selected_network = self.target_network
        else:
            raise ValueError("Invalid network selection")  # unnecessary 
        
        # Flatten the state tensor and preserve batch size
        state_flat = torch.flatten(state, start_dim=1)

        # Forward pass
        out = selected_network(state_flat)
        ### END CODE HERE ###

        return out

    ############################################################
    # Problem 2d: update_target

    def update_target(self):
        """
        The update_target function will be called periodically to copy self.q_network
        weights to self.target_network.

        TODO:
            Update the weights for the self.target_network with those of the
            self.q_network.

        Hint:
            Look up loading pytorch models with load_state_dict function.
            (https://pytorch.org/tutorials/beginner/saving_loading_models.html)
        """

        ### START CODE HERE ###
        self.target_network.load_state_dict(self.q_network.state_dict())
        ### END CODE HERE ###

    ############################################################
    # Problem 2e: calc_loss

    def calc_loss(
        self,
        q_values: torch.Tensor,
        target_q_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated_mask: torch.Tensor, 
        truncated_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the MSE loss of a given step. The loss for an example is defined:
            Q_samp(s) = r if terminated or truncated
                        = r + gamma * max_a' Q_target(s', a') otherwise
            loss = (Q_samp(s) - Q(s, a))^2

        Args:
            q_values: (torch tensor) shape = (batch_size, num_actions)
                The Q-values that your current network estimates (i.e. Q(s, a') for all a')

            target_q_values: (torch tensor) shape = (batch_size, num_actions)
                The Target Q-values that your target network estimates (i.e. (i.e. Q_target(s', a') for all a')

            actions: (torch tensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)

            rewards: (torch tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)
            
            terminated_mask: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where we reached the terminal state

            truncated_mask: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where the episode was truncated

        TODO:
            Return the MSE loss for a given step. You may use the function description
            for guidance in your implementation.

        Hint:
            You may find the following functions useful
                - torch.max (https://pytorch.org/docs/stable/generated/torch.max.html)
                - torch.sum (https://pytorch.org/docs/stable/generated/torch.sum.html)
                - torch.bitwise_or (https://pytorch.org/docs/stable/generated/torch.bitwise_or.html)
                - torch.gather:
                    * https://pytorch.org/docs/stable/generated/torch.gather.html
                    * https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms

            You may need to use the variables:
                - self.config["hyper_params"]["gamma"]
        """
        gamma = self.config["hyper_params"]["gamma"]
        ### START CODE HERE ###
        # Select Q(s,a)
        actions = actions.long()  # convert to int64 for gather function
        Qsa = torch.gather(q_values, dim=1, index=actions.unsqueeze(1)).squeeze(1)

        # Compute max
        maxQ = torch.max(target_q_values, dim=1)[0]

        # Mask for non-terminal states
        nonterminal_mask = torch.bitwise_or(terminated_mask, truncated_mask) == 0

        # Compute sample Q - does this include the terminal?
        Q_samp = rewards + nonterminal_mask*gamma*maxQ

        # Compute MSE loss
        loss = torch.mean((Q_samp-Qsa)**2)

        return loss
        ### END CODE HERE ###

    ############################################################
    # Problem 2f: add_optimizer

    def add_optimizer(self):
        """
        This function sets the optimizer for our linear network

        TODO:
            Set self.optimizer to be an Adam optimizer optimizing only the self.q_network
            parameters

        Hint:
            Look up torch.optim.Adam (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)
            What are the input to the optimizer's constructor?
        """
        ### START CODE HERE ###
        learning_rate = self.config["hyper_params"].get("learning_rate", 1e-3)  # necessary?
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        ### END CODE HERE ###
