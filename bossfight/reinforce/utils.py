import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RIGHT = 7
LEFT = 0

def preprocess_batch(observation):
    mean = np.mean(observation, axis = (2,3), keepdims=True)
    batch_input = (observation - mean)/255
    batch_input = np.moveaxis(batch_input, 3, 1)
    return torch.from_numpy(batch_input).float().to(device)

# convert states to probability, passing through the policy
def states_to_prob(policy, states):
    states = torch.stack(states)
    policy_input = states.view(-1, *states.shape[-3:])
    return policy(policy_input).view(states.shape[:-3])


# return sum of log-prob divided by T
# same thing as -policy_loss
def surrogate(policy, old_probs, states, actions, rewards,
              discount, beta):

    discount = discount ** np.arange(len(actions))

    # rewards = np.asarray(rewards) * discount[:, np.newaxis]
    rewards = rewards * discount[:, np.newaxis]
    """
    If there are any -1 in the reward list for an episode, we set all rewards in said episode to be 0
    Otherwise, we set all rewards in episode to be 0, except for the very last reward, which we set to be 1 or 2?
    """

    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]

    # convert everything into pytorch tensors and move to gpu if available
    actions = np.asarray(actions)
    actions = torch.from_numpy(actions)
    actions = actions.to(torch.int8)
    actions = actions.to(device)

    old_probs = np.asarray(old_probs)
    old_probs = torch.from_numpy(old_probs)
    old_probs = old_probs.to(torch.float)
    old_probs = old_probs.to(device)

    rewards = np.asarray(rewards_normalized)
    rewards = torch.from_numpy(rewards)
    rewards = rewards.to(torch.float)
    rewards = rewards.to(device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0 - new_probs)

    ratio = new_probs / old_probs

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs * torch.log(old_probs + 1.e-10) +
                (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

    return torch.mean(ratio * rewards + beta * entropy)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    """
    def __init__(self):
        super(Policy, self).__init__()

        # 80x80 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride
        # (round up if not an integer)

        # conv1 : 80 x 80 -> 40 x 40
        self.conv1 = nn.Conv2d(2, 4, kernel_size=2, stride=2)
        # conv2 : 40 x 40 -> 20 x 20
        self.conv2 = nn.Conv2d(4, 8, kernel_size=2, stride=2)
        # conv3 : 20 x 20 -> 10 x 10
        self.conv3 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
        # conv4 : 10 x 10 ->  5 x  5
        self.conv4 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.size = 32 * 5 * 5

        # 1 fully connected layer
        self.fc1 = nn.Linear(self.size, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x"""

    def __init__(self):
        """
        proposed dimension transformations
        64 x 64 x 3 to 30 x 30 x 8, to 9 x 9 x 16
        """
        super(Policy, self).__init__()
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        #self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        self.conv1 = nn.Conv2d(3, 8, kernel_size= 6, stride=2, bias=False)

        #self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=6 , stride= 3)
        self.size = 9 * 9 * 16

        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)

        # Sigmoid to
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = x.view(-1, self.size)
        x = x.reshape(-1, self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))

class Shallow(nn.Module):
    def __init__(self):
        """
        proposed dimension transformations
        chop off to 3, 35 , 64
        flatten to 6720
        pass through sigmoid
        """
        super(Shallow, self).__init__()
        self.size = 6720

        # fully connected layer
        self.fc1 = nn.Linear(self.size, 1)

        # Sigmoid to
        self.sig = nn.Sigmoid()

        # flatten
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x[:, :, :35, :]
        x = self.flatten(x)
        x = self.fc1(x)
        return self.sig(x)

class Twolayer(nn.Module):
    def __init__(self):
        """
        proposed transformations
        chop off to 3, 35 , 64
        flatten to 6720
        fully connected to 4480
        ReLU
        fully connected to 1
        sigmoid
        """
        super(Twolayer, self).__init__()
        self.size = 6720

        # fully connected layer
        self.fc1 = nn.Linear(self.size, 4480)
        self.fc2 = nn.Linear(4480, 1)

        # Sigmoid to
        self.sig = nn.Sigmoid()

        # flatten
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x[:, :, :35, :]
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sig(x)
