import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression example. You may need to double check the dimensions :)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(110, 50) # 110 is the length of vocabulary
        self.fc2 = nn.Linear(50, 2)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        '''
        x (tensor): the input to the model
        '''
        x = x.squeeze(1).float()
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        return x


