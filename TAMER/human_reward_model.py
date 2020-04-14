import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, 0, 1)


class HumanRewardModel(nn.Module):
    def __init__(self, env):
        super(HumanRewardModel, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = 50
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=False)
        self.l2 = nn.Linear(self.hidden, self.action_space, bias=False)

    def forward(self, x):
        model = torch.nn.Sequential(self.l1, self.l2)
        return model(x)
