from torch import nn
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize



class InnAiModel(nn.Module):
    def __init__(self):
        super(InnAiModel, self).__init__()


        self.lin1 = nn.Linear(28, 100)
        self.lin2 = nn.Linear(100, 4)
        self.lin3 = nn.Linear(4, 4)

    def forward(self, precipitationMaps, currentInnLevels):
        x1 = F.sigmoid(self.conv1(precipitationMaps).float())
        x1 = F.sigmoid(self.conv2(x1))

        x2 = F.sigmoid(self.lin1(currentInnLevels))

        x = torch.cat((x1.view(x1.size(0), -1),
                              x2.view(x2.size(0), -1)), dim=1)

        x = F.sigmoid(self.lin2(x))
        x = F.sigmoid(self.lin3(x))
        x = self.lin4(x)

        x1 = F.relu(self.lin1(precipitationMaps))
        x1 = F.relu(self.lin2(precipitationMaps))
        x1 = F.relu(self.lin3(precipitationMaps))

        return x
