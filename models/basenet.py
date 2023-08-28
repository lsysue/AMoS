import torch
from torch import nn
import torch.nn.functional as F



class BaseFC(nn.Module):
    def __init__(self,
                 device='cpu',
                 d_inputs = 768, 
                 dropout = 0.2):
        super().__init__()

        self.device = device

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_inputs, d_inputs * 2)
        self.fc2 = nn.Linear(d_inputs * 2, d_inputs)

        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.layer_norm = nn.LayerNorm(d_inputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.fc2(x)
        x = self.leakyrelu(x)
        # x = self.layer_norm(x)
        # print(x)
        
        return x

class ThreeLayerFC(nn.Module):
    def __init__(self,
                 device='cpu',
                 d_inputs = 768,
                 dropout = 0.2):
        super().__init__()

        self.device = device

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_inputs, d_inputs * 2)
        self.fc2 = nn.Linear(d_inputs * 2, d_inputs * 2)
        self.fc3 = nn.Linear(d_inputs * 2, d_inputs)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.fc2(x)
        x = self.leakyrelu(x)
        x = self.fc3(x)
        x = self.leakyrelu(x)
        return x

class FourLayerFC(nn.Module):
    def __init__(self,
                 device='cpu',
                 d_inputs = 768,
                 dropout = 0.2):
        super().__init__()

        self.device = device

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_inputs, d_inputs * 2)
        self.fc2 = nn.Linear(d_inputs * 2, d_inputs * 2)
        self.fc3 = nn.Linear(d_inputs * 2, d_inputs * 2)
        self.fc4 = nn.Linear(d_inputs * 2, d_inputs)
        self.relu = nn.ReLU()
        self.leakyrelu1 = nn.LeakyReLU()
        self.leakyrelu2 = nn.LeakyReLU()
        self.leakyrelu3 = nn.LeakyReLU()
        self.leakyrelu4 = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc1(x)
        x = self.leakyrelu1(x)
        x = self.fc2(x)
        x = self.leakyrelu2(x)
        x = self.fc3(x)
        x = self.leakyrelu3(x)
        x = self.fc4(x)
        # x = self.sig(x)
        # x = self.leakyrelu4(x)
        # x = normalize(x)
        return x


def normalize(x):
    x_min = torch.min(x)
    if torch.isinf(x).any():
        x_masked = torch.where(torch.isinf(x), -1, x)
        x_max = torch.max(x_masked)
        x = torch.where(x_masked==-1, x_max, x)
    else:
        x_max = torch.max(x)
    if x_max == x_min:
        normalized_x = torch.ones_like(x)
    else:
        normalized_x = torch.div((x - x_min), (x_max - x_min))
    # print(normalized_x)
    # x_mean = torch.mean(normalized_x)
    # x_std = torch.std(normalized_x)
    # normalized_x = torch.div(normalized_x - x_mean, x_std)

    return normalized_x