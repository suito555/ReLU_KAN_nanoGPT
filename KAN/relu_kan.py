import torch
import torch.nn as nn

class ReLUKANLinear(nn.Module):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 g=5,
                 k=3,
                 train_ab: bool = True):
        super().__init__()
        self.g, self.k, self.r = g, k, 4 * g * g / ((k + 1) * (k + 1))
        self.input_size, self.output_size = input_size, output_size
        phase_low = torch.arange(-k, g) / g
        phase_high = phase_low + (k + 1) / g
        self.phase_low = nn.Parameter(phase_low[None, :].expand(input_size, -1),
                                      requires_grad=train_ab)
        self.phase_high = nn.Parameter(phase_high[None, :].expand(input_size, -1),
                                         requires_grad=train_ab)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g + k, input_size))

    def forward(self, x):
        #x = x.view(x.size(0), 1)
        x = x.view(x.size(0), x.size(1), 1)
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_high - x)
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        x = x.reshape((len(x), self.output_size))
        return x


class ReLUKAN(nn.Module):

    def __init__(self, width, grid=5, k=3):
        super().__init__()
        self.width = width
        self.grid = grid
        self.k = k
        self.rk_layers = []
        for i in range(len(width) - 1):
            self.rk_layers.append(ReLUKANLayer(width[i], width[i + 1],grid, k))
        self.rk_layers = nn.ModuleList(self.rk_layers)

    def forward(self, x):
        for rk_layer in self.rk_layers:
            #x = x.view(x.size(0), 1)
            x = x.view(x.size(0), x.size(1), 1)
            x = rk_layer(x)
        return x