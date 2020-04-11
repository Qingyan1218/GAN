import torch.nn as nn
import numpy as np
import torch

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 反转卷积后图像尺寸N=(w-1)×s+k-2p w为输入，s是步长，k是卷积核，p是padding
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(100,24, kernel_size=4,stride=2,bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24,12, kernel_size=4,stride=2,bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 6, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, kernel_size=6, stride=2, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.model(z)
        return z

if __name__ == '__main__':
    z = np.random.randn(2,100,2,2)
    z = torch.from_numpy(z).type(torch.FloatTensor)
    print(z.shape)
    g = Generator()
    prob = g(z)
    # print(prob)
