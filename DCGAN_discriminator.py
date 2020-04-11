import torch.nn as nn
import numpy as np
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3,24, kernel_size=4,stride=2,bias=False),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24,12, kernel_size=4,stride=2,bias=False),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(),
            nn.Conv2d(12, 6, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.Conv2d(6, 3, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(3)

        )

        self.FullCon=nn.Sequential(
            nn.Linear(27,1)
        )


    def forward(self, img):
        x = self.model(img)
        result=self.FullCon(x.view(-1,27))
        return result

if __name__=='__main__':
    x=np.random.randn(2,3,64,64)
    x=torch.from_numpy(x).type(torch.FloatTensor)
    d=Discriminator()
    prob=d(x)
    print(prob)

