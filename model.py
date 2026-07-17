import torch
import torch.nn as nn

class MiniUNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.block1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.up_block1 = nn.Sequential(
            nn.Conv2d(64,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.up_block2 = nn.Sequential(
            nn.Conv2d(32,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2
        )
        self.up2 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            stride=2
        )
        self.to_pred = nn.Conv2d(
            in_channels=16,
            out_channels=1,
            kernel_size=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,X):
        x1 = self.block1(X)
        p1 = self.pool(x1)

        x2 = self.block2(p1)
        p2 = self.pool(x2)

        x3 = self.block3(p2)

        
        u1 = self.up1(x3)
        c1 = torch.cat([u1,x2],dim=1)
        y1 = self.up_block1(c1)

        
        u2 = self.up2(y1)
        c2 = torch.cat([u2,x1],dim=1)
        y2 = self.up_block2(c2)

        pred = self.to_pred(y2)

        return pred
    