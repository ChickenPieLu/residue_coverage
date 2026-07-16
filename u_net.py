import torch
import torch.nn as nn

def main():
    torch.manual_seed(114514)

    X = torch.randn(4,3,512,512)

    block1 = nn.Sequential(
        nn.Conv2d(3,16,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(16,16,kernel_size=3,padding=1),
        nn.ReLU()
    )
    block2 = nn.Sequential(
        nn.Conv2d(16,32,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(32,32,kernel_size=3,padding=1),
        nn.ReLU()
    )
    block3 = nn.Sequential(
        nn.Conv2d(32,64,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(64,64,kernel_size=3,padding=1),
        nn.ReLU()
    )
    up_block1 = nn.Sequential(
        nn.Conv2d(64,32,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(32,32,kernel_size=3,padding=1),
        nn.ReLU()
    )
    up_block2 = nn.Sequential(
        nn.Conv2d(32,16,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(16,16,kernel_size=3,padding=1),
        nn.ReLU()
    )
    pool = nn.MaxPool2d(kernel_size=2, stride=2)

    x1 = block1(X)
    p1 = pool(x1)

    x2 = block2(p1)
    p2 = pool(x2)

    x3 = block3(p2)

    up1 = nn.ConvTranspose2d(
        in_channels=64,
        out_channels=32,
        kernel_size=2,
        stride=2
    )
    u1 = up1(x3)
    c1 = torch.cat([u1,x2],dim=1)
    y1 = up_block1(c1)

    up2 = nn.ConvTranspose2d(
        in_channels=32,
        out_channels=16,
        kernel_size=2,
        stride=2
    )
    u2 = up2(y1)
    c2 = torch.cat([u2,x1],dim=1)
    y2 = up_block2(c2)

    to_pred = nn.Conv2d(
        in_channels=16,
        out_channels=1,
        kernel_size=1
    )
    pred = to_pred(y2)

    print(pred.shape)

if __name__ == "__main__":
    main()