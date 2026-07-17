import torch
import torch.nn as nn
import torch.optim as optim

def true_fn(x):
    return 2*x + 1

def main():
    torch.manual_seed(114514)

    X = torch.linspace(0,1,100).reshape(-1,1)
    y = true_fn(X)

    model = nn.Linear(1,1)
    loss_fn = nn.MSELoss()
    optimiser = optim.SGD(
            model.parameters(),
            lr=0.1
        )
    
    for epoch in range(501):
        optimiser.zero_grad()

        prediction = model(X)
        loss = loss_fn(prediction,y)

        if epoch %20 == 0:
            print(f"epoch {epoch}: ")
            print(f"loss: {loss.item():.6f}")
            print(f"weight: {model.weight.item():.4f}")
            print(f"bias: {model.bias.item():.4f}")
            print()
        
        loss.backward()
        optimiser.step()
        
    
if __name__ == "__main__":
    main()