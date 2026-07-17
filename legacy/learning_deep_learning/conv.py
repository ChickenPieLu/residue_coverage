import torch
import torch.nn as nn
import torch.optim as optim

def true_fn(X):
    red = X[:,0,:,:]
    green = X[:,1,:,:]
    mask = (red > green).float()
    mask = mask.unsqueeze(1)
    return mask

def main():
    torch.manual_seed(42)

    X = torch.randn(1, 3, 32, 32)
    X_test = torch.randn(1, 3, 32, 32)
    mask = true_fn(X)
    test_mask = true_fn(X_test)
    conv = nn.Conv2d(
        in_channels=3,
        out_channels=1,
        kernel_size=1,
    )
    optimiser = optim.SGD(
        conv.parameters(),
        lr = 0.2
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1000):
        optimiser.zero_grad()

        output = conv(X)
        loss = loss_fn(output,mask)

        # if epoch % 50 == 0:
        #     print(f"epoch {epoch}:")
        #     print(f"loss: {loss:.6f}")
        #     print()

        loss.backward()
        optimiser.step()

    with torch.no_grad():
        output = conv(X)
        probability = torch.sigmoid(output)
        prediction = (probability >0.5).float()
        accuracy = (prediction==mask).float().mean()
        print(f"Training set accuracy: {accuracy:.6f}")

        test_output = conv(X_test)
        test_probability = torch.sigmoid(test_output)
        test_prediction = (test_probability >0.5).float()
        test_accuracy = (test_prediction==test_mask).float().mean()
        print(f"Testing set accuracy: {test_accuracy:.6f}")

    print("Weights:")
    print(conv.weight)
    print("Bias:")
    print(conv.bias)

if __name__ == "__main__":
    main()