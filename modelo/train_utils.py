import torch
import torch.optim as optim
import torch.nn as nn


def train(model, dataloader, epochs):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    for epoch in range(epochs):
        for data, target, in dataloader:
            optimizer.zero_grad()
            output = model(data)

            print(f"Output type: {output.dtype}, output shape: {output.shape}")
            print(f"Target type: {target.dtype}, output shape: {target.shape}")
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'Epoch {epoch}, Loss lowered to {best_loss}, model saved!')


def test(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({accuracy:.0f}%)')
