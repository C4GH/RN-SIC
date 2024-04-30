import torch
import torch.optim as optim
import torch.nn as nn


def train(model, dataloader, epochs):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    for epoch in range(epochs):
        for i, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)

            if torch.isnan(output).any():
                print(f"Epoch {epoch}, Batch {i}: NaN detected in model output")
                continue  # Skip this batch

            loss = criterion(output, target)

            if torch.isnan(loss):
                print(f"Epoch {epoch}, Batch {i}: Loss is NaN")
                continue  # Skip the backward pass if loss is NaN

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Apply gradient clipping
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"Epoch {epoch}, Batch {i}, Loss lowered to {best_loss}, model saved!")

            if i % 10 == 0:
                print(f'Epoch {epoch}, Batch {i}, Current Loss: {loss.item()}')


def test(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            if torch.isnan(output).any():
                print("NaN detected in model output during testing")
                continue  # Skip this batch
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({accuracy:.0f}%)')
