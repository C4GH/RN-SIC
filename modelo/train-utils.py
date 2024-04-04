# train_utils.py
import torch

def train_model(model, dataloader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs)
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
