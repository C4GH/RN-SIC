import torch
from torch.utils.data import DataLoader
from TestModel import CustomDataset, Modelo, collate_fn  # Ensure these are imported correctly
import torch.nn as nn


def diagnose_model(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()  # Assuming you are using CrossEntropyLoss, adjust if necessary

    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            if torch.isnan(data).any():
                print(f"Batch {i}: NaN detected in input data")
                continue

            outputs = model(data)
            if torch.isnan(outputs).any():
                print(f"Batch {i}: NaN detected in model output")
                continue  # Skip loss calculation if output is NaN

            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                print(f"Batch {i}: Loss is NaN")
            else:
                print(f"Batch {i}: Loss calculated without NaN - Loss: {loss.item()}")


def main():
    # Paths to your test data
    embeddings_path = r"C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\embs_npa.npy"
    vocab_path = r"C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\vocab_npa.npy"
    json_path_test = r"C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\salida_prueba.json"

    # Initialize the dataset and dataloader
    test_dataset = CustomDataset(json_path_test, embeddings_path, vocab_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize the model
    vocab_size = 1866360  # Adjust according to your vocabulary size
    embedding_dim = 300  # This should match the dimension of your embeddings
    d_model = embedding_dim
    num_classes = 3  # Adjust based on your number of classes
    num_tokens = 200  # This should match the expected sequence length

    model = Modelo(vocab_size, embedding_dim, d_model, 4, 3, num_tokens, num_classes, 0.05)

    # Optionally, load the model weights if necessary
    # model.load_state_dict(torch.load('path_to_your_saved_model_weights.pth'))

    # Run the diagnostic
    diagnose_model(model, test_loader)


if __name__ == '__main__':
    main()
