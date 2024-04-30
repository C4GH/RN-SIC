from torch.utils.data import DataLoader
from TestModel import CustomDataset, Modelo, collate_fn
from train_utils import train, test


def main():
    # Paths to data and model
    embeddings_path = r"C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\embs_npa.npy"
    vocab_path = r"C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\vocab_npa.npy"
    json_path_train = r"C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\salida_entrenamiento.json"
    json_path_test = r"C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\salida_prueba.json"

    # Load datasets
    train_dataset = CustomDataset(json_path_train, embeddings_path, vocab_path)
    test_dataset = CustomDataset(json_path_test, embeddings_path, vocab_path)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Initialize the model
    model = Modelo(vocab_size=1866360, embedding_dim=300, d_model=300, nhead=4, num_layers=3, num_tokens=200, num_classes=3, dropout=0.05)

    # Train and test the model
    train(model, train_loader, epochs=5)
    test(model, test_loader)


if __name__ == '__main__':
    main()
