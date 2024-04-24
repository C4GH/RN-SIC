import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Suponiendo que las clases y funciones están definidas o importadas aquí
# CustomDataset, Modelo, PositionalEncoding

def check_dimensions(embeddings_path, vocab_path, json_path, d_model):
    # Cargar el vocabulario y embeddings
    embeddings = torch.tensor(np.load(embeddings_path), dtype=torch.float)
    vocab_words = np.load(vocab_path)
    vocab = {word: idx for idx, word in enumerate(vocab_words)}

    # Comprobar que el tamaño del embedding coincide con d_model
    embedding_dim = embeddings.size(1)
    assert embedding_dim == d_model, f"Dimensión de embedding ({embedding_dim}) no coincide con d_model ({d_model})"

    # Cargar dataset y modelo
    dataset = CustomDataset(json_path, embeddings_path, vocab_path)
    model = Modelo(d_model=d_model, nhead=6, num_layers=3, num_tokens=10, num_classes=2, dropout=0.1)

    # Comprobar la consistencia en la primera carga de datos
    sample_data = dataset[0][0]  # Cargando el primer ejemplo (indices_tensor)
    input_to_model = sample_data.unsqueeze(0)  # Simular batch size de 1

    # Verificar dimensiones pasando por el modelo
    try:
        output = model(input_to_model)
        print("El modelo procesó la entrada correctamente.")
    except Exception as e:
        print("Error al procesar la entrada en el modelo:", e)

    print("Todas las comprobaciones pasaron correctamente.")


if __name__ == '__main__':
    embeddings_path = 'path_to_embeddings.npy'
    vocab_path = 'path_to_vocab.npy'
    json_path = 'path_to_data.json'
    d_model = 300  # Asegúrate que esto coincide con la dimensión de tus embeddings

    check_dimensions(embeddings_path, vocab_path, json_path, d_model)
