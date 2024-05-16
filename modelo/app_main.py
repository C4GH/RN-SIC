import requests
from urllib.parse import quote
import torch
from torch.nn import functional as F
from modelo.ModelC import Modelo
from embedding import Tokenizador

# Paths to necessary files
VOCAB_PATH = r"C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\vocab_npa.npy"
MODELO_ARCH = r"C:\Users\afloresre\Documents\Cagh\Red\modelo\best_model_10Epochs.pth"
URL_BASE = 'https://sic.cultura.gob.mx/utiles/cosic/xcon.php?busquedaavanzada='

# Model parameters
vocab_size = 1866360
embedding_dim = 300
d_model = embedding_dim
nhead = 4
num_layers = 3
num_tokens = 300
num_classes = 3
dropout = 0.05

# Initialize tokenizer and model
print("Loading Vocabulary")
tokenizer = Tokenizador.Tokenizador(VOCAB_PATH)
print("Loading Model")
model = Modelo(vocab_size, embedding_dim, d_model, nhead, num_layers, num_tokens, num_classes, dropout)
model.load_state_dict(torch.load(MODELO_ARCH))
model.eval()

# Disable gradient computation
torch.set_grad_enabled(False)


def prepara_datos(data, pregunta):
    """
    Prepare data for the model by concatenating relevant information from the data.
    Include a line jump before each 'nombre' for better readability.
    """
    text = ' '.join([f"\n{item['nombre']} {item['contexto']}" for item in data])
    return f'{pregunta} {text}'


def fetch_and_predict(query):
    encoded_query = quote(query)
    url = f"{URL_BASE}{encoded_query}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data_rec = response.json()
            processed_data = prepara_datos(data_rec, query)
            tokens = tokenizer.vectoriza(processed_data)
            tokens = tokens.unsqueeze(0)  # Add batch dimension
            logits = model(tokens)
            probabilities = F.softmax(logits, dim=1).squeeze().tolist()

            # Print fetched data with corresponding probabilities
            for idx, item in enumerate(data_rec):
                print(f"Opción {idx + 1}: {item['nombre']} {item['contexto']} - Probabilidad: {probabilities[idx]:.4f}")
        else:
            print(f"HTTP Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


if __name__ == '__main__':
    while True:
        user_input = input("Ingresa tu búsqueda (o 'f' para salir): ")
        if user_input.lower() == 'f':
            break
        fetch_and_predict(user_input)
