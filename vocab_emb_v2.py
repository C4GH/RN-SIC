import numpy as np
import torch
import os

# Cargar el contenido del archivo
with open('C:\\Users\\afloresre\\Documents\\Cagh\\Red\\vectors_test.txt', 'rt') as fi:
    # Procesa el archivo aquí
    contenido = [line.strip().split(' ') for line in fi.readlines()]

# Separar palabras y embeddings
vocab, embeddings = zip(*[(line[0], list(map(float, line[1:]))) for line in contenido])

# Convertir a numpy arrays
vocab_npa = np.array(vocab, dtype='<U10') # ajuste del tipo de datos.
embs_npa = np.array(embeddings)

# Insertar tokens especiales
vocab_npa = np.insert(vocab_npa, 0, ['<pad>', '<sep>'])
pad_emb_npa = np.zeros((1, embs_npa.shape[1])) # Embedding para '<pad>'
sep_emd_npa = np.mean(embs_npa, axis=0, keepdims=True) # Embedding promedio para '<sep>'

# Añadir embeddings para tokens especiales
embs_npa =np.vstack((pad_emb_npa, sep_emd_npa, embs_npa))

# Crea la capa de embedding
my_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())

# Asegurar que las dimensiones coincidan
assert my_embedding_layer.weight.shape == embs_npa.shape

# guardar vocabulario y embeddings
if not os.path.exists('./salida'):
    os.makedirs('./salida')
np.save('./salida/vocab_npa.npy', vocab_npa)
np.save('./salida/embs_npa.npy', embs_npa)

