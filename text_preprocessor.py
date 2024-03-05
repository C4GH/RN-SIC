import numpy as np
import torch

# Carga del vocabulario
def carga_vocabulario(path='salida/vocab_npa.npy'):
    vocab = np.load(path)
    list_vocab = vocab.tolist()
    vocab_dict = {word: index for index, word in enumerate(list_vocab)}
    valor_default = vocab_dict.get('<unk>', -1)  # Asume que tienes un token <unk> en tu vocabulario
    return vocab_dict, valor_default

def vectoriza_cadena(cad, vocab_dict, valor_default):
    """
    Convierte una cadena de texto en una lista de índices según el vocabulario proporcionado.

    Parameters:
    - cad (str): Cadena de texto a convertir.
    - vocab_dict (dict): Diccionario de vocabulario mapeando palabras a índices
    - valor_default (int): Valor a usar para palabras desconocidas.

    Returns:
    - List[int]: Lista de índices correspondientes a las palabras en la cadena.
    """
    cad = cad.lower().replace('.', '').replace(',', ' ').replace('  ', ' ').replace('-', ' ')
    aux = []
    for item in cad.split(" "):
        itaux = item.strip()
        if len(itaux) == 0:
            continue
        aux.append(vocab_dict.get(itaux, valor_default))
    return aux

# Ejemplo de uso
if __name__ == "__main__":
    vocab_dict, valor_default = carga_vocabulario()

    cadena = "<bos> hola <unk> mundo <sep> esta es una prueba <eos> <pad> <pad>"
    vcad = vectoriza_cadena(cadena, vocab_dict, valor_default)
    tcad = torch.tensor(vcad, dtype=torch.long)

    print(f'Cadena vectorizada: {vcad}')
    print(f'Tensor de cadena: {tcad}')