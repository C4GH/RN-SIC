import numpy as np
import torch


class TextPreprocessor:
    def __init__(self, vocab_path='salida/vocab_npa.npy'):
        self.vocab_dict, self.valor_default = self.carga_vocabulario(vocab_path)
cadena = "<bos> hola <unk> mundo <sep> esta es una prueba <eos> <pad> <pad>"
    vcad = preprocessor.vectoriza_cadena(cadena)
    tcad = torch.tensor(vcad, dtype=torch.long)

    print(f'Cadena vectorizada: {vcad}')
    print(f'Tensor de cadena: {tcad}')
    def carga_vocabulario(self, path):
        """
        Carga el vocabulario desde el .npy y crea un diccionario
        :param path: ruta al archivo
        :return: diccionario de vocabulario
        """
        vocab = np.load(path)
        list_vocab = vocab.tolist()
        vocab_dict = {word: index for index, word in enumerate(list_vocab)}
        valor_default = vocab_dict.get('<unk>', -1)
        return vocab_dict, valor_default

    def vectoriza_cadena(self, cad):
        """
        Convierte la cadena en una lista de índices

        :param cad: Cadena de texto
        :return: Lista de índices
        """
        cad = cad.lower().replace('.', '').replace(',', ' ').replace('  ', ' ').replace('-', ' ')
        aux = []
        for item in cad.split(" "):
            itaux = item.strip()
            if itaux:
                aux.append(self.vocab_dict.get(itaux, self.valor_default))
        return aux

