import numpy as np
import torch


class Tokenizador:

    def __init__(self, archivo_vocab):
        vocab = np.load(archivo_vocab)
        self.lista_vocab = vocab.tolist()
        self.valor_default = np.where(vocab == '<unk>')[0][0]

    def vectoriza(self, cadena):
        """

        :param cadena:
        :return:
        """
        cadena = cadena.lower().replace('.', '').replace(',', ' ').replace('  ', ' ').replace('-', ' ')
        aux = []
        for item in cadena.split(" "):
            itaux = item.strip()
            if len(itaux) == 0:
                continue
            try:
                aux.append(self.lista_vocab.index(itaux))
            except ValueError as error:
                aux.append(self.valor_default)

        if len(aux) < 300:
            for i in range(300 - len(aux)):
                aux.append(self.lista_vocab.index('<pad>'))
        elif len(aux) > 300:
            aux = aux[:300]

        return torch.tensor(aux)