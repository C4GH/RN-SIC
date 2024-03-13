from modelo import Modelo as M
import numpy as np
import torch
import psutil
import time

ARCH_EMBD = 'C:\\Users\\afloresre\\Documents\\Cagh\\Red\\salida\\embs_npa800.npy'

def ejecuta_modelo(tokens, nheads, num_layers):
    modelo = M.Modelo(tokens, nheads, num_layers, 1627916, 100, ARCH_EMBD, 0.05)

    # Asegurarse de que los índices generados están dentro del rango permitido por el tamaño del vocabulario
    vocab_size = 1627916  # Tamaño del vocabulario
    src = torch.randint(low=0, high=vocab_size, size=(tokens,))

    proceso = psutil.Process()
    recursos = proceso.memory_info().rss  # Uso de memoria RSS
    total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

    print('forward')
    try:
        out = modelo.forward(src)
    except IndexError as e:
        print(f"Índice fuera de rango: {src}")
        print(f"Mensaje de error: {e}")
        raise

    return recursos, total_params, out.shape[0]

if __name__ == '__main__':
    ldexp = []
    emb_size = 100

    for toks in range(100, 500, 100):
        for nhead in range(2, 7):
            if emb_size % nhead == 0:
                for nl in range(2, 9):
                    start = time.time()
                    res = ejecuta_modelo(toks, nhead, nl)
                    teje = time.time() - start
                    ldexp.append([toks, nhead, nl, res[0], res[1], res[2], teje])
                    print(toks, nhead, nl, res)

    with open('./salidas/salida_p2_modelo.txt', 'w') as fw:
        fw.write('tokens,nheads,num_layers,recursos,total_params,out_shape,tiempo\n')
        for ld in ldexp:
            fw.write(",".join(str(c) for c in ld))
            fw.write('\n')
