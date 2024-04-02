from modelo import Modelo4 as M
import torch
# import resource
import psutil
import torch.nn as nn
import time
import sys

def ejecuta_modelo(tokens, nheads, num_layers, d_model, src_emb):
    modelo = M.Modelo(d_model, nheads, num_layers, tokens, 3)

    tam_modelo = sys.getsizeof(modelo)
    process = psutil.Process()
    recursos = process.memory_info().rss
    # recursos = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

    print('forward')
    out = modelo.forward(src_emb)

    return recursos, total_params, out.shape[0], tam_modelo


if __name__ == '__main__':

    ldexp = []
    for batch_z in range(1, 33, 2):
        for emb_z in range(100, 600, 100):
            for toks in range(200, 600, 100):
                src_emb = torch.rand(batch_z, toks, emb_z)
                for nhead in range(2, 5):
                    if emb_z % nhead == 0:
                        print(f'emb_z % nhead : {emb_z % nhead}({emb_z},{nhead}')
                        for nlayers in range(2,7):
                            start = time.time()
                            res = ejecuta_modelo(toks, nhead, nlayers, emb_z, src_emb)
                            teje = time.time() - start
                            ldexp.append([batch_z, emb_z, toks, nhead, nlayers, res[0], res[1], res[2], res[3], teje])
                            print(batch_z, emb_z, toks, nhead, nlayers, res)

        with open('./salidas/salida_p13_modelo4.txt', 'w') as fw:
            fw.write('batch_size,emb_size,tokens,nheads,num_layers,recursos,total_params,out_shape,tam_modelo,tiempo\n')
            for ld in ldexp:
                fw.write(",".join(str(c) for c in ld))
                fw.write('\n')

