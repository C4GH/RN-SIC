import numpy as np

def load_vocab_size(vocab_path):
    # Load vocabulary from a numpy file
    vocab = np.load(vocab_path, allow_pickle=True)
    vocab_size = len(vocab) # Get the size of the vocabulary
    return vocab_size

if __name__ == "__main__":
    vocab_path = r"C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\vocab_npa.npy"
    vocab_size = load_vocab_size(vocab_path)
    print("Vocabulary size: ", vocab_size)