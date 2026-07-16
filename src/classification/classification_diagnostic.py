import torch

from .hybrid_hmm.hybrid_hmm_trainer import Hybrid_HMM
from .pure_hmm.pure_hmm_trainer import HMM_Trainer
from .hsmm import hsmm_trainer

if __name__ == '__main__':

    DATA_PATH = r'data/training_data'

    print('Running hidden markov model classification benchmarks')
    DEVICE = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {DEVICE}')

    print("Hybrid HMM:")
    hybrid_hmm = Hybrid_HMM(DATA_PATH, DEVICE, window_size=5, preprocess_images=False, lstm_module=False, img_size=(800, 800))
    hybrid_hmm.train_hmm()

    print("Pure HMM:")
    pure_hmm = HMM_Trainer(DATA_PATH, DEVICE, window_size=5, preprocess_images=False, lstm_module=False, img_size=(800, 800))
    pure_hmm.train_hmm()

    print("HSMM:")
    pure_hmm = hsmm_trainer(DATA_PATH, DEVICE, window_size=5, preprocess_images=False, lstm_module=False, img_size=(800, 800))
    pure_hmm.train_hmm()
