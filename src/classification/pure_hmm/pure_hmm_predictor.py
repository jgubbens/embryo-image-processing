import numpy as np
import tifffile
import torch

from classification.base_hmm_predictor import HMM_Predictor


class Pure_HMM_Predictor(HMM_Predictor):

    def _load_extra_model_info(self, info):
        self.transition_matrix = np.array(info['transition_matrix'])

    def _initialize_extra_live_state(self):
        self.viterbi_log_dp = None

    def _select_state(self, model_probs):
        log_emit = np.log(np.array(model_probs) + 1e-10)
        log_trans = np.log(self.transition_matrix + 1e-10)
        if self.viterbi_log_dp is None:
            self.viterbi_log_dp = log_emit - np.log(self.n_states)
        else:
            candidates = self.viterbi_log_dp[:, None] + log_trans
            self.viterbi_log_dp = np.max(candidates, axis=0) + log_emit
        return int(np.argmax(self.viterbi_log_dp))

if __name__ == "__main__":
    print('Running hidden markov model classification')
    DEVICE = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {DEVICE}')

    predictor = Pure_HMM_Predictor(DEVICE, 'models/pure_hmm/pure_hmm_model_info.json', time_between_frames=60)

    # Test live classifier
    print('Testing live classifier')
    test_vid = tifffile.imread("data/training_data/brightfield/embryo3.tif")
    # test_vid = tifffile.imread("data/hmm_tifs/processed_tifs/NCEmbryo34.tif")
    for frame in test_vid:
        predictor.predict_frame(torch.tensor(frame))
