import numpy as np
from scipy.stats import lognorm
import tifffile
import torch

from classification.base_hmm_predictor import HMM_Predictor


class Hybrid_HMM_Predictor(HMM_Predictor):

    def _load_extra_model_info(self, info):
        self.duration_model = {int(k): v for k, v in info['duration_model'].items()}

    def _initialize_extra_live_state(self):
        self.current_state = None
        self.frames_in_state = 0

    def _get_duration_probs(self, current_state, seconds_in_state):
        probs = np.zeros(self.n_states)

        if current_state is None or current_state == 0:
            return np.ones(self.n_states) / self.n_states

        if current_state in self.duration_model:
            d = self.duration_model[current_state]
            p_stay = 1 - lognorm.cdf(seconds_in_state, d['std'], scale=np.exp(d['mean']))
            probs[current_state] = p_stay
            if current_state + 1 < self.n_states:
                probs[current_state + 1] = 1 - p_stay
            else:
                probs[current_state] = 1.0
        else:
            probs[current_state] = 1.0

        probs /= probs.sum() + 1e-9
        return probs

    def _select_state(self, model_probs):
        seconds_in_state = self.frames_in_state * self.time_between_frames
        duration_probs = self._get_duration_probs(self.current_state, seconds_in_state)

        combined = model_probs * duration_probs
        combined /= combined.sum() + 1e-9
        prediction = np.argmax(combined)
        prediction = max(prediction, self.current_state or 0)

        self.frames_in_state = self.frames_in_state + 1 if prediction == self.current_state else 1
        self.current_state = prediction
        return prediction

if __name__ == "__main__":
    print('Running hidden markov model classification')
    DEVICE = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {DEVICE}')

    predictor = Hybrid_HMM_Predictor(DEVICE, 'models/hybrid_hmm/hybrid_hmm_model_info.json', time_between_frames=60)

    # Test live classifier
    print('Testing live classifier')
    test_vid = tifffile.imread("data/training_data/brightfield/embryo3.tif")
    # test_vid = tifffile.imread("data/hmm_tifs/processed_tifs/NCEmbryo34.tif")
    for frame in test_vid:
        predictor.predict_frame(torch.tensor(frame))
