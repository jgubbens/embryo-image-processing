import numpy as np
from scipy.stats import lognorm
import tifffile
import torch

from classification.base_hmm_predictor import HMM_Predictor


class Combined_HMM_Predictor(HMM_Predictor):

    def _load_extra_model_info(self, info):
        self.duration_model = {int(k): v for k, v in info['duration_model'].items()}
        self.transition_matrix = np.array(info['transition_matrix'])
        self.switch_state = np.array(info['switch_state'])

    def _initialize_extra_live_state(self):
        self.current_state = None
        self.frames_in_state = 0
        self.viterbi_log_dp = None
        self.hybrid_mode = False

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
        model_pred = np.argmax(model_probs)
        if model_pred >= self.switch_state:
            self.hybrid_mode = True
        if self.hybrid_mode:
            return self._select_state_hybrid(model_probs)
        else:
            return self._select_state_pure(model_probs)
    
    def _select_state_hybrid(self, model_probs):
        seconds_in_state = self.frames_in_state * self.time_between_frames
        duration_probs = self._get_duration_probs(self.current_state, seconds_in_state)

        combined = model_probs * duration_probs
        combined /= combined.sum() + 1e-9
        prediction = np.argmax(combined)
        prediction = max(prediction, self.current_state or 0)

        self.frames_in_state = self.frames_in_state + 1 if prediction == self.current_state else 1
        self.current_state = prediction
        return prediction
    
    def _select_state_pure(self, model_probs):
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

    predictor = Combined_HMM_Predictor(DEVICE, 'models/combined_hmm/combined_hmm_model_info.json', time_between_frames=60)

    # Test live classifier
    print('Testing live classifier')
    test_vid = tifffile.imread("data/training_data/processed_tifs/embryo3.tif")
    for frame in test_vid:
        predictor.predict_frame(torch.tensor(frame))
