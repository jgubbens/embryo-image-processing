import numpy as np
from scipy.stats import norm
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classification.base_hmm_trainer import HMM_Trainer


class Hybrid_HMM(HMM_Trainer):

    def __init__(self, data_dir, device, window_size, preprocess_images=False, lstm_module=False, img_size=None, augment_factor=5):
        super().__init__(
            data_dir, device, window_size,
            preprocess_images=preprocess_images,
            lstm_module=lstm_module,
            img_size=img_size,
            augment_factor=augment_factor,
            model_name='hybrid_hmm',
        )

    def _additional_training(self):
        self._train_duration_model()

    def _train_duration_model(self):
        durations = {i: [] for i in range(self.n_states)}
        for vid in self.train_vids:
            labels = list(vid.frame_labels.values())
            current_state = labels[0]
            count = 1
            for t in range(1, len(labels)):
                if labels[t] == current_state:
                    count += 1
                else:
                    duration_seconds = count * vid.time_between_frames
                    durations[current_state].append(duration_seconds)
                    current_state = labels[t]
                    count = 1
            duration_seconds = count * vid.time_between_frames
            durations[current_state].append(duration_seconds)

        self.duration_model = {}
        for state, d in durations.items():
            if d:
                self.duration_model[state] = {'mean': np.mean(d), 'std': np.std(d) + 1e-6}

    def _extra_model_info(self):
        return {'duration_model': {str(state): stats for state, stats in self.duration_model.items()}}

    def _load_extra_model_info(self, info):
        self.duration_model = {int(k): v for k, v in info['duration_model'].items()}

    def _get_duration_probs(self, current_state, seconds_in_state):
        probs = np.zeros(self.n_states)

        if current_state is None or current_state == 0:
            return np.ones(self.n_states) / self.n_states

        if current_state in self.duration_model:
            d = self.duration_model[current_state]
            p_stay = 1 - norm.cdf(seconds_in_state, d['mean'], d['std'])
            probs[current_state] = p_stay
            if current_state + 1 < self.n_states:
                probs[current_state + 1] = 1 - p_stay
            else:
                probs[current_state] = 1.0
        else:
            probs[current_state] = 1.0

        probs /= probs.sum() + 1e-9
        return probs

    def _evaluate_sample(self, vid, ax=None):
        print(f'Inferring for sample {vid.vid_path}')
        current_state = None
        frames_in_state = 0

        labels = []
        preds = []
        model_preds = []

        model_label = 'LSTM' if self.lstm_module else 'CNN'
        model_color = 'tab:purple' if self.lstm_module else 'tab:green'

        if self.lstm_module:
            seq_probs, _ = self.lstm.predict_probs(vid)

        for t in range(len(vid)):
            frame, label = vid[t]

            if self.lstm_module:
                model_probs = seq_probs[t]
            else:
                with torch.no_grad():
                    frame = frame.unsqueeze(0).to(self.device, dtype=torch.float16)
                    with torch.autocast('cuda'):
                        logits = self.cnn.model(frame)
                    model_probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
            model_pred = np.argmax(model_probs)

            seconds_in_state = frames_in_state * vid.time_between_frames
            duration_probs = self._get_duration_probs(current_state, seconds_in_state)

            combined = model_probs * duration_probs
            combined /= combined.sum()
            prediction = np.argmax(combined)
            prediction = max(prediction, current_state or 0)

            labels.append(label)
            preds.append(prediction)
            model_preds.append(model_pred)

            if prediction == current_state:
                frames_in_state += 1
            else:
                current_state = prediction
                frames_in_state = 1

        labels = np.array(labels)
        preds = np.array(preds)
        model_preds = np.array(model_preds)

        self._plot_sample_progression(vid, labels, preds, model_preds, model_label, model_color, ax=ax)

        return labels, preds, model_preds


if __name__ == '__main__':

    DATA_PATH = r'data/training_data'

    print('Running hidden markov model classification')
    DEVICE = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {DEVICE}')
    classifier = Hybrid_HMM(DATA_PATH, DEVICE, window_size=10, preprocess_images=True, lstm_module=False, img_size=(800, 800), augment_factor=0)

    classifier.train_hmm()
    # classifier.load_pretrained_models()
    # classifier.evaluate()
