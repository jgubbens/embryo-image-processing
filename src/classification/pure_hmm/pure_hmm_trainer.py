import numpy as np
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classification.base_hmm_trainer import HMM_Trainer


class Pure_HMM(HMM_Trainer):

    def __init__(self, data_dir, device, window_size, preprocess_images=False, lstm_module=False, img_size=None):
        super().__init__(
            data_dir, device, window_size,
            preprocess_images=preprocess_images,
            lstm_module=lstm_module,
            img_size=img_size,
            augment_factor=0,
            model_name='pure_hmm',
        )

    def _additional_training(self):
        counts = np.zeros((self.n_states, self.n_states))
        for vid in self.train_vids:
            labels = list(vid.frame_labels.values())
            for t in range(1, len(labels)):
                counts[labels[t - 1], labels[t]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = counts / row_sums
        self.transition_matrix[0] = 0.0
        self.transition_matrix[0][0] = 0.5
        self.transition_matrix[0][1] = 0.5

    def _extra_model_info(self):
        return {'transition_matrix': self.transition_matrix.tolist()}

    def _load_extra_model_info(self, info):
        self.transition_matrix = np.array(info['transition_matrix'])

    def viterbi(self, obs_probs):
        T, n = obs_probs.shape
        log_trans = np.log(self.transition_matrix + 1e-10)
        log_emit = np.log(obs_probs + 1e-10)

        dp = np.full((T, n), -np.inf)
        backptr = np.zeros((T, n), dtype=int)

        # Uniform start distribution
        dp[0] = log_emit[0] - np.log(n)

        for t in range(1, T):
            candidates = dp[t - 1, :, None] + log_trans
            backptr[t] = np.argmax(candidates, axis=0)
            dp[t] = candidates[backptr[t], np.arange(n)] + log_emit[t]

        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(dp[-1])
        for t in range(T - 2, -1, -1):
            best_path[t] = backptr[t + 1, best_path[t + 1]]

        return float(dp[-1, best_path[-1]]), best_path.tolist()

    def _evaluate_sample(self, vid, ax=None):
        print(f'Inferring for sample {vid.vid_path}')

        labels = []
        all_probs = []

        model_label = 'LSTM' if self.lstm_module else 'CNN'
        model_color = 'tab:purple' if self.lstm_module else 'tab:green'

        if self.lstm_module:
            seq_probs, _ = self.lstm.predict_probs(vid)

        for t in range(len(vid)):
            frame, label = vid[t]
            labels.append(label)

            if self.lstm_module:
                model_probs = seq_probs[t]
            else:
                with torch.no_grad():
                    frame = frame.unsqueeze(0).to(self.device, dtype=torch.float16)
                    with torch.autocast('cuda'):
                        logits = self.cnn.model(frame)
                    model_probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()

            all_probs.append(model_probs)

        obs_probs = np.stack(all_probs)
        model_preds = np.argmax(obs_probs, axis=1).tolist()

        # Get preds for video up to every frame
        preds = []
        for t in range(len(vid)):
            _, path = self.viterbi(obs_probs[:t+1])
            preds.append(path[-1])
        preds = np.array(preds)

        labels = np.array(labels)
        model_preds = np.array(model_preds)

        self._plot_sample_progression(vid, labels, preds, model_preds, model_label, model_color, ax=ax)

        return labels, preds, model_preds


if __name__ == '__main__':

    DATA_PATH = r'E:\Justin\training_data'

    print('Running hidden markov model classification')
    DEVICE = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {DEVICE}')
    classifier = Pure_HMM(DATA_PATH, DEVICE, window_size=5, preprocess_images=False, lstm_module=False, img_size=(800, 800))

    classifier.train_hmm()
    # classifier.load_pretrained_models()
    # classifier.evaluate()
