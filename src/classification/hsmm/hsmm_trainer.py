import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import shutil
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import tifffile
import torch
import torch.nn as nn
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classification.cnn_classifier import cnn_classifier
from classification.lstm_classifier import lstm_classifier
from classification.embryo_video import embryo_video
from processing.extract_embryo import extract_embryo


class HSMM_Trainer:

    STATES = ['undetectable', 'NC9', 'NC9M', 'NC10', 'NC10M', 'NC11', 'NC11M', 'NC12', 'NC12M', 'NC13', 'NC13M', 'NC14+']

    def __init__(self, data_dir, device, window_size, preprocess_images=False, lstm_module=False, img_size=None, max_duration=300):
        self.data_dir = data_dir
        self.device = device
        self.n_states = len(self.STATES)
        self.window_size = window_size
        self.lstm_module = lstm_module
        self.preprocess_images = preprocess_images
        self.max_duration = max_duration
        if self.preprocess_images and img_size is None:
            raise ValueError('img_size must be provided when preprocess_images is True')
        self.img_size = img_size or (800, 800)
        if preprocess_images:
            self.process_training_data()
        self.load_embryo_videos(processed=preprocess_images)
        self.cnn = cnn_classifier(self.device, window_size, self.STATES)
        self.cnn.best_model_path = 'models/hsmm/hsmm_cnn.pt'
        if lstm_module:
            self.hidden_size = self.cnn.get_hidden_size()
            self.lstm = lstm_classifier(self.hidden_size, self.device, self.STATES, self.cnn)
            self.lstm.best_model_path = 'models/hsmm/hsmm_lstm.pt'
    
    def train_hmm(self):
        Path('models/hsmm').mkdir(parents=True, exist_ok=True)
        self.train_vids, self.val_vids = train_test_split(
            self.vids, test_size=0.2, random_state=1
        )
        print(f'Validation vids: {[embryo.vid_path for embryo in self.val_vids]}')
        # self.cnn.train_model(self.train_vids, self.val_vids, best_model_path=self.cnn.best_model_path, epochs=10, batch_size=16)
        info = self.load_model_info()
        self.cnn.load_from_path(info['cnn_model_path'])
        self.cnn.model.eval()
        self.cnn.evaluate(self.val_vids)
        if self.lstm_module:
            self.lstm.train_model(self.train_vids, self.val_vids, epochs=10, batch_size=16, lr=0.0001)
            self.lstm.evaluate(self.val_vids)
        self.create_transition_matrix()
        print('Fitting duration distributions:')
        self.create_duration_distributions()
        self.save_model_info()
        self.evaluate()

    def load_pretrained_models(self):
        info = self.load_model_info()
        self.transition_matrix = np.array(info['transition_matrix'])
        self.duration_params = {
            int(k): tuple(v) if v is not None else None
            for k, v in info['duration_params'].items()
        }
        self.cnn.load_from_path(info['cnn_model_path'])
        self.cnn.model.eval()
        if self.lstm_module:
            self.cnn.remove_head()
            self.lstm.load_from_path(info['lstm_model_path'])
        val_paths = set(info['val_vid_paths'])
        train_paths = set(info['train_vid_paths'])
        self.val_vids = [vid for vid in self.vids if str(vid.vid_path) in val_paths]
        self.train_vids = [vid for vid in self.vids if str(vid.vid_path) in train_paths]
    
    def load_embryo_videos(self, processed):
        yaml_data = self._load_annotations()
        self.vids = []
        for embryo in yaml_data:
            if processed:
                vid_path = Path(self.data_dir, 'processed_tifs', f'{embryo}.tif')
            else:
                #vid_path = Path(self.data_dir, 'labeled_tifs', f'{embryo}.tif')
                # vid_path = Path(self.data_dir, 'histone', f'{embryo}.tif')
                vid_path = Path(self.data_dir, 'brightfield', f'{embryo}.tif')
                # vid_path = Path(self.data_dir, 'processed_tifs', f'{embryo}.tif')
            self.vids.append(embryo_video(yaml_data[embryo], vid_path, self.STATES, window_size=self.window_size, img_size=self.img_size))

    def _load_annotations(self) -> dict:
        with open(Path(self.data_dir, 'labels.yaml')) as f:
            return yaml.safe_load(f)
    
    def create_transition_matrix(self):
        counts = np.zeros((self.n_states, self.n_states))
        for vid in self.train_vids:
            labels = list(vid.frame_labels.values())
            for t in range(1, len(labels)):
                counts[labels[t - 1], labels[t]] += 1
        # Zero diagonal — duration handles self-repetition in HSMM
        np.fill_diagonal(counts, 0)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = counts / row_sums
        # Undetectable has no duration constraint and can only enter NC9
        self.transition_matrix[0] = 0.0
        self.transition_matrix[0][1] = 1.0
        return self.transition_matrix

    def create_duration_distributions(self):
        """Fit Gaussian duration distributions for states 1-11; skip state 0 (undetectable)."""
        dwell_times = {s: [] for s in range(self.n_states)}
        for vid in self.train_vids:
            labels = list(vid.frame_labels.values())
            i = 0
            while i < len(labels):
                s = labels[i]
                j = i
                while j < len(labels) and labels[j] == s:
                    j += 1
                dwell_times[s].append(j - i)
                i = j

        self.duration_params = {}
        for s in range(self.n_states):
            if s == 0:
                self.duration_params[s] = None  # flat prior — no duration constraint
                continue
            dwells = dwell_times[s]
            if len(dwells) >= 2:
                mu = float(np.mean(dwells))
                sigma = max(float(np.std(dwells)), 1.0)
                self.duration_params[s] = (mu, sigma)
                print(f'  {self.STATES[s]}: n={len(dwells)}, mean={mu:.1f}, std={sigma:.1f}')
            else:
                self.duration_params[s] = None
                if dwells:
                    print(f'  {self.STATES[s]}: only {len(dwells)} sample(s), skipping duration fit')

    def viterbi_hsmm(self, obs_probs):
        T, n = obs_probs.shape
        max_d = min(self.max_duration, T)

        log_emit = np.log(obs_probs + 1e-10)               # (T, n)
        log_trans = np.log(self.transition_matrix + 1e-10) # (n, n)
        cum_emit = np.cumsum(log_emit, axis=0)              # (T, n)

        # Precompute duration log-probs: log_dur[j, d] for d in 1..max_d
        log_dur = np.zeros((n, max_d + 1))  # index 0 unused
        for j in range(n):
            params = self.duration_params.get(j)
            if params is not None:
                mu, sigma = params
                log_dur[j, 1:] = norm.logpdf(np.arange(1, max_d + 1), mu, sigma)
            # else: remains 0.0 — flat prior (no duration constraint)

        dp = np.full((T, n), -np.inf)
        bp_state = np.full((T, n), -1, dtype=int)  # -1 = start of sequence
        bp_time = np.full((T, n), -1, dtype=int)   # -1 = start of sequence

        # Init: all segments starting at t=0
        for d in range(1, max_d + 1):
            t_end = d - 1
            if t_end >= T:
                break
            seg = cum_emit[t_end] + log_dur[:, d]  # (n,)
            scores = -np.log(n) + seg
            update = scores > dp[t_end]
            dp[t_end, update] = scores[update]
            # bp stays -1/-1: sentinel for "segment starts at t=0"

        # Fill DP: t is the end of a segment, d is its duration
        for t in range(1, T):
            for d in range(1, min(max_d, t) + 1):
                t_prev = t - d  # end of the preceding segment
                # Sum of log-emissions over [t_prev+1 .. t] plus duration score
                seg = cum_emit[t] - cum_emit[t_prev] + log_dur[:, d]  # (n,)
                # candidates[i, j] = dp[t_prev, i] + log_trans[i, j] + seg[j]
                candidates = dp[t_prev, :, None] + log_trans + seg[None, :]  # (n, n)
                np.fill_diagonal(candidates, -np.inf)  # no consecutive same-state segments
                best_i = np.argmax(candidates, axis=0)        # (n,)
                best_score = candidates[best_i, np.arange(n)] # (n,)
                update = best_score > dp[t]
                dp[t, update] = best_score[update]
                bp_state[t, update] = best_i[update]
                bp_time[t, update] = t_prev

        # Traceback
        frame_labels = np.zeros(T, dtype=int)
        s = int(np.argmax(dp[T - 1]))
        t = T - 1
        while t >= 0:
            prev_t = bp_time[t, s]
            prev_s = bp_state[t, s]
            t_start = 0 if prev_t < 0 else prev_t + 1
            frame_labels[t_start:t + 1] = s
            if prev_t < 0:
                break
            t = prev_t
            s = prev_s

        return float(dp[T - 1, int(np.argmax(dp[T - 1]))]), frame_labels.tolist()

    def save_model_info(self, path='models/hsmm/hsmm_model_info.json'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        info = {
            'window_size': self.window_size,
            'lstm_module': self.lstm_module,
            'preprocess_images': self.preprocess_images,
            'max_duration': self.max_duration,
            'cnn_model_path': self.cnn.best_model_path,
            'lstm_model_path': self.lstm.best_model_path if self.lstm_module else None,
            'train_vid_paths': list(set(str(vid.vid_path) for vid in self.train_vids)),
            'val_vid_paths': [str(vid.vid_path) for vid in self.val_vids],
            'transition_matrix': self.transition_matrix.tolist(),
            'duration_params': {
                str(s): list(params) if params is not None else None
                for s, params in self.duration_params.items()
            },
            'img_size': list(self.img_size),
        }
        with open(path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f'Model info saved to {path}')

    def load_model_info(self, path='models/hsmm/hsmm_model_info.json'):
        with open(path) as f:
            info = json.load(f)
        return info

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
                    frame = frame.unsqueeze(0).to(self.device)
                    logits = self.cnn.model(frame)
                    model_probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()

            all_probs.append(model_probs)

        obs_probs = np.stack(all_probs)
        model_preds = np.argmax(obs_probs, axis=1).tolist()

        # Get preds for video up to every frame
        preds = []
        for t in range(len(vid)):
            _, path = self.viterbi_hsmm(obs_probs[:t+1])
            preds.append(path[-1])
        preds = np.array(preds)

        labels = np.array(labels)
        preds = np.array(preds)
        model_preds = np.array(model_preds)

        if ax is not None:
            x = np.arange(len(labels))
            ax.step(x, labels, where='post', linewidth=2, color='tab:blue', label='True')
            ax.step(x, preds, where='post', linewidth=2, color='tab:red', alpha=0.8, label='Predicted')
            ax.step(x, model_preds, where='post', linewidth=2, color=model_color, alpha=0.6, linestyle='--', label=model_label)
            ax.set_title(Path(vid.vid_path).stem)
            ax.set_xlabel('Frame')
            ax.set_ylabel('State')
            ax.set_yticks(range(self.n_states))
            ax.set_yticklabels(self.STATES, fontsize=7)
            ax.grid(True, alpha=0.3)
            handles, labels_ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            x = np.arange(len(labels))
            ax.step(x, labels, where='post', linewidth=2, color='tab:blue', label='True')
            ax.step(x, preds, where='post', linewidth=2, color='tab:red', alpha=0.8, label='Predicted')
            ax.step(x, model_preds, where='post', linewidth=2, color=model_color, alpha=0.6, linestyle='--', label=model_label)
            ax.set_title(Path(vid.vid_path).stem)
            ax.set_xlabel('Frame')
            ax.set_ylabel('State')
            ax.set_yticks(range(self.n_states))
            ax.set_yticklabels(self.STATES)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            Path('models/hsmm').mkdir(parents=True, exist_ok=True)
            plt.savefig(
                f"models/hsmm/{Path(vid.vid_path).stem}_state_progression.png",
                dpi=150
            )
            plt.close()

        return labels, preds, model_preds

    def evaluate(self):
        all_preds = []
        all_labels = []
        all_model_preds = []

        model_label = 'LSTM' if self.lstm_module else 'CNN'

        # Progression plots
        n_vids = len(self.val_vids)
        ncols = 2
        nrows = int(np.ceil(n_vids / ncols))

        fig_progress, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(16, 4 * nrows),
            squeeze=False
        )
        axes = axes.flatten()

        for vid_idx, vid in enumerate(self.val_vids):
            labels, preds, model_preds = self._evaluate_sample(vid, ax=axes[vid_idx])

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_model_preds.extend(model_preds.tolist())

        for ax in axes[n_vids:]:
            ax.axis('off')

        fig_progress.tight_layout()
        fig_progress.savefig(
            'models/hsmm/validation_state_progression_grid.png',
            dpi=150
        )
        plt.close(fig_progress)

        print('State progression grid saved to models/hsmm/validation_state_progression_grid.png')

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_model_preds = np.array(all_model_preds)

        model_acc = (all_model_preds == all_labels).mean()
        hmm_acc = (all_preds == all_labels).mean()

        # Confusion matrix heatmap
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(self.STATES))))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=self.STATES, yticklabels=self.STATES, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (accuracy={hmm_acc:.3f})')
        plt.tight_layout()
        heatmap_path = 'models/hsmm/hsmm_heatmap.png'
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f'Heatmap saved to {heatmap_path}')

        if self.lstm_module:
            self.lstm.evaluate(self.val_vids)

        print(f'{model_label}:\taccuracy: {model_acc:.3f}')
        print(f'HSMM:\taccuracy: {hmm_acc:.3f}')
    
    def process_training_data(self):
        processed_dir = Path(self.data_dir, 'processed_tifs')
        processed_dir.mkdir(parents=True, exist_ok=True)

        yaml_data = self._load_annotations()

        for embryo in yaml_data:
            # vid_path = tifffile.imread(Path(self.data_dir, 'labeled_tifs', f'{embryo}.tif'))
            # vid_path = tifffile.imread(Path(self.data_dir, 'histone', f'{embryo}.tif'))
            vid_path = tifffile.imread(Path(self.data_dir, 'brightfield', f'{embryo}.tif'))
            output_path = processed_dir / f'{embryo}.tif'
            extract_embryo(vid_path, output_path=output_path)
            

if __name__ == '__main__':

    DATA_PATH = r'data/training_data'

    print('Running hidden semi-markov model classification')
    DEVICE = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {DEVICE}')
    classifier = HSMM_Trainer(DATA_PATH, DEVICE, window_size=5, preprocess_images=False, lstm_module=False, img_size=(800, 800))

    classifier.train_hmm()
    # classifier.load_pretrained_models()
    # classifier.evaluate()
