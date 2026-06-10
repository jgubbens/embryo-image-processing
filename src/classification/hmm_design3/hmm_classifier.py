import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import torch
import torch.nn as nn
from embryo_video import hist_video
from pathlib import Path
import yaml
        
        
from cnn_classifier import cnn_classifier


class NeuralHMM:

    STATES = ['undetectable', 'NC9', 'NC9M', 'NC10', 'NC10M', 'NC11', 'NC11M', 'NC12', 'NC12M', 'NC13', 'NC13M', 'NC14+']

    def __init__(self, data_dir, device, window_size):
        self.data_dir = data_dir
        self.device = device
        self.n_states = len(self.STATES)
        self.window_size = window_size
        self.load_embryo_videos()
        self.cnn = cnn_classifier(self.device, window_size, self.STATES)
        self.hidden_size = self.cnn.get_hidden_size()
        self.linear_layer = nn.Linear(self.hidden_size, self.n_states)
    
    def load_embryo_videos(self):
        yaml_data = self._load_annotations()
        self.vids = []
        for embryo in yaml_data:
            vid_path = Path(self.data_dir, 'labeled_tifs', f'{embryo}.tif')
            self.vids.append(hist_video(yaml_data[embryo], vid_path, self.STATES, window_size=self.window_size, img_size=(800, 800)))

    def _load_annotations(self) -> dict:
        with open(Path(self.data_dir, 'labels.yaml')) as f:
            return yaml.safe_load(f)
        
    def _train_duration_model(self):
        durations = {i: [] for i in range(self.n_states)}
        for vid in self.vids:
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

    def _get_duration_probs(self, current_state, seconds_in_state):
        probs = np.zeros(self.n_states)

        if current_state is None:
            return np.ones(self.n_states) / self.n_states

        if current_state in self.duration_model:
            d = self.duration_model[current_state]
            p_stay = 1 - norm.cdf(seconds_in_state, d['mean'], d['std'])
            probs[current_state] = p_stay
            if current_state == 0:
                probs[0] = p_stay
                remaining = (1 - p_stay) / (self.n_states - 1)
                probs[1:] = remaining
            else:
                probs[current_state] = p_stay
                if current_state + 1 < self.n_states:
                    probs[current_state + 1] = 1 - p_stay
                else:
                    probs[current_state] = 1.0
        else:
            probs[current_state] = 1.0

        probs /= probs.sum() + 1e-9
        return probs
    
    def train_hmm(self):
        train_vids, val_vids = train_test_split(
            self.vids, test_size=0.2, random_state=42
        )

        self.cnn.train_model(train_vids, val_vids, best_model_path='models/best_hmm_cnn.pt', epochs=10, batch_size=16)
        #self.cnn.load_from_path('models/best_hmm_cnn.pt')
        self.cnn.evaluate(val_vids)
        self._train_duration_model()
        self.evaluate(val_vids)

    def evaluate_sample(self, vid, ax=None):
        print(f'Inferring for sample {vid.vid_path}')
        current_state = None
        frames_in_state = 0

        labels = []
        preds = []
        cnn_preds = []

        for t in range(len(vid)):
            frame, label = vid[t]

            with torch.no_grad():
                frame = frame.unsqueeze(0).to(self.device)
                logits = self.cnn.model(frame)
                cnn_probs = (torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
            )
            cnn_pred = np.argmax(cnn_probs)
            
            seconds_in_state = frames_in_state * vid.time_between_frames
            duration_probs = self._get_duration_probs(current_state, seconds_in_state)

            combined = cnn_probs * duration_probs
            combined /= combined.sum()
            prediction = np.argmax(combined)
            prediction = max(prediction, current_state or 0)

            labels.append(label)
            preds.append(prediction)
            cnn_preds.append(cnn_pred)

            #print(f'Prediction for frame {t}: {self.STATES[prediction]}\tTrue label: {self.STATES[label]}')

            if prediction == current_state:
                frames_in_state += 1
            else:
                current_state = prediction
                frames_in_state = 1

        labels = np.array(labels)
        preds = np.array(preds)
        cnn_preds = np.array(cnn_preds)

        if ax is not None:
            x = np.arange(len(labels))
            ax.step(
                x,
                labels,
                where='post',
                linewidth=2,
                color='tab:blue',
                label='True'
            )
            ax.step(
                x,
                preds,
                where='post',
                linewidth=2,
                color='tab:red',
                alpha=0.8,
                label='Predicted'
            )
            ax.step(
                x,
                cnn_preds,
                where='post',
                linewidth=2,
                color='tab:green',
                alpha=0.6,
                linestyle='--',
                label='CNN'
            )
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
            ax.step(
                x,
                labels,
                where='post',
                linewidth=2,
                color='tab:blue',
                label='True'
            )
            ax.step(
                x,
                preds,
                where='post',
                linewidth=2,
                color='tab:red',
                alpha=0.8,
                label='Predicted'
            )
            ax.step(
                x,
                cnn_preds,
                where='post',
                linewidth=2,
                color='tab:green',
                alpha=0.6,
                linestyle='--',
                label='CNN'
            )
            ax.set_title(Path(vid.vid_path).stem)
            ax.set_xlabel('Frame')
            ax.set_ylabel('State')
            ax.set_yticks(range(self.n_states))
            ax.set_yticklabels(self.STATES)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(
                f"models/{Path(vid.vid_path).stem}_state_progression.png",
                dpi=150
            )
            plt.close()

        return labels, preds, cnn_preds

    def evaluate(self, val_vids):
        all_preds = []
        all_labels = []
        all_cnn_preds = []

        # Progression plots
        n_vids = len(val_vids)
        ncols = 2
        nrows = int(np.ceil(n_vids / ncols))

        fig_progress, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(16, 4 * nrows),
            squeeze=False
        )
        axes = axes.flatten()

        for vid_idx, vid in enumerate(val_vids):
            labels, preds, cnn_preds = self.evaluate_sample(vid, ax=axes[vid_idx])

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_cnn_preds.extend(cnn_preds.tolist())

        for ax in axes[n_vids:]:
            ax.axis('off')

        fig_progress.tight_layout()
        fig_progress.savefig(
            'models/validation_state_progression_grid.png',
            dpi=150
        )
        plt.close(fig_progress)

        print('State progression grid saved to models/validation_state_progression_grid.png')

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_cnn_preds = np.array(all_cnn_preds)

        cnn_acc = (all_cnn_preds == all_labels).mean()
        hmm_acc = (all_preds == all_labels).mean()

        print(f'CNN only: {cnn_acc:.3f}')
        print(f'HMM: {hmm_acc:.3f}')

if __name__ == '__main__':

    print('Running hidden markov model classification')
    DEVICE = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {DEVICE}')
    classifier = NeuralHMM('data/hmm_tifs', DEVICE, window_size=1)

    classifier.train_hmm()