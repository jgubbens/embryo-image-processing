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

from cnn_classifier import cnn_classifier
from embryo_video import embryo_video
from processing.extract_embryo import extract_embryo


class NeuralHMM:

    STATES = ['undetectable', 'NC9', 'NC9M', 'NC10', 'NC10M', 'NC11', 'NC11M', 'NC12', 'NC12M', 'NC13', 'NC13M', 'NC14+']

    def __init__(self, data_dir, device, window_size, preprocess_images=False):
        self.data_dir = data_dir
        self.device = device
        self.n_states = len(self.STATES)
        self.window_size = window_size
        if preprocess_images:
            self.process_training_data()
        self.load_embryo_videos(processed=preprocess_images)
        self.cnn = cnn_classifier(self.device, window_size, self.STATES)
        self.hidden_size = self.cnn.get_hidden_size()
    
    def train_hmm(self):
        train_vids, val_vids = train_test_split(
            self.vids, test_size=0.2, random_state=1
        )
        print(f'Validation vids: {[embryo.vid_path for embryo in val_vids]}')
        self.cnn.train_model(train_vids, val_vids, best_model_path='models/best_hmm_cnn.pt', epochs=10, batch_size=16)
        self.cnn.model.eval()
        self.cnn.evaluate(val_vids)
        self._train_duration_model(train_vids)
        self.save_duration_model()
        self.evaluate(val_vids)

    def load_pretrained_models(self):
        self.cnn.load_from_path('models/best_hmm_cnn.pt')
        self.cnn.model.eval()
        self.load_duration_model('models/duration_model.json')
    
    def load_embryo_videos(self, processed):
        yaml_data = self._load_annotations()
        self.vids = []
        for embryo in yaml_data:
            if processed:
                vid_path = Path(self.data_dir, 'processed_tifs', f'{embryo}.tif')
            else:
                #vid_path = Path(self.data_dir, 'labeled_tifs', f'{embryo}.tif')
                vid_path = Path(self.data_dir, 'histone', f'{embryo}.tif')
                # vid_path = Path(self.data_dir, 'brightfield', f'{embryo}.tif')
            self.vids.append(embryo_video(yaml_data[embryo], vid_path, self.STATES, window_size=self.window_size, img_size=(800, 800)))

    def _load_annotations(self) -> dict:
        with open(Path(self.data_dir, 'labels.yaml')) as f:
            return yaml.safe_load(f)
        
    def _train_duration_model(self, vids):
        durations = {i: [] for i in range(self.n_states)}
        for vid in vids:
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

    def save_duration_model(self, path='models/duration_model.json'):
        with open(path, 'w') as f:
            json.dump(self.duration_model, f, indent=2)

    def load_duration_model(self, path='models/duration_model.json'):
        with open(path) as f:
            raw = json.load(f)
        self.duration_model = {int(k): v for k, v in raw.items()}

    def _get_duration_probs(self, current_state, seconds_in_state):
        probs = np.zeros(self.n_states)

        if current_state is None:
            return np.ones(self.n_states) / self.n_states

        if current_state in self.duration_model:
            d = self.duration_model[current_state]
            p_stay = 1 - norm.cdf(seconds_in_state, d['mean'], d['std'])
            # Only allow undetectable to jump to NC9
            probs[current_state] = p_stay
            if current_state + 1 < self.n_states:
                probs[current_state + 1] = 1 - p_stay
            else:
                probs[current_state] = 1.0
            # Allow undetectable to jump to any state
            # if current_state == 0:
            #     probs[0] = p_stay
            #     remaining = (1 - p_stay) / (self.n_states - 1)
            #     probs[1:] = remaining
            # else:
            #     probs[current_state] = p_stay
            #     if current_state + 1 < self.n_states:
            #         probs[current_state + 1] = 1 - p_stay
            #     else:
            #         probs[current_state] = 1.0
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
            ax.step(x, labels, where='post', linewidth=2, color='tab:blue', label='True')
            ax.step(x, preds, where='post', linewidth=2, color='tab:red', alpha=0.8, label='Predicted')
            ax.step(x, cnn_preds, where='post', linewidth=2, color='tab:green', alpha=0.6, linestyle='--', label='CNN')
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
            ax.step(x, cnn_preds, where='post', linewidth=2, color='tab:green', alpha=0.6, linestyle='--', label='CNN')
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
            labels, preds, cnn_preds = self._evaluate_sample(vid, ax=axes[vid_idx])

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

        # Confusion matrix heatmap
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(self.STATES))))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=self.STATES, yticklabels=self.STATES, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (accuracy={hmm_acc:.3f})')
        plt.tight_layout()
        heatmap_path = 'models/hmm_heatmap.png'
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f'Heatmap saved to {heatmap_path}')

        print(f'CNN:\taccuracy: {cnn_acc:.3f}')
        print(f'HMM:\taccuracy: {hmm_acc:.3f}')

    def make_predictions_vid(self, video_path):
        vid = tifffile.open(video_path)
        # Need to make predictions on an embryo video without labels

    def initialize_live_prediction(self, time_between_frames=150, img_size=(800, 800)):
        self.live = True
        self.current_state = None
        self.time_between_frames = time_between_frames
        self.frames_in_state = 0
        self.frame_idx = 0
        self.predictions = []
        self.live_img_size = img_size
        self.frame_buffer = []

    def _preprocess_live_frame(self, frame_np):
        self.frame_buffer.append(frame_np)
        if len(self.frame_buffer) > self.window_size:
            self.frame_buffer.pop(0)
        pad = self.window_size - len(self.frame_buffer)
        window = np.stack(
            [np.zeros_like(self.frame_buffer[0])] * pad + list(self.frame_buffer), axis=0
        ).astype(np.float32)
        for i in range(window.shape[0]):
            vmin, vmax = window[i].min(), window[i].max()
            if vmax > vmin:
                window[i] = (window[i] - vmin) / (vmax - vmin)
        tensor = torch.from_numpy(window).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor, size=self.live_img_size, mode='bilinear', align_corners=False
        )
        return tensor

    def predict_frame(self, frame):
        frame_np = frame.numpy() if isinstance(frame, torch.Tensor) else frame

        with torch.no_grad():
            x = self._preprocess_live_frame(frame_np).to(self.device)
            logits = self.cnn.model(x)
            cnn_probs = (torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
        )
        cnn_pred = np.argmax(cnn_probs)
        
        seconds_in_state = self.frames_in_state * self.time_between_frames
        duration_probs = self._get_duration_probs(self.current_state, seconds_in_state)

        combined = cnn_probs * duration_probs
        combined /= combined.sum()
        prediction = np.argmax(combined)
        prediction = max(prediction, self.current_state or 0)

        print(f'Frame {self.frame_idx}:\tHMM Prediction: {self.STATES[prediction]}\tCNN Prediction: {self.STATES[cnn_pred]}')

        # Update current state
        self.frames_in_state = self.frames_in_state + 1 if prediction == self.current_state else 1
        self.current_state = prediction
        self.predictions.append(prediction)
        self.frame_idx += 1
    
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

    print('Running hidden markov model classification')
    DEVICE = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {DEVICE}')
    # classifier = NeuralHMM('data/hmm_tifs', DEVICE, window_size=1, preprocess_images=True)
    classifier = NeuralHMM('data/training_data', DEVICE, window_size=1, preprocess_images=True)

    classifier.train_hmm()
    # classifier.load_pretrained_models()

    # Test live classifier
    # print('Testing live classifier')
    # classifier.initialize_live_prediction(time_between_frames=150)
    # # test_vid = tifffile.imread("data/hmm_tifs/unlabeled_tifs/inner.i12_channel_638_patterns.tif")
    # test_vid = tifffile.imread("data/hmm_tifs/labeled_tifs/NCEmbryo2.tif")
    # for frame in test_vid:
    #     classifier.predict_frame(torch.tensor(frame))