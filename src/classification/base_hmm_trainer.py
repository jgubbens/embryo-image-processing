import copy
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
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


class HMM_Trainer:

    STATES = ['undetectable', 'NC9', 'NC9M', 'NC10', 'NC10M', 'NC11', 'NC11M', 'NC12', 'NC12M', 'NC13', 'NC13M', 'NC14+']

    def __init__(self, data_dir, device, window_size, preprocess_images=False, lstm_module=False, img_size=None, augment_factor=5, model_name='hybrid_hmm'):
        self.data_dir = data_dir
        self.device = device
        self.n_states = len(self.STATES)
        self.window_size = window_size
        self.lstm_module = lstm_module
        self.preprocess_images = preprocess_images
        if self.preprocess_images and img_size is None:
            raise ValueError('img_size must be provided when preprocess_images is True')
        self.img_size = img_size or (800, 800)
        self.augment_factor = augment_factor
        self.model_name = model_name
        self.model_path = Path('models', model_name)
        if preprocess_images:
            self.process_training_data()
        self.load_embryo_videos(processed=preprocess_images)
        self.cnn = cnn_classifier(self.device, window_size, self.STATES)
        self.cnn.best_model_path = Path(self.model_path, f'{model_name}_cnn.pt')
        if lstm_module:
            self.hidden_size = self.cnn.get_hidden_size()
            self.lstm = lstm_classifier(self.hidden_size, self.device, self.STATES, self.cnn)
            self.lstm.best_model_path = Path(self.model_path, f'{model_name}_lstm.pt')
    
    def train_hmm(self):
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        self.train_vids, self.val_vids = train_test_split(
            self.vids, test_size=0.2, random_state=1
        )
        print(f'Validation vids: {[embryo.vid_path for embryo in self.val_vids]}')
        self.augment_training_data()
        self.cnn.train_model(self.train_vids, self.val_vids, best_model_path=self.cnn.best_model_path, epochs=30, batch_size=32)
        # info = self.load_model_info()
        # self.cnn.load_from_path(info['cnn_model_path'])
        self.cnn.model.eval()
        self.cnn.evaluate(self.val_vids, save_path=Path(f'{self.model_path}', f'{self.model_name}_cnn_heatmap.png'))
        if self.lstm_module:
            self.lstm.train_model(self.train_vids, self.val_vids, epochs=30, batch_size=32, lr=0.0001)
            self.lstm.evaluate(self.val_vids, save_path=Path(f'{self.model_path}', f'{self.model_name}_lstm_heatmap.png'))
        self._train_duration_model()
        self.save_model_info()

    def load_pretrained_models(self):
        info = self.load_model_info()
        self.cnn.load_from_path(info['cnn_model_path'])
        self.cnn.model.eval()
        if self.lstm_module:
            self.cnn.remove_head()
            self.lstm.load_from_path(info['lstm_model_path'])
        val_paths = set(info['val_vid_paths'])
        train_paths = set(info['train_vid_paths'])
        self.val_vids = [vid for vid in self.vids if str(vid.vid_path) in val_paths]
        self.train_vids = [vid for vid in self.vids if str(vid.vid_path) in train_paths]
        self.augment_factor = info.get('augment_factor', 0)
    
    def load_embryo_videos(self, processed):
        yaml_data = self._load_annotations()
        self.vids = []
        for embryo in yaml_data:
            if processed:
                vid_path = Path(self.data_dir, 'processed_tifs', f'{embryo}.tif')
            else:
                # vid_path = Path(self.data_dir, 'histone', f'{embryo}.tif')
                vid_path = Path(self.data_dir, 'brightfield', f'{embryo}.tif')
                # vid_path = Path(self.data_dir, 'processed_tifs', f'{embryo}.tif')
            self.vids.append(embryo_video(yaml_data[embryo], vid_path, self.STATES, window_size=self.window_size, img_size=self.img_size))

    def _load_annotations(self) -> dict:
        with open(Path(self.data_dir, 'labels.yaml')) as f:
            return yaml.safe_load(f)

    def save_model_info(self, path=f'models/hybrid_hmm/hybrid_hmm_model_info.json'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        info = {
            'window_size': self.window_size,
            'lstm_module': self.lstm_module,
            'preprocess_images': self.preprocess_images,
            'cnn_model_path': self.cnn.best_model_path,
            'lstm_model_path': self.lstm.best_model_path if self.lstm_module else None,
            'train_vid_paths': list(set(str(vid.vid_path) for vid in self.train_vids)),
            'val_vid_paths': [str(vid.vid_path) for vid in self.val_vids],
            'img_size': list(self.img_size),
            'augment_factor': self.augment_factor,
        }
        with open(path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f'Model info saved to {path}')

    def load_model_info(self, path='models/hybrid_hmm/hybrid_hmm_model_info.json'):
        with open(path) as f:
            info = json.load(f)
        return info
    
    # To be overridden by subclasses
    def _make_frame_prediction(self, model_probs):
        prediction = np.argmax(model_probs)
        return prediction

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
                    frame = frame.unsqueeze(0).to(self.device)
                    logits = self.cnn.model(frame)
                    model_probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
            model_pred = np.argmax(model_probs)

            prediction = self._make_frame_prediction(model_probs)

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
            Path(self.model_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(
                Path(self.model_path, f"{Path(vid.vid_path).stem}_state_progression.png"),
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
            Path(self.model_path, 'validation_state_progression_grid.png'),
            dpi=150
        )
        plt.close(fig_progress)

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
        heatmap_path = Path(self.model_path, f'{self.model_name}_heatmap.png')
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f'Heatmap saved to {heatmap_path}')

        if self.lstm_module:
            heatmap_path = Path(self.model_path, f'{self.model_name}_lstm_heatmap.png')
            self.lstm.evaluate(self.val_vids, save_path=heatmap_path)

        print(f'{model_label}:\taccuracy: {model_acc:.3f}')
        print(f'HMM:\taccuracy: {hmm_acc:.3f}')
    
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
    
    def augment_training_data(self):
        if self.augment_factor == 0:
            return
        print(f'Augmenting data with a factor of {self.augment_factor}')
        originals = list(self.train_vids)
        for vid in originals:
            print(f'Augmenting video: {vid.vid_path}')
            for _ in range(self.augment_factor):
                augmented = copy.copy(vid)
                frames = vid.vid.astype(np.float32) 

                # Gaussian blur
                sigma = np.random.uniform(0.5, 1.5)
                frames = gaussian_filter(frames, sigma=(0, sigma, sigma))

                # Gaussian noise
                noise_std = np.random.uniform(0.001, 0.01) * 65535
                frames = frames + np.random.normal(0, noise_std, frames.shape)

                # Brightness
                brightness = np.random.uniform(0.8, 1.2)
                frames = frames * brightness

                # Contrast
                contrast = np.random.uniform(0.8, 1.2)
                mean = frames.mean()
                frames = mean + contrast * (frames - mean)

                augmented.vid = np.clip(frames, 0, 65535).astype(np.uint16)
                self.train_vids.append(augmented)
