import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from embryo_video import hist_video
from pathlib import Path
import yaml
        
        
from cnn_classifier import cnn_classifier
from lstm_classifier import lstm_classifier
from forward_algorithm import OnlineForwardAlgorithm


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
    
    def _estimate_transition_matrix(self):
        T = np.zeros((self.n_states, self.n_states))
        for vid in self.vids:
            labels = list(vid.frame_labels.values())
            for i in range(len(labels) - 1):
                T[labels[i], labels[i+1]] += 1
        # Normalize rows to sum to 1
        T = T / (T.sum(axis=1, keepdims=True) + 1e-9)
        self.transition_matrix = T
    
    def train_hmm(self):
        train_vids, val_vids = train_test_split(
            self.vids, test_size=0.2, random_state=42
        )

        print([vid.vid_path for vid in val_vids])

        self._estimate_transition_matrix()
        #self.cnn.train_model(train_vids, val_vids, best_model_path='models/best_hmm_cnn.pt', epochs=10, batch_size=16)
        self.cnn.load_from_path('models/best_hmm_cnn.pt')
        self.cnn.evaluate(val_vids)
        #self.cnn.remove_head()
        #self.lstm = lstm_classifier(self.hidden_size, self.device, self.STATES, pretrained_cnn=self.cnn)
        #self.lstm.train_model(train_vids, val_vids, epochs=10, batch_size=16)
        #self.lstm.load_from_path('models/best_hmm_lstm.pt')
        #self.lstm.evaluate(val_vids)
        self._train_duration_model()
        self.evaluate_sample(val_vids[0])
        self.evaluate(val_vids)

    def evaluate_sample(self, vid):
        print(f'Inferring for sample {vid.vid_path}')
        current_state = None
        frames_in_state = 0

        for t in range(len(vid)):
            frame, label = vid[t]

            with torch.no_grad():
                frame = frame.unsqueeze(0).to(self.device)
                logits = self.cnn.model(frame)
                cnn_probs = (torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
            )

            seconds_in_state = frames_in_state * vid.time_between_frames
            duration_probs = self._get_duration_probs(current_state, seconds_in_state)

            # combined = lstm_probs * duration_probs
            combined = cnn_probs * duration_probs
            combined /= combined.sum()
            prediction = np.argmax(combined)

            prediction = max(prediction, current_state or 0)

            print(f'Prediction for frame {t}: {self.STATES[prediction]}\tTrue label: {self.STATES[label]}')
            #print(f'Probabilities: {combined}')

            if prediction == current_state:
                frames_in_state += 1
            else:
                current_state = prediction
                frames_in_state = 1

    def evaluate(self, val_vids):
        all_preds = []
        all_labels = []
        #all_lstm_preds = []
        all_cnn_preds = []

        for vid in val_vids:
            current_state = None
            frames_in_state = 0
            h = None

            for t in range(len(vid)):
                frame, label = vid[t]

                # with torch.no_grad():
                #     frame = frame.unsqueeze(0).to(self.device)
                #     features = self.lstm.pretrained_cnn.model(frame)
                #     lstm_out, h = self.lstm.model(features.unsqueeze(0), h)
                #     logits = self.lstm.fc(lstm_out.squeeze(0))
                #     lstm_probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()


                # lstm_pred = np.argmax(lstm_probs)
                # all_lstm_preds.append(lstm_pred)

                with torch.no_grad():
                    frame = frame.unsqueeze(0).to(self.device)
                    logits = self.cnn.model(frame)
                    cnn_probs = (torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
                )

                cnn_pred = np.argmax(cnn_probs)
                all_cnn_preds.append(cnn_pred)

                seconds_in_state = frames_in_state * vid.time_between_frames
                duration_probs = self._get_duration_probs(current_state, seconds_in_state)

                # combined = lstm_probs * duration_probs
                combined = cnn_probs * duration_probs
                combined /= combined.sum()
                prediction = np.argmax(combined)
                prediction = max(prediction, current_state or 0)

                all_preds.append(prediction)
                all_labels.append(label)

                if prediction == current_state:
                    frames_in_state += 1
                else:
                    current_state = prediction
                    frames_in_state = 1

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        # all_lstm_preds = np.array(all_lstm_preds)
        all_cnn_preds = np.array(all_cnn_preds)

        # lstm_acc = (all_lstm_preds == all_labels).mean()
        cnn_acc = (all_cnn_preds == all_labels).mean()
        hmm_acc = (all_preds == all_labels).mean()
        # print(f'LSTM only: {lstm_acc:.3f}')
        print(f'CNN only: {cnn_acc:.3f}')
        print(f'HMM: {hmm_acc:.3f}')

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.n_states)))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.STATES, yticklabels=self.STATES, ax=axes[0])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_title(f'HMM (acc={hmm_acc:.3f})')

        # cm_lstm = confusion_matrix(all_labels, all_lstm_preds, labels=list(range(self.n_states)))
        # cm_lstm_norm = cm_lstm.astype(float) / (cm_lstm.sum(axis=1, keepdims=True) + 1e-9)

        # sns.heatmap(cm_lstm_norm, annot=True, fmt='.2f', cmap='Blues',
        #             xticklabels=self.STATES, yticklabels=self.STATES, ax=axes[1])
        # axes[1].set_xlabel('Predicted')
        # axes[1].set_ylabel('True')
        # axes[1].set_title(f'LSTM only (acc={lstm_acc:.3f})')

        # plt.tight_layout()
        # plt.savefig('models/hmm_evaluation.png', dpi=150)
        # plt.close()
        # print('Heatmap saved to models/hmm_evaluation.png')

        # return hmm_acc, lstm_acc

        cm_lstm = confusion_matrix(all_labels, all_cnn_preds, labels=list(range(self.n_states)))
        cm_lstm_norm = cm_lstm.astype(float) / (cm_lstm.sum(axis=1, keepdims=True) + 1e-9)

        sns.heatmap(cm_lstm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.STATES, yticklabels=self.STATES, ax=axes[1])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].set_title(f'CNN only (acc={cnn_acc:.3f})')

        plt.tight_layout()
        plt.savefig('models/hmm_evaluation.png', dpi=150)
        plt.close()
        print('Heatmap saved to models/hmm_evaluation.png')

        return hmm_acc, cnn_acc

if __name__ == '__main__':

    print('Running hidden markov model classification')
    DEVICE = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {DEVICE}')
    classifier = NeuralHMM('data/hmm_tifs', DEVICE, window_size=1)

    # print(classifier.vids[0].get_frame_window(6))
    classifier.train_hmm()
    # print(classifier.cnn.predict(classifier.vids[0].get_frame_window(6)))