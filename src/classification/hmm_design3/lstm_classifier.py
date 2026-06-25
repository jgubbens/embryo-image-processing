import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from cnn_classifier import cnn_classifier

# LSTM to learn trends from CNN features

class lstm_classifier:

    best_model_path = 'models/best_hmm_lstm.pt'

    def __init__(self, hidden_size, device, states, cnn):
        self.hidden_size = hidden_size
        self.device = device
        self.STATES = states
        self.cnn = cnn
        self._build_model()
        

    def _build_model(self):
        self.model = nn.LSTM(
            input_size=512,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        ).to(self.device)

        self.fc = nn.Linear(self.hidden_size, len(self.STATES)).to(self.device)
    
    def _extract_features(self, cnn, vid_dataset):
        loader = DataLoader(vid_dataset, batch_size=64, shuffle=False)
        features, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                feat = cnn.model(x)
                features.append(feat)
                labels.append(y)
        features = torch.cat(features, dim=0).unsqueeze(0)
        labels = torch.cat(labels, dim=0).to(self.device)
        return features, labels

    def _forward(self, features):
        lstm_out, _ = self.model(features)
        logits = self.fc(lstm_out.squeeze(0))
        return logits
    
    def load_pretrained_cnn(self):
        # Load and freeze pretrained CNN
        self.cnn.model.eval()
        self.cnn.remove_head()
        for p in self.cnn.model.parameters():
            p.requires_grad = False

    def train_model(self, train_vids, val_vids, epochs=30, batch_size=16, lr=0.0001):
        print('Training LSTM...')

        self.load_pretrained_cnn()

        # Class weights from all training labels
        all_labels = []
        for vid in train_vids:
            all_labels += [vid[i][1] for i in range(len(vid))]
        class_counts = np.bincount(all_labels, minlength=len(self.STATES)).astype(np.float32)
        class_weights = torch.tensor(1.0 / (class_counts + 1e-6)).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        params = list(self.model.parameters()) + list(self.fc.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.fc.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for vid_dataset in tqdm(train_vids, desc=f"Epoch {epoch}/{epochs}"):
                features, labels = self._extract_features(self.cnn, vid_dataset)

                optimizer.zero_grad()
                logits = self._forward(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(labels)
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_total += len(labels)

            train_loss /= train_total
            train_acc = train_correct / train_total
            scheduler.step()

            # Validation
            self.model.eval()
            self.fc.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for vid_dataset in val_vids:
                    features, labels = self._extract_features(self.cnn, vid_dataset)
                    logits = self._forward(features)
                    val_loss += criterion(logits, labels).item() * len(labels)
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_total += len(labels)

            val_loss /= val_total
            val_acc = val_correct / val_total

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'lstm': self.model.state_dict(),
                            'fc':   self.fc.state_dict()}, self.best_model_path)

            print(f'Epoch {epoch}/{epochs}: '
                  f'train_loss {train_loss:.4f}, train_acc {train_acc:.3f} | '
                  f'val_loss {val_loss:.4f}, val_acc {val_acc:.3f}')

        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['lstm'])
        self.fc.load_state_dict(checkpoint['fc'])
        print(f'Loaded best LSTM from {self.best_model_path} (val_loss: {best_val_loss:.4f})')
    
    def load_from_path(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['lstm'])
        self.fc.load_state_dict(checkpoint['fc'])

    @torch.no_grad()
    def predict_probs(self, vid_dataset):
        # cnn.model must already have its head removed (see load_pretrained_cnn)
        self.model.eval()
        self.fc.eval()
        features, labels = self._extract_features(self.cnn, vid_dataset)
        logits = self._forward(features)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs, labels.cpu().numpy()
    
    def evaluate(self, val_vids, save_path='models/hmm_lstm_heatmap.png'):
        all_labels_flat = []
        for vid in val_vids:
            all_labels_flat += [vid[i][1] for i in range(len(vid))]
        class_counts = np.bincount(all_labels_flat, minlength=len(self.STATES)).astype(np.float32)
        class_weights = torch.tensor(1.0 / (class_counts + 1e-6)).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.model.eval()
        self.fc.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for vid_dataset in val_vids:
                features, labels = self._extract_features(self.cnn, vid_dataset)
                logits = self._forward(features)
                val_loss += criterion(logits, labels).item() * len(labels)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f'Evaluation: loss {val_loss:.4f}, accuracy {val_acc:.3f}')

        # Confusion matrix heatmap
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(self.STATES))))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.STATES, yticklabels=self.STATES, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'LSTM Confusion Matrix (val_acc={val_acc:.3f})')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'Heatmap saved to {save_path}')

        return val_loss, val_acc