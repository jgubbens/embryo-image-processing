# CNN classifying embryos by nuclear cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import models

class cnn_classifier:

    best_model_path = 'models/best_hmm_cnn.pt'

    def __init__(self, device, window_size, states):
        self.device = device
        self.window_size = window_size
        self.STATES = states
        self.hidden_size = 512
        self._build_model()

    def _build_model(self):
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze early layers
        for name, param in self.model.named_parameters():
            if any(name.startswith(p) for p in ['layer1', 'layer2', 'bn1']):
                param.requires_grad = False
            
        w = self.model.conv1.weight.mean(dim=1, keepdim=True) # (64, 1, 7, 7)
        w = w.repeat(1, self.window_size, 1, 1) # (64, 5, 7, 7)
        self.model.conv1 = nn.Conv2d(self.window_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1.weight = nn.Parameter(w)

        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.STATES))
        # Remove final classification layer
        # self.model.fc = nn.Identity()
        self.model.to(self.device)
        
        return self.model

    def remove_head(self):
        # Remove final layer to pass second last to LSTM
        self.model.fc = nn.Identity()
        self.model.to(self.device)
    
    def get_hidden_size(self):
        return self.hidden_size
    
    def train_model(self, train_vids, val_vids, best_model_path, epochs=30, batch_size=16, lr=0.0001):
        print('Training CNN...')
        dataset = ConcatDataset(train_vids)
        labels = [dataset[i][1] for i in range(len(dataset))]

        class_counts = np.bincount(labels, minlength=len(self.STATES)).astype(np.float32)
        class_weights = torch.tensor(1.0 / (class_counts + 0.000001)).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        val_dataset = ConcatDataset(val_vids)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss, train_correct = 0.0, 0
            for x, y in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(x)
                train_correct += (logits.argmax(dim=1) == y).sum().item()

            train_loss /= len(dataset)
            train_acc = train_correct / len(dataset)
            scheduler.step()

            # Validation loop
            self.model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)
                    loss = criterion(logits, y)
                    val_loss += loss.item() * len(x)
                    val_correct += (logits.argmax(dim=1) == y).sum().item()

            val_loss /= len(val_dataset)
            val_acc = val_correct / len(val_dataset)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_model_path)

            print(f'Epoch {epoch}/{epochs}: '
                f'train_loss {train_loss:.4f}, train_acc {train_acc:.3f} | '
                f'val_loss {val_loss:.4f}, val_acc {val_acc:.3f}')
            
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        print(f'Loaded best model from {best_model_path} (val_loss: {best_val_loss:.4f})')

    @torch.no_grad()
    def predict(self, x, return_probs=True):
        self.model.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        return (preds, probs) if return_probs else preds
    
    def evaluate(self, val_vids, batch_size=16, save_path='models/hmm_cnn_heatmap.png'):
        val_dataset = ConcatDataset(val_vids)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        labels = [val_dataset[i][1] for i in range(len(val_dataset))]
        class_counts = np.bincount(labels, minlength=len(self.STATES)).astype(np.float32)
        class_weights = torch.tensor(1.0 / (class_counts + 0.000001)).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.model.eval()
        val_loss, val_correct = 0.0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                val_loss += criterion(logits, y).item() * len(x)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        print(f'Evaluation: loss {val_loss:.4f}, accuracy {val_acc:.3f}')

        # Confusion matrix heatmap
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(self.STATES))))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=self.STATES, yticklabels=self.STATES, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (val_acc={val_acc:.3f})')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'Heatmap saved to {save_path}')

        return val_loss, val_acc
    
    def load_from_path(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

if __name__ == '__main__':
    cnn = cnn_classifier(5)
    cnn.predict()