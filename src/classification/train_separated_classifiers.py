import torch
import torch.nn as nn
from torchvision import models
import tifffile as tiff
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split

# Train classifiers for each individual nuclear cycle
# Detect whether or not embryo has moved past current stage of nuclear cycle

class IndividualClassifier:
    def __init__(self, current_cls, next_cls, root, DEVICE, img_size=(224, 224)):
        self.samples = []
        self.DEVICE = DEVICE
        self.img_size = img_size

        for cls_name, label in [(current_cls, 0), (next_cls, 1)]:
            cls_dir = Path(root, cls_name)
            if not cls_dir.exists():
                print(f'{cls_dir} does not exist')
                continue
            for f in sorted(cls_dir.glob('*.tif*')):
                self.samples.append((f, label))
        if not self.samples:
            raise RuntimeError(f'No TIFs found under {root}')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        raw = tiff.imread(str(path))
        img = raw[0] if raw.ndim == 3 else raw
        img = img.astype(np.float32)
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        tensor = torch.from_numpy(img).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=self.img_size,
            mode='bilinear', align_corners=False
        ).squeeze(0)
        return tensor, label
    
    def get_labels(self):
        return [s[1] for s in self.samples]
    
    def build_model(self):
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        w = m.conv1.weight.mean(dim=1, keepdim=True)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.conv1.weight = nn.Parameter(w)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m.to(DEVICE)
    
    def train_classifier(self, epochs=20, batch_size=16, lr=1e-4, val_split=0.2, patience=10):
        # Train/val split
        indices = list(range(len(self)))
        labels = self.get_labels()
        train_idx, val_idx = train_test_split(indices, test_size=val_split, stratify=labels, random_state=42)

        # Weighted sampler
        train_labels = [labels[i] for i in train_idx]
        class_counts = np.bincount(train_labels)
        weights = [1.0 / class_counts[l] for l in train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)

        train_loader = DataLoader(Subset(self, train_idx), batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(Subset(self, val_idx), batch_size=batch_size, shuffle=False)

        # Model
        model = self.build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        self.best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss, train_correct = 0.0, 0
            for imgs, lbls in train_loader:
                imgs = imgs.to(self.DEVICE)
                lbls = lbls.float().unsqueeze(1).to(self.DEVICE)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, lbls)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(imgs)
                train_correct += ((logits.sigmoid() > 0.5) == lbls.bool()).sum().item()

            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs = imgs.to(self.DEVICE)
                    lbls = lbls.float().unsqueeze(1).to(self.DEVICE)
                    logits = model(imgs)
                    val_loss += criterion(logits, lbls).item() * len(imgs)
                    val_correct += ((logits.sigmoid() > 0.5) == lbls.bool()).sum().item()

            train_loss /= len(train_idx)
            val_loss /= len(val_idx)
            train_acc = train_correct / len(train_idx)
            val_acc = val_correct / len(val_idx)
            scheduler.step(val_loss)

            print(
                f'Epoch {epoch:>3}/{epochs}:\t'
                f'Train loss {train_loss:.4f} acc {train_acc:.3f}\t'
                f'Val loss {val_loss:.4f} acc {val_acc:.3f}'
            )

            # Early stopping and checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

        # Restore best weights
        model.load_state_dict(best_state)
        self.model = model
        print(f'\nBest val loss: {self.best_val_loss:.4f}')
        return model
    

DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print(f'Device: {DEVICE}')
DATA_PATH = 'data/unprocessed_nc_binned'
MODEL_PATH = 'models'

results = []

print('\nNC9 Classifier:')
nc9_classifier = IndividualClassifier('NC9', 'NC9M', DATA_PATH, DEVICE)
print(f'Samples: {len(nc9_classifier)}  Labels: {nc9_classifier.get_labels()}')
nc9model = nc9_classifier.train_classifier(epochs=20, batch_size=16, lr=1e-4)
torch.save(nc9model.state_dict(), Path(MODEL_PATH, 'best_NC9.pt'))
results.append(('NC9', len(nc9_classifier), nc9_classifier.best_val_loss))

print('\nNC9M Classifier:')
nc9m_classifier = IndividualClassifier('NC9M', 'NC10', DATA_PATH, DEVICE)
print(f'Samples: {len(nc9m_classifier)}  Labels: {nc9m_classifier.get_labels()}')
nc9mmodel = nc9m_classifier.train_classifier(epochs=20, batch_size=16, lr=1e-4)
torch.save(nc9mmodel.state_dict(), Path(MODEL_PATH, 'best_NC9M.pt'))
results.append(('NC9M', len(nc9m_classifier), nc9m_classifier.best_val_loss))

print('\nNC10 Classifier:')
nc10_classifier = IndividualClassifier('NC10', 'NC10M', DATA_PATH, DEVICE)
print(f'Samples: {len(nc10_classifier)}  Labels: {nc10_classifier.get_labels()}')
nc10model = nc10_classifier.train_classifier(epochs=20, batch_size=16, lr=1e-4)
torch.save(nc10model.state_dict(), Path(MODEL_PATH, 'best_NC10.pt'))
results.append(('NC10', len(nc10_classifier), nc10_classifier.best_val_loss))

print('\nNC10M Classifier:')
nc10m_classifier = IndividualClassifier('NC10M', 'NC11', DATA_PATH, DEVICE)
print(f'Samples: {len(nc10m_classifier)}  Labels: {nc10m_classifier.get_labels()}')
nc10mmodel = nc10m_classifier.train_classifier(epochs=20, batch_size=16, lr=1e-4)
torch.save(nc10mmodel.state_dict(), Path(MODEL_PATH, 'best_NC10M.pt'))
results.append(('NC10M', len(nc10m_classifier), nc10m_classifier.best_val_loss))

print('\nNC11 Classifier:')
nc11_classifier = IndividualClassifier('NC11', 'NC11M', DATA_PATH, DEVICE)
print(f'Samples: {len(nc11_classifier)}  Labels: {nc11_classifier.get_labels()}')
nc11model = nc11_classifier.train_classifier(epochs=20, batch_size=16, lr=1e-4)
torch.save(nc11model.state_dict(), Path(MODEL_PATH, 'best_NC11.pt'))
results.append(('NC11', len(nc11_classifier), nc11_classifier.best_val_loss))

print('\nNC11M Classifier:')
nc11m_classifier = IndividualClassifier('NC11M', 'NC12', DATA_PATH, DEVICE)
print(f'Samples: {len(nc11m_classifier)}  Labels: {nc11m_classifier.get_labels()}')
nc11mmodel = nc11m_classifier.train_classifier(epochs=20, batch_size=16, lr=1e-4)
torch.save(nc11mmodel.state_dict(), Path(MODEL_PATH, 'best_NC11M.pt'))
results.append(('NC11M', len(nc11m_classifier), nc11m_classifier.best_val_loss))

print('\nNC12 Classifier:')
nc12_classifier = IndividualClassifier('NC12', 'NC12M', DATA_PATH, DEVICE)
print(f'Samples: {len(nc12_classifier)}  Labels: {nc12_classifier.get_labels()}')
nc12model = nc12_classifier.train_classifier(epochs=20, batch_size=16, lr=1e-4)
torch.save(nc12model.state_dict(), Path(MODEL_PATH, 'best_NC12.pt'))
results.append(('NC12', len(nc12_classifier), nc12_classifier.best_val_loss))

print('\nNC12M Classifier:')
nc12m_classifier = IndividualClassifier('NC12M', 'NC13', DATA_PATH, DEVICE)
print(f'Samples: {len(nc12m_classifier)}  Labels: {nc12m_classifier.get_labels()}')
nc12mmodel = nc12m_classifier.train_classifier(epochs=20, batch_size=16, lr=1e-4)
torch.save(nc12mmodel.state_dict(), Path(MODEL_PATH, 'best_NC12M.pt'))
results.append(('NC12M', len(nc12m_classifier), nc12m_classifier.best_val_loss))

print('\nNC13 Classifier:')
nc13_classifier = IndividualClassifier('NC13', 'NC13M', DATA_PATH, DEVICE)
print(f'Samples: {len(nc13_classifier)}  Labels: {nc13_classifier.get_labels()}')
nc13model = nc13_classifier.train_classifier(epochs=20, batch_size=16, lr=1e-4)
torch.save(nc13model.state_dict(), Path(MODEL_PATH, 'best_NC13.pt'))
results.append(('NC13', len(nc13_classifier), nc13_classifier.best_val_loss))

print('\nNC13M Classifier:')
nc13m_classifier = IndividualClassifier('NC13M', 'NC14+', DATA_PATH, DEVICE)
print(f'Samples: {len(nc13m_classifier)}  Labels: {nc13m_classifier.get_labels()}')
nc13mmodel = nc13m_classifier.train_classifier(epochs=20, batch_size=16, lr=1e-4)
torch.save(nc13mmodel.state_dict(), Path(MODEL_PATH, 'best_NC13M.pt'))
results.append(('NC13M', len(nc13m_classifier), nc13m_classifier.best_val_loss))

for name, samples, loss in results:
    print(f'{name}:\tSamples:{samples}\tLoss:{loss}')